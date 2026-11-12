# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic solver interface for (sparse) matrix solvers.

This interface is only supposed to be a basis to add matrix solvers to optlang. It
creates a thin layer between all optlang operations and the standard formulation of
LP, QP, and MILP problems.

To create a solver from this, you need to do the following:

1. Create a class that inherits from MatrixProblem and implement the `solve` method
   and optionally overwrite the `reset` and `still_valid` methods.

2. Create derived classes for Variable, Constraint, Objective, and Configuration.
   You don't need to overwrite anything. Those should work out of the box.

3. Create a derived class for Model and set the `ProblemClass` attribute to the class
   from (1) and the `status_map` to map the solver status to an Optlang status from
   `optlang.interface`.

For an example take a look at `optlang.osqp_interface`.
"""

import abc
import logging
import pickle
from collections import defaultdict
from typing import NamedTuple

from numpy import inf, array, concatenate, zeros
from scipy.sparse import csc_matrix

from optlang import available_solvers, interface, symbolics
from optlang.exceptions import SolverError
from optlang.expression_parsing import parse_optimization_expression
from optlang.symbolics import add, mul

log = logging.getLogger(__name__)

_STATUS_MAP = defaultdict(lambda: interface.UNDEFINED)

_LP_METHODS = ("auto", )

_QP_METHODS = ("auto", )

_TYPES = ("continuous", "binary", "integer")


class SparseProblem(NamedTuple):
    """A representation of the convex optimization problem in standard form.

    This defines the problem in the form.

    ..math::

      \\text{minimize }& \\frac{1}{2}x^T\\mathbf{P}x + q^Tx \\\\
      \\text{s.t.: }& bounds_{.0} \\leq \\mathbf{A} \\leq bounds_{.1} \\\\
      & vbounds_{.0} \\leq x \\leq vbounds_{.1} \\\\
      & \\{x_k \\in \\mathbb{N}^0 \\text{ if } integer_k = 1 \\}

    Attributes
    ----------
    P : csc_matrix
        A semidefinite positive matrix specifying the coefficients for the quadratic
        objective.
    q : array
        A vector specfiying the linear objective coefficients.
    A : csc_matrix
        The constraint matrix.
    bounds : array
        The lower and upper bounds corresponding to the rows in `A`.
    vbounds : array
        The lower and upper bounds of the variables `x`.
    integer : array
        Indicator vector denoting if `x[k]` is integer.
    """
    P : csc_matrix
    q : array
    A : csc_matrix
    bounds : array
    vbounds : array
    integer : array


class MatrixProblem(abc.ABC):
    """A concise representation of an LP/QP/MILP problem.

    This class assumes that the problem will be pretty much immutable. This is
    a small intermediate layer based on dictionaries that is fast to modify
    but can also be converted to a matrix formulation quickly.
    """

    def __init__(self, **kwargs):
        self.variables = set()
        self.integer_vars = set()
        self.constraints = set()
        self.constraint_coefs = dict()
        self.constraint_lbs = dict()
        self.constraint_ubs = dict()
        self.variable_lbs = dict()
        self.variable_ubs = dict()
        self.obj_linear_coefs = dict()
        self.obj_quadratic_coefs = dict()
        self.primals = {}
        self.cprimals = {}
        self.duals = {}
        self.vduals = {}
        self.obj_value = None
        self.direction = -1
        self.info = None
        self.status = None
        self._solution = None
        self.default_settings()

    def default_settings(self):
        """Set the default settings of the solver.

        Returns
        -------
        dict
            The default settings for the solver.
        """
        self.settings = {
            "presolve": True,
            "lp_method": "auto",
            "qp_method": "auto",
            "time_limit": 0,
            "primal_inf_tolerance": 1e-6,
            "dual_inf_tolerance": 1e-6,
            "optimality_tolerance": 1e-6,
            "mip_tolerance": 1e-6,
            "verbose": 1,
            "threads": 1,
            "iteration_limit": 100000
        }

    def build(self, add_variable_constraints=False):
        """Build the problem instance.

        Parameters
        ----------
        add_variable_constraints : bool
            Whether to add constraints for each individual variables as a (sparse)
            block diagonal to the constraint matrix A.

        Returns
        -------
        tuple of [P, q, A, bounds, vbounds, integer] with the following components:
            - P : Quadratic objective coefficients as an upper triangular CSC matrix
            - q : Linear objective coefficients as an array.
            - A : Constraint matrix as a CSC matrix.
            - bounds : Constraint lower and upper bounds as a 2xNc matrix.
            - vbounds : Variable bounds as a 2xNv matrix.
            - integer : Binary array indicating whether variable i is integer
                        (for MIPs).

        """
        d = self.direction
        vmap = dict(zip(self.variables, range(len(self.variables))))
        nv = len(self.variables)
        cmap = dict(zip(self.constraints, range(len(self.constraints))))
        nc = len(self.constraints)
        if len(self.obj_quadratic_coefs) > 0:
            P = array(
                [
                    [vmap[vn[0]], vmap[vn[1]], coef * d * 2.0]
                    for vn, coef in self.obj_quadratic_coefs.items()
                ]
            )
            P = csc_matrix(
                (P[:, 2], (P[:, 0].astype("int64"), P[:, 1].astype("int64"))),
                shape=(nv, nv),
            )
        else:
            P = None
        q = zeros(nv)
        q[[vmap[vn] for vn in self.obj_linear_coefs]] = list(
            self.obj_linear_coefs.values()
        )
        q = q * d
        vbounds = array(
            [[self.variable_lbs[vn], self.variable_ubs[vn]] for vn in self.variables]
        )

        if len(self.constraint_coefs) > 0:
            A = array(
                [
                    (cmap[cname], vmap[vname], co)
                    for cname, coefs in self.constraint_coefs.items()
                    for vname, co in coefs.items()
                ]
            )
            bounds = array(
                [
                    [self.constraint_lbs[cn], self.constraint_ubs[cn]]
                    for cn in self.constraints
                ]
            )
            if add_variable_constraints:
                Av = array([[vmap[k] + nc, vmap[k], 1.0] for k in self.variables])
                A = concatenate((A, Av))
                bounds = concatenate((bounds, vbounds))
            A = csc_matrix(
                (A[:, 2], (A[:, 0].astype("int64"), A[:, 1].astype("int64"))),
                shape=(bounds.shape[0], nv),
            )
        elif add_variable_constraints:
            Av = array([[vmap[k], vmap[k], 1.0] for k in self.variables])
            A = csc_matrix(
                (Av[:, 2], (Av[:, 0].astype("int64"), Av[:, 1].astype("int64"))),
                shape=(nv, nv),
            )
            bounds = vbounds
        else:
            A = None
            bounds = None

        integer = array([int(vn in self.integer_vars) for vn in self.variables])
        return SparseProblem(P, q, A, bounds, vbounds, integer)

    @abc.abstractmethod
    def solve(self):
        """Solve the problem.

        This will return nothing but will fill the solution attributes.

        Returns
        -------
        Nothing.
        """
        raise NotImplementedError("This needs to be overwritten by the child class.")

    def reset(self, everything=False):
        """Reset the public solver solution.

        Parameters
        ----------
        everything : bool
            If true will also clear cached solution bases or warm start solutions.

        Returns
        -------
        Nothing.
        """
        self.info = None
        self.primals = {}
        self.cprimals = {}
        self.duals = {}
        self.vduals = {}
        if everything:
            self._solution = None

    @abc.abstractmethod
    def still_valid(self, A, bounds):
        """Check if previous solutions is still feasible.

        Parameters
        ----------
        A : numpy.csc_matrix
            The constraint matrix A.
        bounds : numpy.array
            The constraint bounds.

        Returns
        -------
        bool
            Whether the current solution still fulfills the bounds.

        """
        raise NotImplementedError("This needs to be overwritten by the child class.")

    def prune(self):
        """Remove unused variables and constraints."""
        self.reset()
        self.variable_lbs = {
            k: v for k, v in self.variable_lbs.items() if k in self.variables
        }
        self.variable_ubs = {
            k: v for k, v in self.variable_ubs.items() if k in self.variables
        }
        self.constraint_coefs = {
            cname: {
                vname: coef
                for vname, coef in coefs.items()
                if vname in self.variables
            }
            for cname, coefs in self.constraint_coefs.items()
            if cname in self.constraints
        }
        self.obj_linear_coefs = {
            k: v for k, v in self.obj_linear_coefs.items() if k in self.variables
        }
        self.obj_quadratic_coefs = {
            k: v
            for k, v in self.obj_quadratic_coefs.items()
            if k[0] in self.variables and k[1] in self.variables
        }
        self.constraint_lbs = {
            k: v for k, v in self.constraint_lbs.items() if k in self.constraints
        }
        self.constraint_ubs = {
            k: v for k, v in self.constraint_ubs.items() if k in self.constraints
        }
        self.integer_vars = {v for v in self.integer_vars if v in self.variables}

    def rename_constraint(self, old, new):
        """Rename a constraint.

        Parameters
        ----------
        old : str
            The old name of the constraint.
        new : str
            The new name of the constraint.

        Returns
        -------
        Nothing.

        """
        self.reset()
        self.constraints.remove(old)
        self.constraints.add(new)
        coefs = self.constraint_coefs.pop(old)
        self.constraint_coefs[new] = coefs
        self.constraint_lbs[new] = self.constraint_lbs.pop(old)
        self.constraint_ubs[new] = self.constraint_ubs.pop(old)

    def rename_variable(self, old, new):
        """Rename a variable.

        Parameters
        ----------
        old : str
            The old name of the variable.
        new : str
            The new name of the variable.

        Returns
        -------
        Nothing.

        """
        self.reset()
        self.variables.remove(old)
        self.variables.add(new)
        name_map = {k: k for k in self.variables}
        name_map[old] = new
        for cname in self.constraint_coefs:
            coefs = self.constraint_coefs[cname]
            self.constraint_coefs[cname] = {
                name_map[k]: v for k, v in coefs.items()
            }
        self.obj_quadratic_coefs = {
            (name_map[k[0]], name_map[k[1]]): v
            for k, v in self.obj_quadratic_coefs.items()
        }
        self.obj_linear_coefs = {
            name_map[k]: v for k, v in self.obj_linear_coefs.items()
        }
        self.variable_lbs[new] = self.variable_lbs.pop(old)
        self.variable_ubs[new] = self.variable_ubs.pop(old)
        if old in self.integer_vars:
            self.integer_vars.remove(old)
            self.integer_vars.add(new)


class Variable(interface.Variable):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name=name, **kwargs)

    @interface.Variable.type.setter
    def type(self, value):
        if self.problem is not None:
            if value not in _TYPES:
                raise ValueError(
                    "This interface cannot handle variables of type '%s'. " % value
                    + "The following variable types are available: "
                    + ", ".join(_TYPES)
                )
            if value == "binary":
                self.lb = 0
                self.ub = 1
                self.problem.problem.integer_vars.add(self.name)
            elif value == "integer":
                self.problem.problem.integer_vars.add(self.name)
            elif value == "continuous":
                if self.name in self.problem.problem.integer_vars:
                    self.problem.problem.integer_vars.remove(self.name)
        super(Variable, Variable).type.fset(self, value)

    def _get_primal(self):
        return self.problem.problem.primals.get(self.name, None)

    @property
    def dual(self):
        if self.problem.is_integer:
            raise ValueError("Dual values are not well-defined for integer problems")
        if self.problem is not None:
            return self.problem.problem.vduals.get(self.name, None)
        return None

    @interface.Variable.name.setter
    def name(self, value):
        old_name = getattr(self, "name", None)
        super(Variable, Variable).name.fset(self, value)
        if getattr(self, "problem", None) is not None:
            if old_name != value:
                self.problem.problem.rename_variable(old_name, value)


class Constraint(interface.Constraint):
    _INDICATOR_CONSTRAINT_SUPPORT = False

    def __init__(self, expression, sloppy=False, **kwargs):
        super().__init__(expression=expression, sloppy=sloppy, **kwargs)

    def set_linear_coefficients(self, coefficients):
        if self.problem is not None:
            self.problem.update()
            self.problem.problem.reset()
            for var, coef in coefficients.items():
                self.problem.problem.constraint_coefs[self.name][var.name] = float(
                    coef
                )
        else:
            raise Exception(
                "Can't change coefficients if constraint is not associated "
                "with a model."
            )

    def get_linear_coefficients(self, variables):
        if self.problem is not None:
            self.problem.update()
            coefs = {
                v: self.problem.problem.constraint_coefs.get(self.name, {}).get(v.name, 0.0)
                for v in variables
            }
            return coefs
        else:
            raise Exception(
                "Can't get coefficients from solver if constraint is not " "in a model"
            )

    def _get_expression(self):
        if self.problem is not None:
            variables = self.problem._variables
            coefs = self.problem.problem.constraint_coefs[self.name]
            terms = [
                mul((symbolics.Real(coef), variables[vname]))
                for vname, coef in coefs.items()
            ]
            self._expression = add(terms)
        return self._expression

    @property
    def problem(self):
        return self._problem

    @problem.setter
    def problem(self, value):
        if value is None:
            # Update expression from solver instance one last time
            self._get_expression()
        self._problem = value

    @property
    def primal(self):
        if self.problem is None:
            return None
        return self.problem.problem.cprimals.get(self.name, None)

    @property
    def dual(self):
        if self.problem.is_integer:
            raise ValueError("Dual values are not well-defined for integer problems")
        if self.problem is None:
            return None
        d = self.problem.problem.duals.get(self.name, None)
        return d

    @interface.Constraint.name.setter
    def name(self, value):
        old_name = self.name
        super(Constraint, Constraint).name.fset(self, value)
        if getattr(self, "problem", None) is not None:
            if old_name != value:
                self.problem.problem.rename_constraint(old_name, value)

    @interface.Constraint.lb.setter
    def lb(self, value):
        self._check_valid_lower_bound(value)
        if getattr(self, "problem", None) is not None:
            lb = -inf if value is None else float(value)
            self.problem.problem.constraint_lbs[self.name] = lb
        self._lb = value

    @interface.Constraint.ub.setter
    def ub(self, value):
        self._check_valid_upper_bound(value)
        if getattr(self, "problem", None) is not None:
            ub = inf if value is None else float(value)
            self.problem.problem.constraint_ubs[self.name] = ub
        self._ub = value

    def __iadd__(self, other):
        # if self.problem is not None:
        #     self.problem._add_to_constraint(self.index, other)
        if self.problem is not None:
            problem_reference = self.problem
            self.problem._remove_constraint(self)
            super().__iadd__(other)
            problem_reference._add_constraint(self, sloppy=False)
        else:
            super().__iadd__(other)
        return self


class Objective(interface.Objective):
    def __init__(self, expression, sloppy=False, **kwargs):
        super().__init__(expression=expression, sloppy=sloppy, **kwargs)
        self._expression_expired = False
        if not (sloppy or self.is_Linear or self.is_Quadratic):
            raise ValueError(
                "This interface only supports linear and quadratic objectives.")

    @property
    def value(self):
        if getattr(self, "problem", None) is None:
            return None
        if self.problem.problem.obj_value is None:
            return None
        return self.problem.problem.obj_value + self.problem._objective_offset

    @interface.Objective.direction.setter
    def direction(self, value):
        super(Objective, Objective).direction.__set__(self, value)
        if value == "min":
            self.problem.problem.direction = 1
        else:
            self.problem.problem.direction = -1

    def _get_expression(self):
        if (
            self.problem is not None
            and self._expression_expired
            and len(self.problem._variables) > 0
        ):
            model = self.problem
            vars = model._variables
            expression = add(
                [
                    coef * vars[vn]
                    for vn, coef in model.problem.obj_linear_coefs.items()
                ]
            )
            q_ex = add(
                [
                    coef * vars[vn[0]] * vars[vn[1]]
                    for vn, coef in model.problem.obj_quadratic_coefs.items()
                ]
            )
            expression += q_ex
            self._expression = expression
            self._expression_expired = False
        return self._expression

    def set_linear_coefficients(self, coefficients):
        if self.problem is not None:
            self.problem.update()
            for v, coef in coefficients.items():
                self.problem.problem.obj_linear_coefs[v.name] = float(coef)
            self._expression_expired = True
        else:
            raise Exception(
                "Can't change coefficients if objective is not associated "
                "with a model."
            )

    def get_linear_coefficients(self, variables):
        if self.problem is not None:
            self.problem.update()
            coefs = {
                v: self.problem.problem.obj_linear_coefs.get(v.name, 0.0)
                for v in variables
            }
            return coefs
        else:
            raise Exception(
                "Can't get coefficients from solver if objective " "is not in a model"
            )


class Configuration(interface.MathematicalProgrammingConfiguration):
    lp_methods = _LP_METHODS
    qp_methods = _QP_METHODS

    def __init__(
        self,
        lp_method="auto",
        presolve=False,
        verbosity=0,
        timeout=None,
        qp_method="auto",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lp_method = lp_method
        self.presolve = presolve
        self.verbosity = verbosity
        self.timeout = timeout
        self.qp_method = qp_method
        if "tolerances" in kwargs:
            for key, val in kwargs["tolerances"].items():
                if key in self._tolerance_functions():
                    setattr(self.tolerances, key, val)

    @property
    def lp_method(self):
        """The algorithm used to solve LP problems."""
        return self.problem.problem.settings["lp_method"]

    @lp_method.setter
    def lp_method(self, value):
        if value not in self.lp_methods:
            raise ValueError(
                "LP Method %s is not valid (choose one of: %s)"
                % (value, ", ".join(self.lp_methods))
            )
        if getattr(self, "problem", None) is not None:
            self.problem.problem.settings["lp_method"] = value
        self._lp_method = value

    def _set_presolve(self, value):
        if getattr(self, "problem", None) is not None:
            if value in [True, False, "auto"]:
                self.problem.problem.settings["presolve"] = value
            else:
                raise ValueError(
                    str(value) + " is not a valid presolve parameter. Must be True, "
                    "False or 'auto'."
                )

    @property
    def presolve(self):
        return self.problem.problem.settings["presolve"]

    @presolve.setter
    def presolve(self, value):
        self._set_presolve(value)
        self._presolve = value

    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value):
        if getattr(self, "problem", None) is not None:
            self.problem.problem.settings["verbose"] = int(value)
        self._verbosity = value

    @property
    def timeout(self):
        """Time limit in seconds, or None when the solver is not limited."""
        # 0 is the stored sentinel for no limit. A numeric 0 copied onto GLPK
        # becomes a zero-millisecond limit.
        limit = self.problem.problem.settings["time_limit"]
        if limit == 0:
            return None
        return limit

    @timeout.setter
    def timeout(self, value):
        if getattr(self, "problem", None) is not None:
            if value is None:
                self.problem.problem.settings["time_limit"] = 0
            else:
                self.problem.problem.settings["time_limit"] = value

    def __getstate__(self):
        return {
            "presolve": self.presolve,
            "timeout": self.timeout,
            "verbosity": self.verbosity,
            "tolerances": {
                "feasibility": self.tolerances.feasibility,
                "optimality": self.tolerances.optimality,
                "integrality": self.tolerances.integrality,
            },
        }

    def __setstate__(self, state):
        for key, val in state.items():
            if key != "tolerances":
                setattr(self, key, val)
        for key, val in state["tolerances"].items():
            if key in self._tolerance_functions():
                setattr(self.tolerances, key, val)

    @property
    def qp_method(self):
        """Change the algorithm used to optimize QP problems."""
        return self.problem.problem.settings["qp_method"]

    @qp_method.setter
    def qp_method(self, value):
        if value not in _QP_METHODS:
            raise ValueError(
                "%s is not a valid qp_method. Choose between %s"
                % (value, str(self.qp_methods))
            )
        if getattr(self, "problem", None) is not None:
            self.problem.problem.settings["qp_method"] = value
        self._qp_method = value

    def _get_feasibility(self):
        return self.problem.problem.settings["primal_inf_tolerance"]

    def _set_feasibility(self, value):
        self.problem.problem.settings["primal_inf_tolerance"] = value
        self.problem.problem.settings["dual_inf_tolerance"] = value

    def _get_integrality(self):
        return self.problem.problem.settings["mip_tolerance"]

    def _set_integrality(self, value):
        self.problem.problem.settings["mip_tolerance"] = value

    def _get_optimality(self):
        return self.problem.problem.settings["optimality_tolerance"]

    def _set_optimality(self, value):
        self.problem.problem.settings["optimality_tolerance"] = value

    def _tolerance_functions(self):
        return {
            "feasibility": (self._get_feasibility, self._set_feasibility),
            "optimality": (self._get_optimality, self._set_optimality),
            "integrality": (self._get_integrality, self._set_integrality),
        }


class Model(interface.Model):
    ProblemClass = MatrixProblem
    status_map = _STATUS_MAP

    def _initialize_problem(self):
        self.problem = self.ProblemClass()

    def _initialize_model_from_problem(self, problem, vc_mapping=None, offset=0):
        if not isinstance(problem, self.ProblemClass):
            raise TypeError("Provided problem is not a valid OSQP model.")
        self.problem = problem
        for name in self.problem.variables:
            var = Variable(
                name,
                lb=self.problem.variable_lbs[name],
                ub=self.problem.variable_ubs[name],
                problem=self,
            )
            super()._add_variables([var])

        for name in self.problem.constraints:
            # Since constraint expressions are lazily retrieved from the
            # solver they don't have to be built here
            lhs = symbolics.Integer(0)
            constr = Constraint(
                lhs,
                lb=self.problem.constraint_lbs[name],
                ub=self.problem.constraint_ubs[name],
                name=name,
                problem=self,
            )

            super()._add_constraints([constr], sloppy=True)

        if vc_mapping is None:
            for constr in self.constraints:
                name = constr.name
                for variable in constr.variables:
                    try:
                        self._variables_to_constraints_mapping[variable.name].add(name)
                    except KeyError:
                        self._variables_to_constraints_mapping[variable.name] = set(
                            [name]
                        )
        else:
            self._variables_to_constraints_mapping = vc_mapping

        linear_expression = add(
            [
                coef * self._variables[vn]
                for vn, coef in self.problem.obj_linear_coefs.items()
            ]
        )
        quadratic_expression = add(
            [
                coef * self._variables[vn[0]] * self._variables[vn[1]]
                for vn, coef in self.problem.obj_quadratic_coefs.items()
            ]
        )

        self._objective_offset = offset
        self._objective = Objective(
            linear_expression + quadratic_expression + offset,
            problem=self,
            direction={-1: "max", 1: "min"}[self.problem.direction],
            name="matrix_objective",
        )

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, value):
        value.problem = None
        if self._objective is not None:  # Reset previous objective
            self.problem.obj_linear_coefs = dict()
            self.problem.obj_quadratic_coefs = dict()
        super(Model, self.__class__).objective.fset(self, value)
        self.update()
        expression = self._objective._expression
        (
            offset,
            linear_coefficients,
            quadratic_coefficients,
        ) = parse_optimization_expression(value, quadratic=True, expression=expression)
        self._objective_offset = offset
        if linear_coefficients:
            self.problem.obj_linear_coefs = {
                v.name: float(c) for v, c in linear_coefficients.items()
            }

        for key, coef in quadratic_coefficients.items():
            if len(key) == 1:
                var = next(iter(key))
                self.problem.obj_quadratic_coefs[(var.name, var.name)] = float(coef)
            else:
                var1, var2 = key
                self.problem.obj_quadratic_coefs[(var1.name, var2.name)] = 0.5 * float(
                    coef
                )
                self.problem.obj_quadratic_coefs[(var2.name, var1.name)] = 0.5 * float(
                    coef
                )

        self._set_objective_direction(value.direction)
        value.problem = self

    def _set_objective_direction(self, direction):
        self.problem.direction = {"min": 1, "max": -1}[direction]

    def _get_primal_values(self):
        if len(self.problem.primals) == 0:
            raise SolverError("The problem has not been solved yet!")
        primal_values = [self.problem.primals[v.name] for v in self._variables]
        return primal_values

    def _get_reduced_costs(self):
        if self.is_integer:
            raise ValueError("Dual values are not well-defined for integer problems")
        if len(self.problem.vduals) == 0:
            raise SolverError("The problem has not been solved yet!")
        reduced_costs = [self.problem.vduals[v.name] for v in self._variables]
        return reduced_costs

    def _get_constraint_values(self):
        if len(self.problem.cprimals) == 0:
            raise SolverError("The problem has not been solved yet!")
        constraint_primals = [self.problem.cprimals[c.name] for c in self._constraints]
        return constraint_primals

    def _get_shadow_prices(self):
        if self.is_integer:
            raise ValueError("Dual values are not well-defined for integer problems")
        if len(self.problem.duals) == 0:
            raise SolverError("The problem has not been solved yet!")
        dual_values = [self.problem.duals[c.name] for c in self._constraints]
        return dual_values

    @property
    def is_integer(self):
        return len(self.problem.integer_vars) > 0

    def _optimize(self):
        self.update()
        self.problem.solve()
        prior_status = self.problem.status
        self._original_status = prior_status
        status = self.status_map[prior_status]
        return status

    def optimize(self):
        self.update()
        status = self._optimize()
        self._status = status
        return status

    def _set_variable_bounds_on_problem(self, var_lb, var_ub):
        self.problem.reset()
        for var, val in var_lb:
            lb = -inf if val is None else float(val)
            self.problem.variable_lbs[var.name] = lb
        for var, val in var_ub:
            ub = inf if val is None else val
            self.problem.variable_ubs[var.name] = float(ub)

    def _add_variables(self, variables):
        super()._add_variables(variables)
        self.problem.reset()
        for variable in variables:
            lb = -inf if variable.lb is None else float(variable.lb)
            ub = inf if variable.ub is None else float(variable.ub)
            self.problem.variables.add(variable.name)
            self.problem.variable_lbs[variable.name] = lb
            self.problem.variable_ubs[variable.name] = ub
            if variable.type in ("binary", "integer"):
                self.problem.integer_vars.add(variable.name)
            variable.problem = self

    def _remove_variables(self, variables):
        # Not calling parent method to avoid expensive variable removal
        # from sympy expressions
        if self.objective is not None:
            self.objective._expression = self.objective.expression.xreplace(
                {var: 0 for var in variables}
            )
        for variable in variables:
            del self._variables_to_constraints_mapping[variable.name]
            variable.problem = None
            del self._variables[variable.name]
            self.problem.variables.remove(variable.name)
        self.problem.prune()

    def _add_constraints(self, constraints, sloppy=False):
        super()._add_constraints(constraints, sloppy=sloppy)
        self.problem.reset()
        for constraint in constraints:
            # This needs to be done in order to not trigger constraint._get_expression()
            constraint._problem = None
            if constraint.is_Linear:
                _, coeff_dict, _ = parse_optimization_expression(constraint)
                lb = -inf if constraint.lb is None else float(constraint.lb)
                ub = inf if constraint.ub is None else float(constraint.ub)
                self.problem.constraints.add(constraint.name)
                self.problem.constraint_coefs[constraint.name] = {
                    v.name: float(co)
                    for v, co in coeff_dict.items()
                }
                self.problem.constraint_lbs[constraint.name] = lb
                self.problem.constraint_ubs[constraint.name] = ub
                constraint.problem = self
            elif constraint.is_Quadratic:
                raise NotImplementedError(
                    "Quadratic constraints (like %s) are not supported "
                    "in this interface yet." % constraint
                )
            else:
                raise ValueError(
                    "This interface only supports linear or quadratic constraints. "
                    "%s is neither linear nor quadratic." % constraint
                )

    def _remove_constraints(self, constraints):
        super()._remove_constraints(constraints)
        for constraint in constraints:
            self.problem.constraints.remove(constraint.name)
        self.problem.prune()

    def _get_variable_indices(self, names):
        vmap = dict(zip(self.variables, range(len(self.variables))))
        return [vmap[n] for n in names]

    def __setstate__(self, d):
        self.__init__()
        problem = pickle.loads(d["problem"])
        # workaround to conserve the order
        problem.variables = d["variables"]
        problem.constraints = d["constraints"]
        self._initialize_model_from_problem(
            problem, vc_mapping=d["v_to_c"], offset=d["offset"]
        )
        problem.variables = set(problem.variables)
        problem.constraints = set(problem.constraints)
        self.configuration = self.interface.Configuration()
        self.configuration.problem = self
        self.configuration.__setstate__(d["config"])

    def __getstate__(self):
        self.problem.reset()
        self.update()
        return {
            "problem": pickle.dumps(self.problem),
            "variables": [v.name for v in self._variables],
            "constraints": [c.name for c in self._constraints],
            "v_to_c": self._variables_to_constraints_mapping,
            "config": self.configuration.__getstate__(),
            "offset": getattr(self, "_objective_offset", 0.0),
        }

    @classmethod
    def from_lp(self, lp_problem_str):
        """Read a model from an LP file.

        The solver may not have an integrated LP reader so it will either use
        cplex or glpk to read the model. This means that QP problems will
        currently require cplex to be read :(
        """
        if available_solvers["CPLEX"]:
            from optlang import cplex_interface

            mod = cplex_interface.Model.from_lp(lp_problem_str)
            mod.configuration.lp_method = "auto"
            mod.configuration.qp_method = "auto"
            return super().clone(mod)
        else:
            from optlang import glpk_interface

            mod = glpk_interface.Model.from_lp(lp_problem_str)
            mod.configuration.lp_method = "auto"
            return super().clone(mod)
