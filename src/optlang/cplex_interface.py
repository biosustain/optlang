# Copyright 2013 Novo Nordisk Foundation Center for Biosustainability,
# Technical University of Denmark.
#
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

"""Solver interface for the IBM ILOG CPLEX Optimization Studio solver.

Wraps the cplex solver by subclassing and extending :class:`Model`,
:class:`Variable`, and :class:`Constraint` from :mod:`interface`.

To use this interface, install the cplex solver and the bundled python interface.
Make sure that 'import cplex' runs without error.
"""
import logging
import sys

import os
from io import StringIO

import cplex
from cplex.exceptions import CplexSolverError

from optlang import interface
from optlang import symbolics
from optlang.util import inheritdocstring, TemporaryFilename
from optlang.expression_parsing import parse_optimization_expression
from optlang.exceptions import SolverError

log = logging.getLogger(__name__)

from optlang.symbolics import add, mul, One, Zero

_STATUS_MAP = {
    'MIP_abort_feasible': interface.ABORTED,
    'MIP_abort_infeasible': interface.ABORTED,
    'MIP_dettime_limit_feasible': interface.TIME_LIMIT,
    'MIP_dettime_limit_infeasible': interface.TIME_LIMIT,
    'MIP_feasible': interface.FEASIBLE,
    'MIP_feasible_relaxed_inf': interface.SPECIAL,
    'MIP_feasible_relaxed_quad': interface.SPECIAL,
    'MIP_feasible_relaxed_sum': interface.SPECIAL,
    'MIP_infeasible': interface.INFEASIBLE,
    'MIP_infeasible_or_unbounded': interface.INFEASIBLE_OR_UNBOUNDED,
    'MIP_optimal': interface.OPTIMAL,
    'MIP_optimal_infeasible': interface.SPECIAL,
    'MIP_optimal_relaxed_inf': interface.SPECIAL,
    'MIP_optimal_relaxed_sum': interface.SPECIAL,
    'MIP_time_limit_feasible': interface.TIME_LIMIT,
    'MIP_time_limit_infeasible': interface.TIME_LIMIT,
    'MIP_unbounded': interface.UNBOUNDED,
    'abort_dettime_limit': interface.ABORTED,
    'abort_dual_obj_limit': interface.ABORTED,
    'abort_iteration_limit': interface.ABORTED,
    'abort_obj_limit': interface.ABORTED,
    'abort_primal_obj_limit': interface.ABORTED,
    'abort_relaxed': interface.ABORTED,
    'abort_time_limit': interface.TIME_LIMIT,
    'abort_user': interface.ABORTED,
    'conflict_abort_contradiction': interface.SPECIAL,
    'conflict_abort_dettime_limit': interface.SPECIAL,
    'conflict_abort_iteration_limit': interface.SPECIAL,
    'conflict_abort_memory_limit': interface.SPECIAL,
    'conflict_abort_node_limit': interface.SPECIAL,
    'conflict_abort_obj_limit': interface.SPECIAL,
    'conflict_abort_time_limit': interface.SPECIAL,
    'conflict_abort_user': interface.SPECIAL,
    'conflict_feasible': interface.SPECIAL,
    'conflict_minimal': interface.SPECIAL,
    'fail_feasible': interface.SPECIAL,
    'fail_feasible_no_tree': interface.SPECIAL,
    'fail_infeasible': interface.SPECIAL,
    'fail_infeasible_no_tree': interface.SPECIAL,
    'feasible': interface.FEASIBLE,
    'feasible_relaxed_inf': interface.SPECIAL,
    'feasible_relaxed_quad': interface.SPECIAL,
    'feasible_relaxed_sum': interface.SPECIAL,
    'first_order': interface.SPECIAL,
    'infeasible': interface.INFEASIBLE,
    'infeasible_or_unbounded': interface.INFEASIBLE_OR_UNBOUNDED,
    'mem_limit_feasible': interface.MEMORY_LIMIT,
    'mem_limit_infeasible': interface.MEMORY_LIMIT,
    'node_limit_feasible': interface.NODE_LIMIT,
    'node_limit_infeasible': interface.NODE_LIMIT,
    'num_best': interface.NUMERIC,
    'optimal': interface.OPTIMAL,
    'optimal_face_unbounded': interface.SPECIAL,
    'optimal_infeasible': interface.INFEASIBLE,
    'optimal_populated': interface.SPECIAL,
    'optimal_populated_tolerance': interface.SPECIAL,
    'optimal_relaxed_inf': interface.SPECIAL,
    'optimal_relaxed_quad': interface.SPECIAL,
    'optimal_relaxed_sum': interface.SPECIAL,
    'optimal_tolerance': interface.OPTIMAL,
    'populate_solution_limit': interface.SPECIAL,
    'solution_limit': interface.SPECIAL,
    'unbounded': interface.UNBOUNDED,
    'relaxation_unbounded': interface.UNBOUNDED,
    'non-existing-status': 'Here for testing that missing statuses are handled.'
    # 102: interface.OPTIMAL # The same as cplex.Cplex.solution.status.optimal_tolerance
}

# Check if each status is supported by the current cplex version
_CPLEX_STATUS_TO_STATUS = {}
_solution = cplex.Cplex().solution
for status_name, optlang_status in _STATUS_MAP.items():
    cplex_status = getattr(_solution.status, status_name, None)
    if cplex_status is not None:
        _CPLEX_STATUS_TO_STATUS[cplex_status] = optlang_status

_LP_METHODS = ["auto", "primal", "dual", "network", "barrier", "sifting", "concurrent"]

_SOLUTION_TARGETS = ("auto", "convex", "local", "global")

_QP_METHODS = ("auto", "primal", "dual", "network", "barrier")

_CPLEX_VTYPE_TO_VTYPE = {'C': 'continuous', 'I': 'integer', 'B': 'binary'}

_VTYPE_TO_CPLEX_VTYPE = dict(
    [(val, key) for key, val in _CPLEX_VTYPE_TO_VTYPE.items()]
)

_CPLEX_MIP_TYPES_TO_CONTINUOUS = {
    cplex.Cplex.problem_type.MILP: cplex.Cplex.problem_type.LP,
    cplex.Cplex.problem_type.MIQP: cplex.Cplex.problem_type.QP,
    cplex.Cplex.problem_type.MIQCP: cplex.Cplex.problem_type.QCP
}


def _constraint_lb_and_ub_to_cplex_sense_rhs_and_range_value(lb, ub):
    """Helper function used by Constraint and Model"""
    if lb is None and ub is None:
        raise Exception("Free constraint ...")
    elif lb is None:
        sense = 'L'
        rhs = float(ub)
        range_value = 0.
    elif ub is None:
        sense = 'G'
        rhs = float(lb)
        range_value = 0.
    elif lb == ub:
        sense = 'E'
        rhs = float(lb)
        range_value = 0.
    elif lb > ub:
        raise ValueError("Lower bound is larger than upper bound.")
    else:
        sense = 'R'
        rhs = float(lb)
        range_value = float(ub - lb)
    return sense, rhs, range_value


def _get_solver_parameters(problem):
    """Return the numeric parameters of a CPLEX problem that differ from their defaults."""
    return {repr(parameter): value for parameter, value in problem.parameters.get_changed()
            if not isinstance(value, str)}


def _set_solver_parameters(problem, parameters):
    """Set the given parameters of a CPLEX problem."""
    for parameter, _ in problem.parameters.get_all():
        if repr(parameter) in parameters:
            parameter.set(parameters[repr(parameter)])


class Variable(interface.Variable, metaclass=inheritdocstring):
    def __init__(self, name, *args, **kwargs):
        super(Variable, self).__init__(name, **kwargs)

    @interface.Variable.type.setter
    def type(self, value):
        if self.problem is not None:
            try:
                cplex_kind = _VTYPE_TO_CPLEX_VTYPE[value]
            except KeyError:
                raise ValueError(
                    "CPLEX cannot handle variables of type '%s'. " % value +
                    "The following variable types are available: " +
                    ", ".join(_VTYPE_TO_CPLEX_VTYPE.keys())
                )
            self.problem.problem.variables.set_types(self.name, cplex_kind)
            if value == "continuous" and not self.problem.is_integer:
                cplex_type = self.problem.problem.get_problem_type()
                if cplex_type in _CPLEX_MIP_TYPES_TO_CONTINUOUS:
                    self.problem.problem.set_problem_type(_CPLEX_MIP_TYPES_TO_CONTINUOUS[cplex_type])
        super(Variable, Variable).type.fset(self, value)

    def _get_primal(self):
        try:
            return self.problem.problem.solution.get_values(self.name)
        except CplexSolverError as err:
            raise SolverError(str(err))

    @property
    def dual(self):
        if self.problem is None:
            return None
        if self.problem.is_integer:
            raise ValueError("Dual values are not well-defined for integer problems")
        try:
            return self.problem.problem.solution.get_reduced_costs(self.name)
        except CplexSolverError as err:
            raise SolverError(str(err))

    @interface.Variable.name.setter
    def name(self, value):
        old_name = getattr(self, "name", None)
        super(Variable, Variable).name.fset(self, value)
        if getattr(self, "problem", None) is not None:
            self.problem.problem.variables.set_names(old_name, value)


class Constraint(interface.Constraint, metaclass=inheritdocstring):
    _INDICATOR_CONSTRAINT_SUPPORT = True

    def __init__(self, expression, sloppy=False, *args, **kwargs):
        super(Constraint, self).__init__(expression, *args, sloppy=sloppy, **kwargs)

    def set_linear_coefficients(self, coefficients):
        if self.problem is not None:
            self.problem.update()
            triplets = [(self.name, var.name, float(coeff)) for var, coeff in coefficients.items()]
            self.problem.problem.linear_constraints.set_coefficients(triplets)
        else:
            raise Exception("Can't change coefficients if constraint is not associated with a model.")

    def get_linear_coefficients(self, variables):
        if self.problem is not None:
            self.problem.update()
            coefs = self.problem.problem.linear_constraints.get_coefficients([(self.name, v.name) for v in variables])
            return {v: c for v, c in zip(variables, coefs)}
        else:
            raise Exception("Can't get coefficients from solver if constraint is not in a model")

    def _get_expression(self):
        if self.problem is not None:
            cplex_problem = self.problem.problem
            try:
                cplex_row = cplex_problem.linear_constraints.get_rows(self.name)
            except CplexSolverError as e:
                if 'CPLEX Error  1219:' not in str(e):
                    raise e
                else:
                    cplex_row = cplex_problem.indicator_constraints.get_linear_components(self.name)
            variables = self.problem._variables
            expression = add(
                [mul((symbolics.Real(cplex_row.val[i]), variables[ind])) for i, ind in
                 enumerate(cplex_row.ind)])
            self._expression = expression
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
        try:
            # return self._round_primal_to_bounds(primal_from_solver)  # Test assertions fail
            return self.problem.problem.solution.get_activity_levels(self.name)
        except CplexSolverError as err:
            raise SolverError(str(err))

    @property
    def dual(self):
        if self.problem is None:
            return None
        if self.problem.is_integer:
            raise ValueError("Dual values are not well-defined for integer problems")
        try:
            return self.problem.problem.solution.get_dual_values(self.name)
        except CplexSolverError as err:
            raise SolverError(str(err))

    @interface.Constraint.name.setter
    def name(self, value):
        old_name = self.name
        super(Constraint, Constraint).name.fset(self, value)
        if getattr(self, 'problem', None) is not None:
            if self.indicator_variable is not None:
                raise NotImplementedError(
                    "Unfortunately, the CPLEX python bindings don't support changing an indicator constraint's name"
                )
            else:
                self.problem.problem.linear_constraints.set_names(old_name, value)

    @interface.Constraint.lb.setter
    def lb(self, value):
        self._check_valid_lower_bound(value)
        if getattr(self, 'problem', None) is not None:
            if self.indicator_variable is not None:
                raise NotImplementedError(
                    "Unfortunately, the CPLEX python bindings don't support changing an indicator constraint's bounds"
                )
            sense, rhs, range_value = _constraint_lb_and_ub_to_cplex_sense_rhs_and_range_value(value, self.ub)
            if self.is_Linear:
                self.problem.problem.linear_constraints.set_rhs(self.name, rhs)
                self.problem.problem.linear_constraints.set_senses(self.name, sense)
                self.problem.problem.linear_constraints.set_range_values(self.name, range_value)
        self._lb = value

    @interface.Constraint.ub.setter
    def ub(self, value):
        self._check_valid_upper_bound(value)
        if getattr(self, 'problem', None) is not None:
            if self.indicator_variable is not None:
                raise NotImplementedError(
                    "Unfortunately, the CPLEX python bindings don't support changing an indicator constraint's bounds"
                )
            sense, rhs, range_value = _constraint_lb_and_ub_to_cplex_sense_rhs_and_range_value(self.lb, value)
            if self.is_Linear:
                self.problem.problem.linear_constraints.set_rhs(self.name, rhs)
                self.problem.problem.linear_constraints.set_senses(self.name, sense)
                self.problem.problem.linear_constraints.set_range_values(self.name, range_value)
        self._ub = value

    def __iadd__(self, other):
        # if self.problem is not None:
        #     self.problem._add_to_constraint(self.index, other)
        if self.problem is not None:
            problem_reference = self.problem
            self.problem._remove_constraint(self)
            super(Constraint, self).__iadd__(other)
            problem_reference._add_constraint(self, sloppy=False)
        else:
            super(Constraint, self).__iadd__(other)
        return self


class Objective(interface.Objective, metaclass=inheritdocstring):
    def __init__(self, expression, sloppy=False, **kwargs):
        super(Objective, self).__init__(expression, sloppy=sloppy, **kwargs)
        self._expression_expired = False
        if not (sloppy or self.is_Linear or self.is_Quadratic):
            raise ValueError("Cplex only supports linear and quadratic objectives.")

    @property
    def value(self):
        if getattr(self, 'problem', None) is None:
            return None
        try:
            return self.problem.problem.solution.get_objective_value() + getattr(self.problem, "_objective_offset", 0)
        except CplexSolverError as err:
            raise SolverError(str(err))

    @interface.Objective.direction.setter
    def direction(self, value):
        if getattr(self, 'problem', None) is not None:
            self.problem.problem.objective.set_sense(
                {'min': self.problem.problem.objective.sense.minimize,
                 'max': self.problem.problem.objective.sense.maximize}[value])
        super(Objective, Objective).direction.__set__(self, value)

    def _get_expression(self):
        if self.problem is not None and self._expression_expired and len(self.problem._variables) > 0:
            cplex_problem = self.problem.problem
            coeffs = cplex_problem.objective.get_linear()
            expression = add([coeff * var for coeff, var in zip(coeffs, self.problem._variables) if coeff != 0.])
            if cplex_problem.objective.get_num_quadratic_nonzeros() > 0:
                expression += self.problem._get_quadratic_expression(cplex_problem.objective.get_quadratic())
            self._expression = expression + getattr(self.problem, "_objective_offset", 0)
            self._expression_expired = False
        return self._expression

    def set_linear_coefficients(self, coefficients):
        if self.problem is not None:
            self.problem.update()
            self.problem.problem.objective.set_linear([(variable.name, float(coefficient)) for variable, coefficient in coefficients.items()])
            self._expression_expired = True
        else:
            raise Exception("Can't change coefficients if objective is not associated with a model.")

    def get_linear_coefficients(self, variables):
        if self.problem is not None:
            self.problem.update()
            coefs = self.problem.problem.objective.get_linear([v.name for v in variables])
            return {v: c for v, c in zip(variables, coefs)}
        else:
            raise Exception("Can't get coefficients from solver if objective is not in a model")


class Configuration(interface.MathematicalProgrammingConfiguration, metaclass=inheritdocstring):
    def __init__(self, lp_method='primal', presolve="auto", verbosity=0, timeout=None,
                 solution_target="auto", qp_method="primal", *args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)
        self.lp_method = lp_method
        self.presolve = presolve
        self.verbosity = verbosity
        self.timeout = timeout
        self.solution_target = solution_target
        self.qp_method = qp_method
        if "tolerances" in kwargs:
            for key, val in kwargs["tolerances"].items():
                try:
                    setattr(self.tolerances, key, val)
                except AttributeError:
                    pass

    @property
    def lp_method(self):
        """The algorithm used to solve LP problems."""
        lpmethod = self.problem.problem.parameters.lpmethod
        value = lpmethod.get()
        return lpmethod.values[value]

    @lp_method.setter
    def lp_method(self, lp_method):
        if lp_method not in _LP_METHODS:
            raise ValueError("LP Method %s is not valid (choose one of: %s)" % (lp_method, ", ".join(_LP_METHODS)))
        lp_method = getattr(self.problem.problem.parameters.lpmethod.values, lp_method)
        self.problem.problem.parameters.lpmethod.set(lp_method)

    def _set_presolve(self, value):
        if getattr(self, 'problem', None) is not None:
            presolve = self.problem.problem.parameters.preprocessing.presolve
            if value is True:
                presolve.set(presolve.values.on)
            elif value is False or value == "auto":
                presolve.set(presolve.values.off)
            else:
                raise ValueError(str(value) + " is not a valid presolve parameter. Must be True, False or 'auto'.")

    @property
    def presolve(self):
        return self._presolve

    @presolve.setter
    def presolve(self, value):
        self._set_presolve(value)
        self._presolve = value

    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value):

        class StreamHandler(StringIO):

            def __init__(self, logger, *args, **kwargs):
                StringIO.__init__(self, *args, **kwargs)
                self.logger = logger

        class ErrorStreamHandler(StreamHandler):

            def flush(self):
                self.logger.error(self.getvalue())

        class WarningStreamHandler(StreamHandler):

            def flush(self):
                self.logger.warning(self.getvalue())

        class LogStreamHandler(StreamHandler):

            def flush(self):
                self.logger.debug(self.getvalue())

        class ResultsStreamHandler(StreamHandler):

            def flush(self):
                self.logger.debug(self.getvalue())

        logger = logging.getLogger(__name__ + ".Model")  # TODO: Make the logger name specific to each solver instance
        logger.setLevel(logging.CRITICAL)
        error_stream_handler = ErrorStreamHandler(logger)
        warning_stream_handler = WarningStreamHandler(logger)
        log_stream_handler = LogStreamHandler(logger)
        results_stream_handler = ResultsStreamHandler(logger)
        if getattr(self, 'problem', None) is not None:
            problem = self.problem.problem
            if value == 0:
                problem.set_error_stream(error_stream_handler)
                problem.set_warning_stream(warning_stream_handler)
                problem.set_log_stream(None)
                problem.set_results_stream(None)
            elif value == 1:
                problem.set_error_stream(sys.stderr)
                problem.set_warning_stream(warning_stream_handler)
                problem.set_log_stream(log_stream_handler)
                problem.set_results_stream(results_stream_handler)
            elif value == 2:
                problem.set_error_stream(sys.stderr)
                problem.set_warning_stream(sys.stderr)
                problem.set_log_stream(log_stream_handler)
                problem.set_results_stream(results_stream_handler)
            elif value == 3:
                problem.set_error_stream(sys.stderr)
                problem.set_warning_stream(sys.stderr)
                problem.set_log_stream(sys.stdout)
                problem.set_results_stream(sys.stdout)
            else:
                raise ValueError(
                    "%s is not a valid verbosity level ranging between 0 and 3."
                    % value
                )
        self._verbosity = value

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        if getattr(self, 'problem', None) is not None:
            if value is None:
                self.problem.problem.parameters.timelimit.reset()
            else:
                self.problem.problem.parameters.timelimit.set(value)
        self._timeout = value

    def __getstate__(self):
        return {"presolve": self.presolve,
                "timeout": self.timeout,
                "verbosity": self.verbosity,
                "tolerances": {"feasibility": self.tolerances.feasibility,
                               "optimality": self.tolerances.optimality,
                               "integrality": self.tolerances.integrality}
                }

    def __setstate__(self, state):
        for key, val in state.items():
            if key != "tolerances":
                setattr(self, key, val)

    @property
    def solution_target(self):
        """Change whether the QP solver will try to find a globally optimal solution or a local optimum.
        This will only"""
        if self.problem is not None:
            params = self.problem.problem.parameters
            try:
                solution_target = params.optimalitytarget
            except AttributeError:  # pragma: no cover
                solution_target = params.solutiontarget  # Compatibility with cplex < 12.6.3
            return _SOLUTION_TARGETS[solution_target.get()]
        else:
            return None

    @solution_target.setter
    def solution_target(self, value):
        if self.problem is not None:
            params = self.problem.problem.parameters
            try:
                solution_target = params.optimalitytarget
            except AttributeError:  # pragma: no cover
                solution_target = params.solutiontarget  # Compatibility with cplex < 12.6.3
            if value is None:
                solution_target.reset()
            else:
                try:
                    target = _SOLUTION_TARGETS.index(value)
                except ValueError:
                    raise ValueError(
                        "%s is not a valid solution target. Choose between %s" % (value, str(_SOLUTION_TARGETS)))
                solution_target.set(target)
        self._solution_target = self.solution_target

    @property
    def qp_method(self):
        """Change the algorithm used to optimize QP problems."""
        value = self.problem.problem.parameters.qpmethod.get()
        return self.problem.problem.parameters.qpmethod.values[value]

    @qp_method.setter
    def qp_method(self, value):
        if value not in _QP_METHODS:
            raise ValueError("%s is not a valid qp_method. Choose between %s" % (value, str(_QP_METHODS)))
        method = getattr(self.problem.problem.parameters.qpmethod.values, value)
        self.problem.problem.parameters.qpmethod.set(method)
        self._qp_method = value

    def _tolerance_functions(self):
        return {
            "feasibility": (
                self.problem.problem.parameters.simplex.tolerances.feasibility.get,
                self.problem.problem.parameters.simplex.tolerances.feasibility.set
            ),
            "optimality": (
                self.problem.problem.parameters.simplex.tolerances.optimality.get,
                self.problem.problem.parameters.simplex.tolerances.optimality.set
            ),
            "integrality": (
                self.problem.problem.parameters.mip.tolerances.integrality.get,
                self.problem.problem.parameters.mip.tolerances.integrality.set
            )
        }


class Model(interface.Model, metaclass=inheritdocstring):
    def _initialize_problem(self):
        self.problem = cplex.Cplex()

    def _initialize_model_from_problem(self, problem):
        if isinstance(problem, cplex.Cplex):
            self.problem = problem
            zipped_var_args = zip(self.problem.variables.get_names(),
                                  self.problem.variables.get_lower_bounds(),
                                  self.problem.variables.get_upper_bounds(),
                                  # self.problem.variables.get_types(), # TODO uncomment when cplex is fixed
                                  )
            for name, lb, ub in zipped_var_args:
                var = Variable(name, lb=lb, ub=ub, problem=self)  # Type should also be in there
                super(Model, self)._add_variables([var])  # This avoids adding the variable to the glpk problem
            zipped_constr_args = zip(self.problem.linear_constraints.get_names(),
                                     self.problem.linear_constraints.get_rows(),
                                     self.problem.linear_constraints.get_senses(),
                                     self.problem.linear_constraints.get_rhs()

                                     )
            variables = self._variables
            for name, row, sense, rhs in zipped_constr_args:
                constraint_variables = [variables[i - 1] for i in row.ind]

                # Since constraint expressions are lazily retrieved from the solver they don't have to be built here
                # lhs = _unevaluated_Add(*[val * variables[i - 1] for i, val in zip(row.ind, row.val)])
                lhs = symbolics.Integer(0)
                if sense == 'E':
                    constr = Constraint(lhs, lb=rhs, ub=rhs, name=name, problem=self)
                elif sense == 'G':
                    constr = Constraint(lhs, lb=rhs, name=name, problem=self)
                elif sense == 'L':
                    constr = Constraint(lhs, ub=rhs, name=name, problem=self)
                elif sense == 'R':
                    range_val = self.problem.linear_constraints.get_range_values(name)
                    if range_val > 0:
                        constr = Constraint(lhs, lb=rhs, ub=rhs + range_val, name=name, problem=self)
                    else:
                        constr = Constraint(lhs, lb=rhs + range_val, ub=rhs, name=name, problem=self)
                else:  # pragma: no cover
                    raise Exception('%s is not a recognized constraint sense.' % sense)

                for variable in constraint_variables:
                    try:
                        self._variables_to_constraints_mapping[variable.name].add(name)
                    except KeyError:
                        self._variables_to_constraints_mapping[variable.name] = set([name])

                super(Model, self)._add_constraints(
                    [constr],
                    sloppy=True
                )
            try:
                objective_name = self.problem.objective.get_name()
            except CplexSolverError as e:
                if 'CPLEX Error  1219:' not in str(e):
                    raise e
            else:
                linear_expression = add(
                    [mul(symbolics.Real(coeff), variables[index]) for index, coeff in enumerate(self.problem.objective.get_linear()) if coeff != 0.]
                )
                try:
                    quadratic = self.problem.objective.get_quadratic()
                except IndexError:
                    quadratic_expression = Zero
                else:
                    quadratic_expression = self._get_quadratic_expression(quadratic)

                self._objective = Objective(
                    linear_expression + quadratic_expression,
                    problem=self,
                    direction=
                    {self.problem.objective.sense.minimize: 'min', self.problem.objective.sense.maximize: 'max'}[
                        self.problem.objective.get_sense()],
                    name=objective_name
                )
        else:
            raise TypeError("Provided problem is not a valid CPLEX model.")

    @classmethod
    def from_lp(cls, lp_form):
        problem = cplex.Cplex()
        with TemporaryFilename(suffix=".lp", content=lp_form) as tmp_file_name:
            problem.read(tmp_file_name)
        model = cls(problem=problem)
        return model

    def __getstate__(self):
        self.update()
        with TemporaryFilename(suffix=".sav") as tmp_file_name:
            self.problem.write(tmp_file_name)
            with open(tmp_file_name, "rb") as tmp_file:
                cplex_binary = tmp_file.read()
        repr_dict = {'cplex_binary': cplex_binary, 'status': self.status,
                     'configuration': Configuration._get_parameters(self.configuration),
                     'parameters': _get_solver_parameters(self.problem)}
        return repr_dict

    def __setstate__(self, repr_dict):
        with TemporaryFilename(suffix=".sav") as tmp_file_name:
            with open(tmp_file_name, "wb") as tmp_file:
                tmp_file.write(repr_dict["cplex_binary"])
            problem = cplex.Cplex()
            # turn off logging completely, get's configured later
            problem.set_error_stream(None)
            problem.set_warning_stream(None)
            problem.set_log_stream(None)
            problem.set_results_stream(None)
            problem.read(tmp_file_name)
        if repr_dict['status'] == 'optimal':
            problem.solve()  # since the start is an optimal solution, nothing will happen here
        self.__init__(problem=problem)
        if 'configuration' in repr_dict:
            self.configuration = Configuration(problem=self, **repr_dict['configuration'])
            _set_solver_parameters(problem, repr_dict['parameters'])
        else:  # pickled by optlang <= 1.9.1
            self.configuration = Configuration.clone(repr_dict['config'], problem=self)

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, value):
        value.problem = None
        if self._objective is not None:  # Reset previous objective
            variables = self.objective.variables
            if len(variables) > 0:
                name_list = [var.name for var in variables]
                index_dict = {n: i for n, i in zip(name_list, self._get_variable_indices(name_list))}
                self.problem.objective.set_linear([(index_dict[variable.name], 0.) for variable in variables])
            if self.problem.objective.get_num_quadratic_variables() > 0:
                self.problem.objective.set_quadratic([0. for _ in range(self.problem.variables.get_num())])
        super(Model, self.__class__).objective.fset(self, value)
        self.update()
        expression = self._objective._expression
        offset, linear_coefficients, quadratic_coeffients = parse_optimization_expression(value, quadratic=True, expression=expression)
        # self.problem.objective.set_offset(float(offset)) # Not available prior to 12.6.2
        self._objective_offset = offset
        if linear_coefficients:
            name_list = [var.name for var in linear_coefficients]
            index_dict = {n: i for n, i in zip(name_list, self._get_variable_indices(name_list))}
            self.problem.objective.set_linear([index_dict[var.name], float(coef)] for var, coef in linear_coefficients.items())

        for key, coef in quadratic_coeffients.items():
            if len(key) == 1:
                var = next(iter(key))
                self.problem.objective.set_quadratic_coefficients(var.name, var.name, float(coef) * 2)
            else:
                var1, var2 = key
                self.problem.objective.set_quadratic_coefficients(var1.name, var2.name, float(coef))


        self._set_objective_direction(value.direction)
        self.problem.objective.set_name(value.name)
        value.problem = self

    def _set_objective_direction(self, direction):
        self.problem.objective.set_sense(
            {'min': self.problem.objective.sense.minimize, 'max': self.problem.objective.sense.maximize}[
                direction])

    def _get_primal_values(self):
        try:
            primal_values = self.problem.solution.get_values()
        except CplexSolverError as err:
            raise SolverError(str(err))
        return primal_values

    def _get_reduced_costs(self):
        if self.is_integer:
            raise ValueError("Dual values are not well-defined for integer problems")
        try:
            return self.problem.solution.get_reduced_costs()
        except CplexSolverError as err:
            raise SolverError(str(err))

    def _get_constraint_values(self):
        try:
            return self.problem.solution.get_activity_levels()
        except CplexSolverError as err:
            raise SolverError(str(err))

    def _get_shadow_prices(self):
        if self.is_integer:
            raise ValueError("Dual values are not well-defined for integer problems")
        try:
            return self.problem.solution.get_dual_values()
        except CplexSolverError as err:
            raise SolverError(str(err))

    @property
    def is_integer(self):
        return (self.problem.variables.get_num_integer() + self.problem.variables.get_num_binary()) > 0

    def to_lp(self):
        self.update()
        with TemporaryFilename(suffix=".lp") as tmp_file_name:
            self.problem.write(tmp_file_name)
            with open(tmp_file_name) as tmp_file:
                lp_form = tmp_file.read()
        return lp_form

    def _optimize(self):
        try:
            self.problem.solve()
        except CplexSolverError as err:
            raise SolverError(str(err))
        cplex_status = self.problem.solution.get_status()
        self._original_status = self.problem.solution.get_status_string()
        status = _CPLEX_STATUS_TO_STATUS[cplex_status]
        return status

    def _set_variable_bounds_on_problem(self, var_lb, var_ub):
        lb = [
            (var.name, -cplex.infinity) if val is None else (var.name, float(val)) for var, val in var_lb
        ]
        if len(lb) > 0:
            self.problem.variables.set_lower_bounds(lb)
        ub = [
            (var.name, cplex.infinity) if val is None else (var.name, float(val)) for var, val in var_ub
        ]
        if len(ub) > 0:
            self.problem.variables.set_upper_bounds(ub)

    def _add_variables(self, variables):
        super(Model, self)._add_variables(variables)
        lb, ub, vtype, names, coeff = list(), list(), list(), list(), list()
        for variable in variables:
            if variable.lb is None:
                lb.append(-cplex.infinity)
            else:
                lb.append(variable.lb)
            if variable.ub is None:
                ub.append(cplex.infinity)
            else:
                ub.append(variable.ub)
            vtype.append(_VTYPE_TO_CPLEX_VTYPE[variable.type])
            names.append(variable.name)
            coeff.append(0.)
            variable.problem = self

        if set(vtype) == set(['C']):  # this is needed because CPLEX will automatically set the problem_type to MILP if types are specified
            self.problem.variables.add(coeff, lb=lb, ub=ub, names=names)
        else:
            self.problem.variables.add(coeff, lb=lb, ub=ub, types=vtype, names=names)

    def _remove_variables(self, variables):
        # Not calling parent method to avoid expensive variable removal from sympy expressions
        if self.objective is not None:
            self.objective._expression = self.objective.expression.xreplace({var: 0 for var in variables})
        self.problem.variables.delete([variable.name for variable in variables])
        for variable in variables:
            del self._variables_to_constraints_mapping[variable.name]
            variable.problem = None
            del self._variables[variable.name]
        if not self.is_integer:
            cplex_type = self.problem.get_problem_type()
            if cplex_type in _CPLEX_MIP_TYPES_TO_CONTINUOUS:
                self.problem.set_problem_type(_CPLEX_MIP_TYPES_TO_CONTINUOUS[cplex_type])

    def _add_constraints(self, constraints, sloppy=False):
        super(Model, self)._add_constraints(constraints, sloppy=sloppy)

        linear_constraints = dict(lin_expr=[], senses=[], rhs=[], range_values=[], names=[])
        for constraint in constraints:
            constraint._problem = None  # This needs to be done in order to not trigger constraint._get_expression()
            if constraint.is_Linear:
                offset, coeff_dict, _ = parse_optimization_expression(constraint)

                sense, rhs, range_value = _constraint_lb_and_ub_to_cplex_sense_rhs_and_range_value(
                    constraint.lb,
                    constraint.ub
                )
                indices = [var.name for var in coeff_dict]
                values = [float(val) for val in coeff_dict.values()]
                if constraint.indicator_variable is None:
                    linear_constraints['lin_expr'].append(cplex.SparsePair(ind=indices, val=values))
                    linear_constraints['senses'].append(sense)
                    linear_constraints['rhs'].append(rhs)
                    linear_constraints['range_values'].append(range_value)
                    linear_constraints['names'].append(constraint.name)
                else:
                    if sense == 'R':
                        raise ValueError(
                            'CPLEX does not support indicator constraints that have both an upper and lower bound.')
                    else:
                        # Indicator constraints cannot be added in batch
                        self.problem.indicator_constraints.add(
                            lin_expr=cplex.SparsePair(ind=indices, val=values), sense=sense, rhs=rhs,
                            name=constraint.name,
                            indvar=constraint.indicator_variable.name, complemented=abs(constraint.active_when) - 1)

            elif constraint.is_Quadratic:
                raise NotImplementedError('Quadratic constraints (like %s) are not supported yet.' % constraint)
            else:
                raise ValueError(
                    "CPLEX only supports linear or quadratic constraints. %s is neither linear nor quadratic." % constraint)
            constraint.problem = self
        self.problem.linear_constraints.add(**linear_constraints)

    def _remove_constraints(self, constraints):
        super(Model, self)._remove_constraints(constraints)
        for constraint in constraints:
            if constraint.indicator_variable is not None:
                self.problem.indicator_constraints.delete(constraint.name)
            elif constraint.is_Linear:
                self.problem.linear_constraints.delete(constraint.name)
            elif constraint.is_Quadratic:
                self.problem.quadratic_constraints.delete(constraint.name)

    def _get_quadratic_expression(self, quadratic=None):
        if quadratic is None:
            try:
                quadratic = self.problem.objective.get_quadratic()
            except IndexError:
                return Zero
        terms = []
        for i, sparse_pair in enumerate(quadratic):
            for j, val in zip(sparse_pair.ind, sparse_pair.val):
                i_name, j_name = self.problem.variables.get_names([i, j])
                if i < j:
                    terms.append(val * self._variables[i_name] * self._variables[j_name])
                elif i == j:
                    terms.append(0.5 * val * self._variables[i_name] ** 2)
                else:
                    pass  # Only look at upper triangle
        return add(terms)

    def _get_variable_indices(self, names):
        # Cplex does not keep an index of variable names
        # Getting indices thus takes roughly quadratic time
        # If many indices are required an alternate and faster method is used, where the full name list must only
        # be traversed once
        if len(names) < 1000:
            return self.problem.variables.get_indices(names)
        else:
            name_set = set(names)
            all_names = self.problem.variables.get_names()
            indices = {n: i for i, n in enumerate(all_names) if n in name_set}
            return [indices[n] for n in names]
