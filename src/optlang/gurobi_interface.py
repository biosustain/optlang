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


"""
Solver interface for the Gurobi (MI)LP/QP solver.

To use this interface, install the gurobi solver and the bundled python interface
and make sure that 'import gurobipy' runs without error.
"""

import logging

log = logging.getLogger(__name__)

import os
from optlang import interface
from optlang.util import inheritdocstring, TemporaryFilename
from optlang.expression_parsing import parse_optimization_expression
from optlang import symbolics

import gurobipy

try:
    version = gurobipy.gurobi.version()
    if version[:2] < (9, 5):
        raise RuntimeError()
except Exception:
    raise RuntimeError(
        "This version of optlang requires a Gurobi version of 9.5 or above.")

_GUROBI_STATUS_TO_STATUS = {
    gurobipy.GRB.LOADED: interface.LOADED,
    gurobipy.GRB.OPTIMAL: interface.OPTIMAL,
    gurobipy.GRB.INFEASIBLE: interface.INFEASIBLE,
    gurobipy.GRB.INF_OR_UNBD: interface.INFEASIBLE_OR_UNBOUNDED,
    gurobipy.GRB.UNBOUNDED: interface.UNBOUNDED,
    gurobipy.GRB.CUTOFF: interface.CUTOFF,
    gurobipy.GRB.ITERATION_LIMIT: interface.ITERATION_LIMIT,
    gurobipy.GRB.NODE_LIMIT: interface.NODE_LIMIT,
    gurobipy.GRB.TIME_LIMIT: interface.TIME_LIMIT,
    gurobipy.GRB.SOLUTION_LIMIT: interface.SOLUTION_LIMIT,
    gurobipy.GRB.INTERRUPTED: interface.INTERRUPTED,
    gurobipy.GRB.NUMERIC: interface.NUMERIC,
    gurobipy.GRB.SUBOPTIMAL: interface.SUBOPTIMAL,
    gurobipy.GRB.INPROGRESS: interface.INPROGRESS
}

_LP_METHODS = {"auto": -1, "primal": 0, "dual": 1, "barrier": 2, "concurrent": 3, "deterministic_concurrent": 4}
_REVERSE_LP_METHODS = {v: k for k, v in _LP_METHODS.items()}
_QP_METHODS = {"auto": -1, "primal": 0, "dual": 1, "barrier": 2}
_REVERSE_QP_METHODS = {v: k for k, v in _QP_METHODS.items()}

_VTYPE_TO_GUROBI_VTYPE = {'continuous': gurobipy.GRB.CONTINUOUS, 'integer': gurobipy.GRB.INTEGER,
                          'binary': gurobipy.GRB.BINARY}
_GUROBI_VTYPE_TO_VTYPE = dict((val, key) for key, val in _VTYPE_TO_GUROBI_VTYPE.items())


def _constraint_lb_and_ub_to_gurobi_sense_rhs_and_range_value(lb, ub):
    """Helper function used by Constraint and Model"""
    if lb is None and ub is None:
        raise Exception("Free constraint ...")
    elif lb is None:
        sense = '<'
        rhs = float(ub)
        range_value = 0.
    elif ub is None:
        sense = '>'
        rhs = float(lb)
        range_value = 0.
    elif lb == ub:
        sense = '='
        rhs = float(lb)
        range_value = 0.
    elif lb > ub:
        raise ValueError("Lower bound is larger than upper bound.")
    else:
        sense = '='
        rhs = float(lb)
        range_value = float(ub - lb)
    return sense, rhs, range_value


def _get_solver_parameters(problem):
    """Return the parameters of a Gurobi model that differ from their defaults, in the PRM format."""
    with TemporaryFilename(suffix=".prm") as tmp_file_name:
        problem.write(tmp_file_name)
        with open(tmp_file_name) as tmp_file:
            return tmp_file.read()


def _set_solver_parameters(problem, parameters):
    """Set the parameters of a Gurobi model, given in the PRM format."""
    with TemporaryFilename(suffix=".prm", content=parameters) as tmp_file_name:
        problem.read(tmp_file_name)


class Variable(interface.Variable, metaclass=inheritdocstring):
    def __init__(self, name, *args, **kwargs):
        super(Variable, self).__init__(name, **kwargs)

    @property
    def _internal_variable(self):
        if getattr(self, 'problem', None) is not None:
            return self.problem.problem.getVarByName(self.name)
        else:
            return None

    def _set_variable_bounds_on_problem(self, var_lb, var_ub):
        lb = [
            (var.name, -gurobipy.GRB.INFINITY) if val is None else (var.name, val) for var, val in var_lb
            ]
        if len(lb) > 0:
            self.problem.variables.set_lower_bounds(lb)
        ub = [
            (var.name, gurobipy.GRB.INFINITY) if val is None else (var.name, val) for var, val in var_ub
            ]
        if len(ub) > 0:
            self.problem.variables.set_upper_bounds(ub)

    @interface.Variable.lb.setter
    def lb(self, value):
        super(Variable, self.__class__).lb.fset(self, value)
        if value is None:
            value = -gurobipy.GRB.INFINITY
        if self.problem:
            self._internal_variable.setAttr('LB', value)
            self.problem.problem.update()

    @interface.Variable.ub.setter
    def ub(self, value):
        super(Variable, self.__class__).ub.fset(self, value)
        if value is None:
            value = gurobipy.GRB.INFINITY
        if self.problem:
            self._internal_variable.setAttr('UB', value)
            self.problem.problem.update()

    def set_bounds(self, lb, ub):
        super(Variable, self).set_bounds(lb, ub)
        if self.problem:
            if lb is None:
                lb = -gurobipy.GRB.INFINITY
            if ub is None:
                ub = gurobipy.GRB.INFINITY
            var = self._internal_variable
            var.setAttr("LB", lb)
            var.setAttr("UB", ub)


    @interface.Variable.type.setter
    def type(self, value):
        super(Variable, Variable).type.fset(self, value)
        if self.problem:
            return self._internal_variable.setAttr('VType', _VTYPE_TO_GUROBI_VTYPE[value])

    def _get_primal(self):
        return self._internal_variable.getAttr('X')

    @property
    def dual(self):
        if self.problem:
            if self.problem.is_integer:
                raise ValueError("Dual values are not well-defined for integer problems")
            return self._internal_variable.getAttr('RC')
        else:
            return None

    @interface.Variable.name.setter
    def name(self, value):
        internal_var = self._internal_variable
        super(Variable, Variable).name.fset(self, value)
        if internal_var is not None:
            internal_var.setAttr('VarName', value)
            self.problem.problem.update()


class Constraint(interface.Constraint, metaclass=inheritdocstring):
    _INDICATOR_CONSTRAINT_SUPPORT = False

    def __init__(self, expression, *args, **kwargs):
        super(Constraint, self).__init__(expression, *args, **kwargs)

    def set_linear_coefficients(self, coefficients):
        if self.problem is not None:
            self.problem.update()
            grb_constraint = self.problem.problem.getConstrByName(self.name)
            for var, coeff in coefficients.items():
                self.problem.problem.chgCoeff(grb_constraint, self.problem.problem.getVarByName(var.name), float(coeff))
            self.problem.update()
        else:
            raise Exception("Can't change coefficients if constraint is not associated with a model.")

    def get_linear_coefficients(self, variables):
        if self.problem is not None:
            self.problem.update()
            grb_constraint = self.problem.problem.getConstrByName(self.name)
            return {v: self.problem.problem.getCoeff(grb_constraint, v._internal_variable) for v in variables}
        else:
            raise Exception("Can't get coefficients from solver if constraint is not in a model")

    @property
    def _internal_constraint(self):
        if self.problem is not None:
            return self.problem.problem.getConstrByName(self.name)

    def _get_expression(self):
        if self.problem is not None:
            gurobi_problem = self.problem.problem
            gurobi_constraint = self._internal_constraint
            row = gurobi_problem.getRow(gurobi_constraint)
            terms = []
            for i in range(row.size()):
                internal_var_name = row.getVar(i).VarName
                if internal_var_name == self.name + '_aux':
                    continue
                variable = self.problem._variables[internal_var_name]
                coeff = symbolics.Real(row.getCoeff(i))
                terms.append(symbolics.mul((coeff, variable)))
            self._expression = symbolics.add(terms)
        return self._expression

    def __str__(self):
        if self.problem is not None:
            self.problem.problem.update()
        return super(Constraint, self).__str__()

    @property
    def problem(self):
        return self._problem

    @problem.setter
    def problem(self, value):
        if value is None:
            # Update expression from solver instance one last time
            self._get_expression()
            self._problem = None
        else:
            self._problem = value

    @property
    def primal(self):
        if self.problem is not None:
            aux_var = self.problem.problem.getVarByName(self._internal_constraint.getAttr('ConstrName') + '_aux')
            if aux_var is not None:
                aux_val = aux_var.X
            else:
                aux_val = 0
            return (self._internal_constraint.RHS -
                    self._internal_constraint.Slack +
                    aux_val)
            # return self._round_primal_to_bounds(primal_from_solver)  # Test assertions fail
            # return primal_from_solver
        else:
            return None

    @property
    def dual(self):
        if self.problem is not None:
            if self.problem.is_integer:
                raise ValueError("Dual values are not well-defined for integer problems")
            return self._internal_constraint.Pi
        else:
            return None

    @interface.Constraint.name.setter
    def name(self, value):
        internal_const = self._internal_constraint
        super(Constraint, Constraint).name.fset(self, value)
        if internal_const is not None:
            internal_const.setAttr('ConstrName', value)
            self.problem.problem.update()

    def _set_constraint_bounds(self, sense, rhs, range_value):
        gurobi_constraint = self.problem.problem.getConstrByName(self.name)
        gurobi_constraint.setAttr('Sense', sense)
        gurobi_constraint.setAttr('RHS', rhs)

        aux_var = self.problem.problem.getVarByName(gurobi_constraint.getAttr('ConstrName') + '_aux')
        if range_value == 0.:
            if aux_var is not None:
                aux_var.setAttr("UB", 0)
        if range_value != 0:
            if aux_var is None:
                aux_var = self.problem.problem.addVar(name=gurobi_constraint.getAttr('ConstrName') + '_aux', lb=0,
                                                      ub=range_value)
                row = self.problem.problem.getRow(gurobi_constraint)
                updated_row = row - aux_var
                self.problem.problem.remove(gurobi_constraint)
                self.problem.problem.update()
                self.problem.problem.addLConstr(updated_row, sense, rhs, self.name)
            aux_var.setAttr("UB", range_value)
        self.problem.update()

    @interface.Constraint.lb.setter
    def lb(self, value):
        super(Constraint, Constraint).lb.fset(self, value)
        if getattr(self, 'problem', None) is not None:
            if value is not None and self.ub is not None and value > self.ub:
                raise ValueError(
                    "Lower bound %f is larger than upper bound %f in constraint %s" %
                    (value, self.ub, self)
                )
            sense, rhs, range_value = _constraint_lb_and_ub_to_gurobi_sense_rhs_and_range_value(value, self.ub)
            self._set_constraint_bounds(sense, rhs, range_value)

    @interface.Constraint.ub.setter
    def ub(self, value):
        super(Constraint, Constraint).ub.fset(self, value)
        if getattr(self, 'problem', None) is not None:
            if value is not None and self.lb is not None and value < self.lb:
                raise ValueError(
                    "Upper bound %f is less than lower bound %f in constraint %s" %
                    (value, self.lb, self)
                )
            sense, rhs, range_value = _constraint_lb_and_ub_to_gurobi_sense_rhs_and_range_value(self.lb, value)
            self._set_constraint_bounds(sense, rhs, range_value)

    def __iadd__(self, other):
        if self.problem is not None:
            problem_reference = self.problem
            self.problem._remove_constraint(self)
            super(Constraint, self).__iadd__(other)
            problem_reference._add_constraint(self, sloppy=False)
        else:
            super(Constraint, self).__iadd__(other)
        return self


class Objective(interface.Objective, metaclass=inheritdocstring):
    def __init__(self, expression, sloppy=False, *args, **kwargs):
        super(Objective, self).__init__(expression, *args, sloppy=sloppy, **kwargs)
        self._expression_expired = False
        if not (sloppy or self.is_Linear or self.is_Quadratic):
            raise ValueError("The given objective is invalid. Must be linear or quadratic.")

    @property
    def value(self):
        if getattr(self, "problem", None) is not None:
            gurobi_problem = self.problem.problem
            try:
                return gurobi_problem.getAttr("ObjVal") + getattr(self.problem, "_objective_offset", 0)
            except (gurobipy.GurobiError, AttributeError):  # TODO: let this check the actual error message
                return None
        else:
            return None

    @interface.Objective.direction.setter
    def direction(self, value):
        super(Objective, Objective).direction.__set__(self, value)
        if getattr(self, 'problem', None) is not None:
            self.problem.problem.setAttr('ModelSense', {'min': 1, 'max': -1}[value])

    def set_linear_coefficients(self, coefficients):
        if self.problem is not None:
            self.problem.update()
            for var, coeff in coefficients.items():
                var._internal_variable.setAttr("Obj", coeff)
            self._expression_expired = True
            self.problem.update()
        else:
            raise Exception("Can't change coefficients if objective is not associated with a model.")

    def get_linear_coefficients(self, variables):
        if self.problem is not None:
            self.problem.update()
            obj = self.problem.problem.getObjective()
            coefs = {obj.getVar(i).getAttr("varName"): obj.getCoeff(i) for i in range(obj.size())}
            return {v: coefs.get(v.name, 0) for v in variables}
        else:
            raise("Can't get coefficients from solver if objective is not in a model")

    def _get_expression(self):
        if self.problem is not None and self._expression_expired and len(self.problem._variables) > 0:
            grb_obj = self.problem.problem.getObjective()
            variables = self.problem._variables
            if self.problem.problem.IsQP:
                quadratic_expression = symbolics.add(
                    [symbolics.Real(grb_obj.getCoeff(i)) *
                     variables[grb_obj.getVar1(i).VarName] *
                     variables[grb_obj.getVar2(i).VarName]
                     for i in range(grb_obj.size())])
                linear_objective = grb_obj.getLinExpr()
            else:
                quadratic_expression = symbolics.Real(0.0)
                linear_objective = grb_obj
            linear_expression = symbolics.add(
                [symbolics.Real(linear_objective.getCoeff(i)) *
                 variables[linear_objective.getVar(i).VarName]
                 for i in range(linear_objective.size())]
            )
            self._expression = (linear_expression + quadratic_expression +
                                getattr(self.problem, "_objective_offset", 0))
            self._expression_expired = False
        return self._expression


class Configuration(interface.MathematicalProgrammingConfiguration, metaclass=inheritdocstring):
    def __init__(self, lp_method='primal', qp_method='primal', presolve=False,
                 verbosity=0, timeout=None, *args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)
        self.verbosity = verbosity
        self.lp_method = lp_method
        self.qp_method = qp_method
        self.presolve = presolve
        self.timeout = timeout
        if "tolerances" in kwargs:
            for key, val in kwargs["tolerances"].items():
                try:
                    setattr(self.tolerances, key, val)
                except AttributeError:
                    pass

    @property
    def lp_method(self):
        return _REVERSE_LP_METHODS[self.problem.problem.params.Method]

    @lp_method.setter
    def lp_method(self, value):
        if value not in _LP_METHODS:
            raise ValueError("Invalid LP method. Please choose among: " + str(list(_LP_METHODS)))
        if getattr(self, "problem", None) is not None:
            self.problem.problem.params.Method = _LP_METHODS[value]

    @property
    def qp_method(self):
        return _REVERSE_QP_METHODS[self.problem.problem.params.Method]

    @qp_method.setter
    def qp_method(self, value):
        if value not in _QP_METHODS:
            raise ValueError("Invalid LP method. Please choose among: " + str(list(_QP_METHODS)))
        if getattr(self, "problem", None) is not None:
            self.problem.problem.params.Method = _QP_METHODS[value]

    @property
    def presolve(self):
        return self._presolve

    @presolve.setter
    def presolve(self, value):
        if getattr(self, "problem", None) is not None:
            if value is True:
                self.problem.problem.params.Presolve = 2
            elif value is False:
                self.problem.problem.params.Presolve = 0
            else:
                self.problem.problem.params.Presolve = -1
        self._presolve = value

    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value):
        if self.problem is not None:
            if value == 0:
                self.problem.problem.params.OutputFlag = 0
            elif 0 < value <= 3:
                self.problem.problem.params.OutputFlag = 1
        self._verbosity = value

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        if self.problem is not None:
            if value is None:
                self.problem.problem.params.TimeLimit = gurobipy.GRB.INFINITY
            else:
                self.problem.problem.params.TimeLimit = value
        self._timeout = value

    def __getstate__(self):
        return {
            "presolve": self.presolve,
            "timeout": self.timeout,
            "verbosity": self.verbosity,
            "lp_method": self.lp_method,
            "qp_method": self.qp_method,
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

    def _get_feasibility(self):
        return getattr(self.problem.problem.params, "FeasibilityTol")

    def _set_feasibility(self, value):
        return setattr(self.problem.problem.params, "FeasibilityTol", value)

    def _get_optimality(self):
        return getattr(self.problem.problem.params, "OptimalityTol")

    def _set_optimality(self, value):
        return setattr(self.problem.problem.params, "OptimalityTol", value)

    def _get_integrality(self):
        return getattr(self.problem.problem.params, "IntFeasTol")

    def _set_integrality(self, value):
        return setattr(self.problem.problem.params, "IntFeasTol", value)

    def _tolerance_functions(self):
        return {
            "feasibility": (self._get_feasibility, self._set_feasibility),
            "optimality": (self._get_optimality, self._set_optimality),
            "integrality": (self._get_integrality, self._set_integrality)
        }


class Model(interface.Model):

    def _initialize_problem(self):
        self.problem = gurobipy.Model()
        self.problem.params.OutputFlag = 0
        if self.name is not None:
            self.problem.setAttr('ModelName', self.name)
            self.problem.update()

    def _initialize_model_from_problem(self, problem):
        if isinstance(problem, gurobipy.Model):
            self.problem = problem
            variables = []
            for gurobi_variable in self.problem.getVars():
                variables.append(Variable(
                    gurobi_variable.getAttr("VarName"),
                    lb=gurobi_variable.getAttr("lb"),
                    ub=gurobi_variable.getAttr("ub"),
                    problem=self,
                    type=_GUROBI_VTYPE_TO_VTYPE[gurobi_variable.getAttr("vType")]
                ))
            super(Model, self)._add_variables(variables)
            constraints = []
            for gurobi_constraint in self.problem.getConstrs():
                sense = gurobi_constraint.Sense
                name = gurobi_constraint.getAttr("ConstrName")
                rhs = gurobi_constraint.RHS
                row = self.problem.getRow(gurobi_constraint)
                lhs = symbolics.add(
                    [symbolics.Real(row.getCoeff(i)) * self.variables[row.getVar(i).VarName] for i in
                     range(row.size())]
                )

                if sense == '=':
                    constraint = Constraint(lhs, name=name, lb=rhs, ub=rhs, problem=self)
                elif sense == '>':
                    constraint = Constraint(lhs, name=name, lb=rhs, ub=None, problem=self)
                elif sense == '<':
                    constraint = Constraint(lhs, name=name, lb=None, ub=rhs, problem=self)
                else:
                    raise ValueError('{} is not a valid sense'.format(sense))
                constraints.append(constraint)
            super(Model, self)._add_constraints(constraints, sloppy=True)

            gurobi_objective = self.problem.getObjective()
            if self.problem.IsQP:
                quadratic_expression = symbolics.add(
                    [symbolics.Real(gurobi_objective.getCoeff(i)) *
                     self.variables[gurobi_objective.getVar1(i).VarName] *
                     self.variables[gurobi_objective.getVar2(i).VarName]
                     for i in range(gurobi_objective.size())])
                linear_objective = gurobi_objective.getLinExpr()
            else:
                quadratic_expression = symbolics.Real(0.0)
                linear_objective = gurobi_objective
            linear_expression = symbolics.add(
                [symbolics.Real(linear_objective.getCoeff(i)) *
                 self.variables[linear_objective.getVar(i).VarName]
                 for i in range(linear_objective.size())]
            )

            self._objective = Objective(
                quadratic_expression + linear_expression,
                problem=self,
                direction={1: 'min', -1: 'max'}[self.problem.getAttr('ModelSense')]
            )
        else:
            raise TypeError("Provided problem is not a valid Gurobi model.")

    @classmethod
    def from_lp(cls, lp_form):
        with TemporaryFilename(suffix=".lp", content=lp_form) as tmp_file_name:
            problem = gurobipy.read(tmp_file_name)
        model = cls(problem=problem)
        return model

    def __getstate__(self):
        self.update()
        with TemporaryFilename(suffix=".lp") as tmp_file_name:
            self.problem.write(tmp_file_name)
            with open(tmp_file_name) as tmp_file:
                lp_form = tmp_file.read()
        repr_dict = {
            'lp': lp_form,
            'status': self.status,
            'config': self.configuration.__getstate__(),
            'parameters': _get_solver_parameters(self.problem)
        }
        return repr_dict

    def __setstate__(self, repr_dict):
        with TemporaryFilename(suffix=".lp", content=repr_dict["lp"]) as tmp_file_name:
            problem = gurobipy.read(tmp_file_name)
        self.__init__(problem=problem)
        self.configuration = Configuration()
        self.configuration.problem = self
        self.configuration.__setstate__(repr_dict["config"])
        if "parameters" in repr_dict:  # absent when pickled by optlang <= 1.9.1
            _set_solver_parameters(problem, repr_dict["parameters"])

    def to_lp(self):
        self.problem.update()
        with TemporaryFilename(suffix=".lp") as tmp_file_name:
            self.problem.write(tmp_file_name)
            with open(tmp_file_name) as tmp_file:
                lp_form = tmp_file.read()
        return lp_form

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, value):
        super(Model, self.__class__).objective.fset(self, value)
        expression = self._objective._expression

        offset, linear_coefficients, quadratic_coefficients = parse_optimization_expression(
            value, quadratic=True, expression=expression
        )
        # self.problem.setAttr("ObjCon", offset) # Does not seem to work
        self._objective_offset = offset
        grb_terms = []
        for var, coef in linear_coefficients.items():
            coef = float(coef)
            var = self.problem.getVarByName(var.name)
            grb_terms.append(coef * var)
        for key, coef in quadratic_coefficients.items():
            coef = float(coef)
            if len(key) == 1:
                var = next(iter(key))
                var = self.problem.getVarByName(var.name)
                grb_terms.append(coef * var * var)
            else:
                var1, var2 = key
                var1 = self.problem.getVarByName(var1.name)
                var2 = self.problem.getVarByName(var2.name)
                grb_terms.append(coef * var1 * var2)

        grb_expression = gurobipy.quicksum(grb_terms)

        self.problem.setObjective(grb_expression,
                                  {'min': gurobipy.GRB.MINIMIZE, 'max': gurobipy.GRB.MAXIMIZE}[value.direction])
        value.problem = self

    def update(self):
        super(Model, self).update(callback=self.problem.update)

    def _optimize(self):
        if self.status != interface.OPTIMAL:
            self.problem.reset()
        self.problem.update()
        self.problem.optimize()
        status = _GUROBI_STATUS_TO_STATUS[self.problem.getAttr("Status")]
        return status

    def _add_variables(self, variables):
        super(Model, self)._add_variables(variables)
        for variable in variables:
            if variable.lb is None:
                lb = -gurobipy.GRB.INFINITY
            else:
                lb = variable.lb
            if variable.ub is None:
                ub = gurobipy.GRB.INFINITY
            else:
                ub = variable.ub
            self.problem.addVar(
                name=variable.name,
                lb=lb,
                ub=ub,
                vtype=_VTYPE_TO_GUROBI_VTYPE[variable.type]
            )

    def _remove_variables(self, variables):
        # Not calling parent method to avoid expensive variable removal from sympy expressions
        self.objective._expression = self.objective.expression.xreplace({var: 0 for var in variables})
        for variable in variables:
            internal_variable = variable._internal_variable
            del self._variables_to_constraints_mapping[variable.name]
            variable.problem = None
            del self._variables[variable.name]
            self.problem.remove(internal_variable)

    def _add_constraints(self, constraints, sloppy=False):
        super(Model, self)._add_constraints(constraints, sloppy=sloppy)
        for constraint in constraints:
            if constraint.lb is None and constraint.ub is None:
                raise ValueError("optlang does not support free constraints in the gurobi interface.")
            self.problem.update()
            constraint._problem = None
            if constraint.is_Linear:
                offset, coef_dict, _ = parse_optimization_expression(constraint, linear=True)

                lhs = gurobipy.quicksum([float(coef) * var._internal_variable for var, coef in coef_dict.items()])
                sense, rhs, range_value = _constraint_lb_and_ub_to_gurobi_sense_rhs_and_range_value(constraint.lb,
                                                                                                    constraint.ub)

                if range_value != 0:
                    aux_var = self.problem.addVar(name=constraint.name + '_aux', lb=0, ub=range_value)
                    self.problem.update()
                    lhs = lhs - aux_var

                self.problem.addLConstr(lhs, sense, rhs, name=constraint.name)
            else:
                raise ValueError(
                    "GUROBI currently only supports linear constraints. %s is not linear." % self)
                # self.problem.addQConstr(lhs, sense, rhs)
            constraint.problem = self
        self.problem.update()

    def _remove_constraints(self, constraints):
        self.problem.update()
        internal_constraints = [constraint._internal_constraint for constraint in constraints]
        super(Model, self)._remove_constraints(constraints)
        for internal_constraint in internal_constraints:
            self.problem.remove(internal_constraint)

    def _get_variables_names(self):
        """The names of model variables.

        Returns
        -------
        list
        """
        return self.problem.VarName

    def _get_constraint_names(self):
        """The names of model constraints.

        Returns
        -------
        list
        """
        return self.problem.ConstrName

    def _get_primal_values(self):
        """The primal values of model variables.

        Returns
        -------
        list
        """
        return self.problem.X

    def _get_reduced_costs(self):
        """The reduced costs/dual values of all variables.

        Returns
        -------
        list
        """
        if self.is_integer:
            raise ValueError(
                "Reduced costs are not well defined for integer problems.")
        if self.is_mip:
            raise ValueError(
                "Reduced costs are not well defined for mixed integer problems. Try relaxing your model first.")
        return self.problem.RC

    def _get_shadow_prices(self):
        """The shadow prices of model (dual values of all constraints).

        Returns
        -------
        collections.OrderedDict
        """
        if self.is_integer:
            raise ValueError(
                "Shadow prices are not well defined for integer problems.")
        if self.is_mip:
            raise ValueError(
                "Shadow prices are not well defined for mixed integer problems. Try relaxing your model first.")
        return self.problem.Pi

    @property
    def is_integer(self):
        self.problem.update()
        return self.problem.NumIntVars > 0

    @property
    def is_mip(self):
        self.problem.update()
        return self.problem.IsMIP == 1


if __name__ == '__main__':
    x = Variable('x', lb=0, ub=10)
    y = Variable('y', lb=0, ub=10)
    z = Variable('z', lb=-100, ub=99.)
    constr = Constraint(0.3 * x + 0.4 * y + 66. * z, lb=-100, ub=0., name='test')

    solver = Model(problem=gurobipy.read("tests/data/model.lp"))

    solver.add(z)
    solver.add(constr)
    print(solver)
    print(solver.optimize())
    print(solver.status)
