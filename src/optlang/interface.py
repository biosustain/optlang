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
Abstract solver interface definitions (:class:`Model`, :class:`Variable`,
:class:`Constraint`, :class:`Objective`) intended to be subclassed and
extended for individual solvers.

This module defines the API of optlang objects and indicates which methods need to be implemented in
subclassed solver interfaces.
The classes in this module can be used to construct and modify problems, but no optimizations can be done.
"""
from collections import OrderedDict

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import inspect
import logging
import sys
import uuid
import warnings
import sympy

import optlang
from optlang.exceptions import IndicatorConstraintsNotSupported

from optlang.util import parse_expr, expr_to_json, is_numeric, SolverTolerances
from optlang import symbolics

from .container import Container

log = logging.getLogger(__name__)

OPTIMAL = 'optimal'
UNDEFINED = 'undefined'
FEASIBLE = 'feasible'
INFEASIBLE = 'infeasible'
NOFEASIBLE = 'nofeasible'
UNBOUNDED = 'unbounded'
INFEASIBLE_OR_UNBOUNDED = 'infeasible_or_unbounded'
LOADED = 'loaded'
CUTOFF = 'cutoff'
ITERATION_LIMIT = 'iteration_limit'
MEMORY_LIMIT = 'memory_limit'
NODE_LIMIT = 'node_limit'
TIME_LIMIT = 'time_limit'
SOLUTION_LIMIT = 'solution_limit'
INTERRUPTED = 'interrupted'
NUMERIC = 'numeric'
SUBOPTIMAL = 'suboptimal'
INPROGRESS = 'in_progress'
ABORTED = 'aborted'
SPECIAL = 'check_original_solver_status'

statuses = {
    OPTIMAL: "An optimal solution as been found.",
    INFEASIBLE: "The problem has no feasible solutions.",
    UNBOUNDED: "The objective can be optimized infinitely.",
    SPECIAL: "The status returned by the solver could not be interpreted. Please refer to the solver's documentation to find the status.",
    UNDEFINED: "The solver determined that the problem is ill-formed. "
    # TODO Add the rest
}


# noinspection PyShadowingBuiltins
class Variable(symbolics.Symbol):
    """Optimization variables.

    Variable objects are used to represents each variable of the optimization problem. When the optimization is
    performed, the combination of variable values that optimizes the objective function, while not violating any
    constraints will be identified. The type of a variable ('continuous', 'integer' or 'binary') can be set using
    the type keyword of the constructor or it can be changed after initialization by :code:`var.type = 'binary'`.

    The variable class subclasses the :code:`sympy.Symbol` class, which means that symbolic expressions of variables
    can be constructed by using regular python syntax, e.g. :code:`my_expression = 2 * var1 + 3 * var2 ** 2`.
    Expressions like this are used when constructing Constraint and Objective objects.
    Once a problem has been optimized, the primal and dual values of a variable can be accessed from the
    primal and dual attributes, respectively.

    Attributes
    ----------
    name: str
        The variable's name.
    lb: float or None, optional
        The lower bound, if None then -inf.
    ub: float or None, optional
        The upper bound, if None then inf.
    type: str, optional
        The variable type, 'continuous' or 'integer' or 'binary'.
    problem: Model or None, optional
        A reference to the optimization model the variable belongs to.

    Examples
    --------
    >>> Variable('x', lb=-10, ub=10)
    '-10 <= x <= 10'
    """

    @staticmethod
    def __test_valid_lower_bound(type, value, name):
        if not (value is None or is_numeric(value)):
            raise TypeError("Variable bounds must be numeric or None.")
        if value is not None:
            if type == 'integer' and value % 1 != 0.:
                raise ValueError(
                    'The provided lower bound %g cannot be assigned to integer variable %s (%g mod 1 != 0).' % (
                        value, name, value))
        if type == 'binary' and (value is None or value not in (0, 1)):
            raise ValueError(
                'The provided lower bound %s cannot be assigned to binary variable %s.' % (value, name))

    @staticmethod
    def __test_valid_upper_bound(type, value, name):
        if not (value is None or is_numeric(value)):
            raise TypeError("Variable bounds must be numeric or None.")
        if value is not None:
            if type == 'integer' and value % 1 != 0.:
                raise ValueError(
                    'The provided upper bound %s cannot be assigned to integer variable %s (%s mod 1 != 0).' % (
                        value, name, value))
        if type == 'binary' and (value is None or value not in (0, 1)):
            raise ValueError(
                'The provided upper bound %s cannot be assigned to binary variable %s.' % (value, name))

    @staticmethod
    def __validate_variable_name(name):
        if len(name) < 1:
            raise ValueError('Variable name must not be empty string')
        for char in name:
            if char.isspace():
                raise ValueError(
                    'Variable names cannot contain whitespace characters. "%s" contains whitespace character "%s".' % (
                        name, char)
                )

    @classmethod
    def clone(cls, variable, **kwargs):
        """
        Make a copy of another variable. The variable being copied can be of the same type or belong to
        a different solver interface.

        Example
        ----------
        >>> var_copy = Variable.clone(old_var)
        """
        return cls(variable.name, lb=variable.lb, ub=variable.ub, type=variable.type, **kwargs)

    def __init__(self, name, lb=None, ub=None, type="continuous", problem=None, *args, **kwargs):

        self.__validate_variable_name(name)

        self._name = name
        symbolics.Symbol.__init__(self, name, *args, **kwargs)
        self._lb = lb
        self._ub = ub
        if self._lb is None and type == 'binary':
            self._lb = 0.
        if self._ub is None and type == 'binary':
            self._ub = 1.
        self.__test_valid_lower_bound(type, self._lb, name)
        self.__test_valid_upper_bound(type, self._ub, name)
        self.problem = None
        self.type = type
        self.problem = problem

    @property
    def name(self):
        """Name of variable."""
        return self._name

    @name.setter
    def name(self, value):
        self.__validate_variable_name(value)
        old_name = getattr(self, 'name', None)
        self._name = value
        if getattr(self, 'problem', None) is not None and value != old_name:
            self.problem.variables.update_key(old_name)
            self.problem._variables_to_constraints_mapping[value] = self.problem._variables_to_constraints_mapping[old_name]
            del self.problem._variables_to_constraints_mapping[old_name]

    @property
    def lb(self):
        """Lower bound of variable."""
        return self._lb

    @lb.setter
    def lb(self, value):
        if hasattr(self, 'ub') and self.ub is not None and value is not None and value > self.ub:
            raise ValueError(
                'The provided lower bound %g is larger than the upper bound %g of variable %s.' % (
                    value, self.ub, self))
        self.__test_valid_lower_bound(self.type, value, self.name)
        self._lb = value
        if self.problem is not None:
            self.problem._pending_modifications.var_lb.append((self, value))

    @property
    def ub(self):
        """Upper bound of variable."""
        return self._ub

    @ub.setter
    def ub(self, value):
        if hasattr(self, 'lb') and self.lb is not None and value is not None and value < self.lb:
            raise ValueError(
                'The provided upper bound %g is smaller than the lower bound %g of variable %s.' % (
                    value, self.lb, self))
        self.__test_valid_upper_bound(self.type, value, self.name)
        self._ub = value
        if self.problem is not None:
            self.problem._pending_modifications.var_ub.append((self, value))

    def set_bounds(self, lb, ub):
        """
        Change the lower and upper bounds of a variable.
        """
        if lb is not None and ub is not None and lb > ub:
            raise ValueError(
                "The provided lower bound {} is larger than the provided upper bound {}".format(lb, ub)
            )
        self._lb = lb
        self._ub = ub
        if self.problem is not None:
            self.problem._pending_modifications.var_lb.append((self, lb))
            self.problem._pending_modifications.var_ub.append((self, ub))

    @property
    def type(self):
        """Variable type ('either continuous, integer, or binary'.)"""
        return self._type

    @type.setter
    def type(self, value):
        if value == 'continuous':
            self._type = value
        elif value == 'integer':
            self._type = value
            try:
                self.lb = round(self.lb)
            except TypeError:
                pass
            try:
                self.ub = round(self.ub)
            except TypeError:
                pass
        elif value == 'binary':
            self._type = value
            self._lb = 0
            self._ub = 1
        else:
            raise ValueError(
                "'%s' is not a valid variable type. Choose between 'continuous, 'integer', or 'binary'." % value)

    @property
    def primal(self):
        """The primal of variable (None if no solution exists)."""
        if self.problem:
            return self._get_primal()
        else:
            return None

    def _get_primal(self):
        return None

    @property
    def dual(self):
        """The dual of variable (None if no solution exists)."""
        return None

    def __str__(self):
        """Print a string representation of variable.

        Examples
        --------
        >>> Variable('x', lb=-10, ub=10)
        '-10 <= x <= 10'
        """
        if self.lb is not None:
            lb_str = str(self.lb) + " <= "
        else:
            lb_str = ""
        if self.ub is not None:
            ub_str = " <= " + str(self.ub)
        else:
            ub_str = ""
        return ''.join((lb_str, super(Variable, self).__str__(), ub_str))

    def __repr__(self):
        """Does exactly the same as __str__ for now."""
        return self.__str__()

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def __reduce__(self):
        return (type(self), (self.name, self.lb, self.ub, self.type, self.problem))

    def to_json(self):
        """
        Returns a json-compatible object from the Variable that can be saved using the json module.

        Example
        --------
        >>> import json
        >>> with open("path_to_file.json", "w") as outfile:
        >>>     json.dump(var.to_json(), outfile)
        """
        json_obj = {
            "name": self.name,
            "lb": self.lb,
            "ub": self.ub,
            "type": self.type
        }
        return json_obj

    @classmethod
    def from_json(cls, json_obj):
        """
        Constructs a Variable from the provided json-object.

        Example
        --------
        >>> import json
        >>> with open("path_to_file.json") as infile:
        >>>     var = Variable.from_json(json.load(infile))
        """
        return cls(json_obj["name"], lb=json_obj["lb"], ub=json_obj["ub"], type=json_obj["type"])

    # def _round_primal_to_bounds(self, primal, tolerance=1e-5):
    #     """Rounds primal value to lie within variables bounds.
    #
    #     Raises if exceeding threshold.
    #
    #     Parameters
    #     ----------
    #     primal : float
    #         The primal value to round.
    #     tolerance : float (optional)
    #         The tolerance threshold (default: 1e-5).
    #     """
    #     if (self.lb is None or primal >= self.lb) and (self.ub is None or primal <= self.ub):
    #         return primal
    #     else:
    #         if (primal <= self.lb) and ((self.lb - primal) <= tolerance):
    #             return self.lb
    #         elif (primal >= self.ub) and ((self.ub - primal) >= -tolerance):
    #             return self.ub
    #         else:
    #             raise AssertionError(
    #                 'The primal value %s returned by the solver is out of bounds for variable %s (lb=%s, ub=%s)' % (
    #                     primal, self.name, self.lb, self.ub))


# noinspection PyPep8Naming
class OptimizationExpression(object):
    """Abstract base class for Objective and Constraint."""

    @staticmethod
    def _validate_optimization_expression_name(name):
        if name is None:
            return
        if len(name) < 1:
            raise ValueError('Name must not be empty string')
        for char in name:
            if char.isspace():
                raise ValueError(
                    'Names cannot contain whitespace characters. "%s" contains whitespace character "%s".' % (
                        name, char)
                )

    @classmethod
    def _substitute_variables(cls, expression, model=None, **kwargs):
        """Substitutes variables in (optimization)expression (constraint/objective) with variables of the appropriate interface type.
        Attributes
        ----------
        expression: Constraint, Objective
            An optimization expression.
        model: Model or None, optional
            A reference to an optimization model that should be searched for appropriate variables first.
        """
        interface = sys.modules[cls.__module__]
        variable_substitutions = dict()
        for variable in expression.variables:
            if model is not None and variable.name in model.variables:
                variable_substitutions[variable] = model.variables[variable.name]
            else:
                variable_substitutions[variable] = interface.Variable.clone(variable)
        adjusted_expression = expression.expression.xreplace(variable_substitutions)
        return adjusted_expression

    def __init__(self, expression, name=None, problem=None, sloppy=False, *args, **kwargs):

        self._validate_optimization_expression_name(name)

        super(OptimizationExpression, self).__init__(*args, **kwargs)
        self._problem = problem
        if sloppy:
            self._expression = expression
        else:
            self._expression = self._canonicalize(expression)
        if name is None:
            self._name = str(uuid.uuid1())
        else:
            self._name = name

    @property
    def name(self):
        """The name of the object"""
        return self._name

    @name.setter
    def name(self, value):
        self._validate_optimization_expression_name(value)
        self._name = value

    @property
    def problem(self):
        """A reference to the model that the object belongs to (or None)"""
        return getattr(self, '_problem', None)

    @problem.setter
    def problem(self, value):
        self._problem = value

    def _get_expression(self):
        return self._expression

    @property
    def expression(self):
        """The mathematical expression defining the objective/constraint."""
        return self._get_expression()

    @property
    def variables(self):
        """Variables in constraint/objective's expression."""
        return self.expression.atoms(Variable)

    def _canonicalize(self, expression):
        if isinstance(expression, float):
            return symbolics.Real(expression)
        elif isinstance(expression, int):
            return symbolics.Integer(expression)
        else:
            if optlang._USING_SYMENGINE:
                expression = expression.expand()  # This is a good way to canonicalize, but is quite slow for sympy
            return expression

    @property
    def is_Linear(self):
        """Returns True if expression is linear (a polynomial with degree 1 or 0) (read-only)."""
        coeff_dict = self.expression.as_coefficients_dict()
        for key in coeff_dict.keys():
            if len(key.free_symbols) < 2 and (key.is_Add or key.is_Mul or key.is_Atom):
                pass
            else:
                return False
            if key.is_Pow and key.args[1] != 1:
                return False
        else:
            return True

    @property
    def is_Quadratic(self):
        """Returns True if the expression is a polynomial with degree exactly 2 (read-only)."""
        if self.expression.is_Atom:
            return False
        if all((len(key.free_symbols) < 2 and (key.is_Add or key.is_Mul or key.is_Atom)
                for key in self.expression.as_coefficients_dict().keys())):
            return False
        if self.expression.is_Add:
            terms = self.expression.args
            is_quad = False
            for term in terms:
                if len(term.free_symbols) > 2:
                    return False
                if term.is_Pow:
                    if not term.args[1].is_Number or term.args[1] > 2:
                        return False
                    else:
                        is_quad = True
                elif term.is_Mul:
                    if len(term.free_symbols) == 2:
                        is_quad = True
                    if term.args[1].is_Pow:
                        if not term.args[1].args[1].is_Number or term.args[1].args[1] > 2:
                            return False
                        else:
                            is_quad = True
            return is_quad
        else:
            if isinstance(self.expression, sympy.Basic):
                sympy_expression = self.expression
            else:
                sympy_expression = sympy.sympify(self.expression)
            # TODO: Find a way to do this with symengine (Poly is not part of symengine, 23 March 2017)
            poly = sympy_expression.as_poly(*sympy_expression.atoms(sympy.Symbol))
            if poly is None:
                return False
            else:
                return poly.is_quadratic

    def __iadd__(self, other):
        self._expression += other
        return self

    def __isub__(self, other):
        self._expression -= other
        return self

    def __imul__(self, other):
        self._expression *= other
        return self

    def __idiv__(self, other):
        self._expression /= other
        return self

    def __itruediv__(self, other):
        self._expression /= other
        return self

    def set_linear_coefficients(self, coefficients):
        """Set coefficients of linear terms in constraint or objective.
        Existing coefficients for linear or non-linear terms will not be modified.

        Note: This method interacts with the low-level solver backend and can only be used on objects that are
        associated with a Model. The method is not part of optlangs basic interface and should be used mainly where
        speed is important.

        Parameters
        ----------
        coefficients : dict
            A dictionary like {variable1: coefficient1, variable2: coefficient2, ...}

        Returns
        -------
        None
        """
        raise NotImplementedError("Child classes should implement this.")

    def get_linear_coefficients(self, variables):
        """Get coefficients of linear terms in constraint or objective.

        Note: This method interacts with the low-level solver backend and can only be used on objects that are
        associated with a Model. The method is not part of optlangs basic interface and should be used mainly where
        speed is important.

        Parameters
        ----------
        variables : iterable
            An iterable of Variable objects

        Returns
        -------
        Coefficients : dict
            {var1: coefficient, var2: coefficient ...}
        """
        raise NotImplementedError("Child classes should implement this.")


class Constraint(OptimizationExpression):
    """
    Constraint objects represent the mathematical (in-)equalities that constrain an optimization problem.
    A constraint is formulated by a symbolic expression of variables and a lower and/or upper bound.
    Equality constraints can be formulated by setting the upper and lower bounds to the same value.

    Some solvers support indicator variables. This lets a binary variable act as a switch that decides whether
    the constraint should be active (cannot be violated) or inactive (can be violated).

    The constraint expression can be an arbitrary combination of variables, however the individual solvers
    have limits to the forms of constraints they allow. Most solvers only allow linear constraints, meaning that
    the expression should be of the form :code:`a * var1 + b * var2 + c * var3 ...`

    Attributes
    ----------
    expression: sympy
        The mathematical expression defining the constraint.
    name: str, optional
        The constraint's name.
    lb: float or None, optional
        The lower bound, if None then -inf.
    ub: float or None, optional
        The upper bound, if None then inf.
    indicator_variable: Variable
        The indicator variable (needs to be binary).
    active_when: 0 or 1 (default 0)
        When the constraint should
    problem: Model or None, optional
        A reference to the optimization model the variable belongs to.

    Examples
    ----------
    >>> expr = 2.4 * var1 - 3.8 * var2
    >>> c1 = Constraint(expr, lb=0, ub=10)

    >>> indicator_var = Variable("var3", type="binary") # Only possible with some solvers
    >>> c2 = Constraint(var2, lb=0, ub=0, indicator_variable=indicator_var, active_when=1) # When the indicator is 1, var2 is constrained to be 0
    """


    _INDICATOR_CONSTRAINT_SUPPORT = True

    def _check_valid_lower_bound(self, value):
        if not (value is None or is_numeric(value)):
            raise TypeError("Constraint bounds must be numeric or None, not {}".format(type(value)))
        if value is not None and getattr(self, "ub", None) is not None and value > self.ub:
            raise ValueError("Cannot set a lower bound that is greater than the upper bound.")

    def _check_valid_upper_bound(self, value):
        if not (value is None or is_numeric(value)):
            raise TypeError("Constraint bounds must be numeric or None, not {}".format(type(value)))
        if value is not None and getattr(self, "lb", None) is not None and value < self.lb:
            raise ValueError("Cannot set an upper bound that is less than the lower bound.")

    @classmethod
    def __check_valid_indicator_variable(cls, variable):
        if variable is not None and not cls._INDICATOR_CONSTRAINT_SUPPORT:
            raise IndicatorConstraintsNotSupported(
                'Solver interface %s does not support indicator constraints' % cls.__module__)
        if variable is not None and variable.type != 'binary':
            raise ValueError('Provided indicator variable %s is not binary.' % variable)

    @staticmethod
    def __check_valid_active_when(active_when):
        if active_when != 0 and active_when != 1:
            raise ValueError('Provided active_when argument %s needs to be either 1 or 0' % active_when)

    @classmethod
    def clone(cls, constraint, model=None, **kwargs):
        """
        Make a copy of another constraint. The constraint being copied can be of the same type or belong to
        a different solver interface.

        Parameters
        ----------
        constraint: interface.Constraint (or subclass)
            The constraint to copy
        model: Model or None
            The variables of the new constraint will be taken from this model. If None, new variables will be
            constructed.

        Example
        ----------
        >>> const_copy = Constraint.clone(old_constraint)
        """
        return cls(cls._substitute_variables(constraint, model=model), lb=constraint.lb, ub=constraint.ub,
                   indicator_variable=constraint.indicator_variable, active_when=constraint.active_when,
                   name=constraint.name, sloppy=True, **kwargs)

    def __init__(self, expression, lb=None, ub=None, indicator_variable=None, active_when=1, *args, **kwargs):
        self._problem = None
        self.lb = lb
        self.ub = ub
        super(Constraint, self).__init__(expression, *args, **kwargs)
        self.__check_valid_indicator_variable(indicator_variable)
        self.__check_valid_active_when(active_when)
        self._indicator_variable = indicator_variable
        self._active_when = active_when

    @property
    def lb(self):
        """Lower bound of constraint."""
        return self._lb

    @lb.setter
    def lb(self, value):
        self._check_valid_lower_bound(value)
        self._lb = value

    @property
    def ub(self):
        """Upper bound of constraint."""
        return self._ub

    @ub.setter
    def ub(self, value):
        self._check_valid_upper_bound(value)
        self._ub = value

    @OptimizationExpression.name.setter
    def name(self, value):
        old_name = getattr(self, 'name', None)
        super(Constraint, Constraint).name.fset(self, value)
        if getattr(self, 'problem', None) is not None and value != old_name:
            self.problem.constraints.update_key(old_name)

    @property
    def indicator_variable(self):
        """The indicator variable of constraint (if available)."""
        return self._indicator_variable

    # @indicator_variable.setter
    # def indicator_variable(self, value):
    #     self.__check_valid_indicator_variable(value)
    #     self._indicator_variable = value

    @property
    def active_when(self):
        """Activity relation of constraint to indicator variable (if supported)."""
        return self._active_when

    def __str__(self):
        if self.lb is not None:
            lhs = str(self.lb) + ' <= '
        else:
            lhs = ''
        if self.ub is not None:
            rhs = ' <= ' + str(self.ub)
        else:
            rhs = ''
        if self.indicator_variable is not None:
            lhs = self.indicator_variable.name + ' = ' + str(self.active_when) + ' -> ' + lhs
        return str(self.name) + ": " + lhs + self.expression.__str__() + rhs

    def _canonicalize(self, expression):
        expression = super(Constraint, self)._canonicalize(expression)
        if expression.is_Atom or expression.is_Mul:
            return expression
        lonely_coeffs = [arg for arg in expression.args if arg.is_Number]
        if not lonely_coeffs:
            return expression
        assert len(lonely_coeffs) == 1
        coeff = lonely_coeffs[0]
        expression = expression - coeff
        if self.lb is not None and self.ub is not None:
            oldub = self.ub
            self.ub = None
            self.lb = self.lb - float(coeff)
            self.ub = oldub - float(coeff)
        elif self.lb is not None:
            self.lb = self.lb - float(coeff)
        elif self.ub is not None:
            self.ub = self.ub - float(coeff)
        else:
            raise ValueError(
                "%s cannot be shaped into canonical form if neither lower or upper constraint bounds are set."
                % expression
            )
        return expression

    @property
    def primal(self):
        """Primal of constraint (None if no solution exists)."""
        return None

    @property
    def dual(self):
        """Dual of constraint (None if no solution exists)."""
        return None

    def _round_primal_to_bounds(self, primal, tolerance=1e-5):
        if (self.lb is None or primal >= self.lb) and (self.ub is None or primal <= self.ub):
            return primal
        else:
            if (primal <= self.lb) and ((self.lb - primal) <= tolerance):
                return self.lb
            elif (primal >= self.ub) and ((self.ub - primal) >= -tolerance):
                return self.ub
            else:
                raise AssertionError(
                    'The primal value %s returned by the solver is out of bounds for variable %s (lb=%s, ub=%s)' % (
                        primal, self.name, self.lb, self.ub))

    def to_json(self):
        """
        Returns a json-compatible object from the constraint that can be saved using the json module.

        Example
        --------
        >>> import json
        >>> with open("path_to_file.json", "w") as outfile:
        >>>     json.dump(constraint.to_json(), outfile)
        """
        if self.indicator_variable is None:
            indicator = None
        else:
            indicator = self.indicator_variable.name
        json_obj = {
            "name": self.name,
            "expression": expr_to_json(self.expression),
            "lb": self.lb,
            "ub": self.ub,
            "indicator_variable": indicator,
            "active_when": self.active_when
        }
        return json_obj

    @classmethod
    def from_json(cls, json_obj, variables=None):
        """
        Constructs a Variable from the provided json-object.

        Example
        --------
        >>> import json
        >>> with open("path_to_file.json") as infile:
        >>>     constraint = Constraint.from_json(json.load(infile))
        """
        if variables is None:
            variables = {}
        expression = parse_expr(json_obj["expression"], variables)
        if json_obj["indicator_variable"] is None:
            indicator = None
        else:
            indicator = variables[json_obj["indicator_variable"]]
        return cls(
            expression,
            name=json_obj["name"],
            lb=json_obj["lb"],
            ub=json_obj["ub"],
            indicator_variable=indicator,
            active_when=json_obj["active_when"]
        )


class Objective(OptimizationExpression):
    """
    Objective objects are used to represent the objective function of an optimization problem.
    An objective consists of a symbolic expression of variables in the problem and a direction. The direction
    can be either 'min' or 'max' and specifies whether the problem is a minimization or a maximization problem.

    After a problem has been optimized, the optimal objective value can be accessed from the 'value' attribute
    of the model's objective, i.e. :code:`obj_val = model.objective.value`.

    Attributes
    ----------
    expression: sympy
        The mathematical expression defining the objective.
    name: str, optional
        The name of the constraint.
    direction: 'max' or 'min'
        The optimization direction.
    value: float, read-only
        The current objective value.
    problem: solver
        The low-level solver object.

    """

    @classmethod
    def clone(cls, objective, model=None, **kwargs):
        """
        Make a copy of an objective. The objective being copied can be of the same type or belong to
        a different solver interface.

        Example
        ----------
        >>> new_objective = Objective.clone(old_objective)
        """
        return cls(cls._substitute_variables(objective, model=model), name=objective.name,
                   direction=objective.direction, sloppy=True, **kwargs)

    def __init__(self, expression, value=None, direction='max', *args, **kwargs):
        self._value = value
        self._direction = direction
        super(Objective, self).__init__(expression, *args, **kwargs)

    @property
    def value(self):
        """The objective value."""
        return self._value

    def __str__(self):
        return {'max': 'Maximize', 'min': 'Minimize'}[self.direction] + '\n' + str(self.expression)
        # return ' '.join((self.direction, str(self.expression)))

    def __eq__(self, other):
        """Tests *mathematical* equality for two Objectives. Solver specific type does NOT have to match.
        Expression and direction must be the same.
        Name does not have to match"""
        if isinstance(other, Objective):
            return self.expression == other.expression and self.direction == other.direction
        else:
            return False
            #

    def _canonicalize(self, expression):
        """For example, changes x + y to 1.*x + 1.*y"""
        expression = super(Objective, self)._canonicalize(expression)
        if isinstance(expression, sympy.Basic):
            expression *= 1.
        else:  # pragma: no cover   # symengine
            expression = (1. * expression).expand()
        return expression

    @property
    def direction(self):
        """The direction of optimization. Either 'min' or 'max'."""
        return self._direction

    @direction.setter
    def direction(self, value):
        if value not in ['max', 'min']:
            raise ValueError("Provided optimization direction %s is neither 'min' or 'max'." % value)
        self._direction = value

    def set_linear_coefficients(self, coefficients):
        """Set linear coefficients in objective.

        coefficients : dict
            A dictionary of the form {variable1: coefficient1, variable2: coefficient2, ...}
        """
        raise NotImplementedError("Child class should implement this.")

    def to_json(self):
        """
        Returns a json-compatible object from the objective that can be saved using the json module.

        Example
        --------
        >>> import json
        >>> with open("path_to_file.json", "w") as outfile:
        >>>     json.dump(obj.to_json(), outfile)
        """
        json_obj = {
            "name": self.name,
            "expression": expr_to_json(self.expression),
            "direction": self.direction
        }
        return json_obj

    @classmethod
    def from_json(cls, json_obj, variables=None):
        """
        Constructs an Objective from the provided json-object.

        Example
        --------
        >>> import json
        >>> with open("path_to_file.json") as infile:
        >>>     obj = Objective.from_json(json.load(infile))
        """
        if variables is None:
            variables = {}
        expression = parse_expr(json_obj["expression"], variables)
        return cls(
            expression,
            direction=json_obj["direction"],
            name=json_obj["name"]
        )


class Configuration(object):
    """
    Optimization solver configuration.
    This object allows the user to change certain parameters and settings in the solver.
    It is meant to allow easy access to a few common and important parameters. For information on changing
    other solver parameters, please consult the documentation from the solver provider.
    Some changeable parameters are listed below. Note that some solvers might not implement all of these
    and might also implement additional parameters.

    Attributes
    ----------
    verbosity: int from 0 to 3
        Changes the level of output.
    timeout: int or None
        The time limit in second the solver will use to optimize the problem.
    presolve: Boolean or 'auto'
        Tells the solver whether to use (solver-specific) pre-processing to simplify the problem.
        This can decrease solution time, but also introduces overhead. If set to 'auto' the solver will
        first try to solve without pre-processing, and only turn in on in case no optimal solution can be found.
    lp_method: str
        Select which algorithm the LP solver uses, e.g. simplex, barrier, etc.

    """

    @classmethod
    def clone(cls, config, problem=None, **kwargs):
        return cls(problem=problem, **cls._get_parameters(config))

    @classmethod
    def _get_parameters(cls, config):
        """Collect the parameters of a configuration as keyword arguments for the constructor."""
        properties = (k for k, v in inspect.getmembers(cls, predicate=inspect.isdatadescriptor) if
                      not k.startswith('__'))
        parameters = {property: getattr(config, property) for property in properties if hasattr(config, property)}
        if hasattr(config, "tolerances"):
            parameters["tolerances"] = config.tolerances.to_dict()
        return parameters

    def __init__(self, problem=None, *args, **kwargs):
        self.problem = problem
        self._add_tolerances()

    @property
    def verbosity(self):
        """Verbosity level.

        0: no output
        1: error and warning messages only
        2: normal output
        3: full output
        """
        raise NotImplementedError

    @verbosity.setter
    def verbosity(self, value):
        raise NotImplementedError

    @property
    def timeout(self):
        """Timeout parameter (seconds)."""
        raise NotImplementedError

    @timeout.setter
    def timeout(self):
        raise NotImplementedError

    @property
    def presolve(self):
        """
        Turn pre-processing on or off. Set to 'auto' to only use presolve if no optimal solution can be found.
        """
        raise NotImplementedError

    @presolve.setter
    def presolve(self):
        raise NotImplementedError

    def _add_tolerances(self):
        self.tolerances = SolverTolerances(self._tolerance_functions())

    def _tolerance_functions(self):
        """
        This should be implemented in child classes. Must return a dict, where keys are available tolerance parameters
        and values are tuples of (getter_function, setter_function).
        The getter functions must be callable with no arguments and the setter functions must be callable with the
        new value as the only argument
        """
        return {}

    def __setstate__(self, state):
        self.__init__()


class MathematicalProgrammingConfiguration(Configuration):
    def __init__(self, *args, **kwargs):
        super(MathematicalProgrammingConfiguration, self).__init__(*args, **kwargs)

    @property
    def presolve(self):
        """If the presolver should be used (if available)."""
        raise NotImplementedError

    @presolve.setter
    def presolve(self, value):
        raise NotImplementedError


class EvolutionaryOptimizationConfiguration(Configuration):
    """docstring for HeuristicOptimization"""

    def __init__(self, *args, **kwargs):
        super(EvolutionaryOptimizationConfiguration, self).__init__(*args, **kwargs)


class Model(object):
    """
    The model object represents an optimization problem and contains the variables, constraints an objective that
    make up the problem. Variables and constraints can be added and removed using the :code:`.add` and :code:`.remove` methods,
    while the objective can be changed by setting the objective attribute,
    e.g. :code:`model.objective = Objective(expr, direction="max")`.

    Once the problem has been formulated the optimization can be performed by calling the :code:`.optimize` method.
    This will return the status of the optimization, most commonly 'optimal', 'infeasible' or 'unbounded'.

    Attributes
    ----------
    objective: str
        The objective function.
    name: str, optional
        The name of the optimization problem.
    variables: Container, read-only
        Contains the variables of the optimization problem.
        The keys are the variable names and values are the actual variables.
    constraints: Container, read-only
         Contains the variables of the optimization problem.
         The keys are the constraint names and values are the actual constraints.
    status: str, read-only
        The status of the optimization problem.

    Examples
    --------
    >>> model = Model(name="my_model")
    >>> x1 = Variable("x1", lb=0, ub=20)
    >>> x2 = Variable("x2", lb=0, ub=10)
    >>> c1 = Constraint(2 * x1 - x2, lb=0, ub=0) # Equality constraint
    >>> model.add([x1, x2, c1])
    >>> model.objective = Objective(x1 + x2, direction="max")
    >>> model.optimize()
    'optimal'
    >>> x1.primal, x2.primal
    '(5.0, 10.0)'

    """

    @classmethod
    def clone(cls, model, use_json=True, use_lp=False):
        """
        Make a copy of a model. The model being copied can be of the same type or belong to
        a different solver interface. This is the preferred way of copying models.

        Example
        ----------
        >>> new_model = Model.clone(old_model)
        """
        model.update()
        interface = sys.modules[cls.__module__]

        if use_lp:
            warnings.warn("Cloning with LP formats can change variable and constraint ID's.")
            new_model = cls.from_lp(model.to_lp())
            new_model.configuration = interface.Configuration.clone(model.configuration, problem=new_model)
            return new_model

        if use_json:
            new_model = cls.from_json(model.to_json())
            new_model.configuration = interface.Configuration.clone(model.configuration, problem=new_model)
            return new_model

        new_model = cls()
        for variable in model.variables:
            new_variable = interface.Variable.clone(variable)
            new_model._add_variable(new_variable)
        for constraint in model.constraints:
            new_constraint = interface.Constraint.clone(constraint, model=new_model)
            new_model._add_constraint(new_constraint)
        if model.objective is not None:
            new_model.objective = interface.Objective.clone(model.objective, model=new_model)
        new_model.configuration = interface.Configuration.clone(model.configuration, problem=new_model)
        return new_model

    def __init__(self, name=None, objective=None, variables=None, constraints=None, problem=None, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)

        self._objective = self.interface.Objective(0)
        self._objective.problem = self
        self._variables = Container()
        self._constraints = Container()
        self._variables_to_constraints_mapping = dict()
        self._status = None
        self.name = name

        class Modifications():

            def __init__(self):
                self.add_var = []
                self.add_constr = []
                self.add_constr_sloppy = []
                self.rm_var = []
                self.rm_constr = []
                self.var_lb = []
                self.var_ub = []
                self.toggle = 'add'

            def __str__(self):
                return str(self.__dict__)

        self._pending_modifications = Modifications()

        if problem is not None:
            if not (objective is None and variables is None and constraints is None):
                raise ValueError("Models constructor must be called with the problem argument OR with variables, constraints and objective")
            self._initialize_model_from_problem(problem)

        else:
            self._initialize_problem()
            if variables is not None:
                self.add(variables)
            if constraints is not None:
                self.add(constraints)
            if objective is not None:
                self.objective = objective
        self._initialize_configuration()

    def _initialize_problem(self):
        """
        Constructs an empty solver-specific problem. Is called during construction of new Model objects when problem=None.
        Should be implemented in all solver interfaces.
        """
        pass

    def _initialize_model_from_problem(self, problem):
        """
        Parses a solver-specific problem object and initializes the Model object to a state that is consistent with the problem.
        The problem is set as the Model's corresponding problem.
        Should be implemented in all solver interfaces.
        """
        raise NotImplementedError

    def _initialize_configuration(self):
        """
        Initializes a Configuration object. Should be implemented in solver interfaces if non-standard behaviour is needed
        """
        self.configuration = self.interface.Configuration(problem=self)

    @property
    def interface(self):
        """Provides access to the solver interface the model belongs to

        Returns a Python module, for example optlang.glpk_interface
        """
        return sys.modules[self.__module__]

    @property
    def objective(self):
        """The model's objective function."""
        return self._objective

    @objective.setter
    def objective(self, value):
        self.update()
        for atom in sorted(value.expression.atoms(Variable), key=lambda v: v.name):
            if isinstance(atom, Variable) and (atom.problem is None or atom.problem != self):
                self._pending_modifications.add_var.append(atom)
        self.update()
        if self._objective is not None:
            self._objective.problem = None
        self._objective = value
        self._objective.problem = self

    @property
    def variables(self):
        """The model variables."""
        self.update()
        return self._variables

    @property
    def constraints(self):
        """The model constraints."""
        self.update()
        return self._constraints

    @property
    def status(self):
        """The solver status of the model."""
        return self._status

    def _get_variables_names(self):
        """The names of model variables.

        Returns
        -------
        list
        """
        return [variable.name for variable in self.variables]

    @property
    def primal_values(self):
        """The primal values of model variables.

        The primal values are rounded to the bounds.
        Returns
        -------
        collections.OrderedDict
        """
        return OrderedDict(
            zip(self._get_variables_names(), self._get_primal_values())
        )

    def _get_primal_values(self):
        """The primal values of model variables.

        Returns
        -------
        list
        """
        # Fallback, if nothing faster is available
        return [variable.primal for variable in self.variables]

    @property
    def reduced_costs(self):
        """The reduced costs/dual values of all variables.

        Returns
        -------
        collections.OrderedDict
        """
        return OrderedDict(
            zip(self._get_variables_names(), self._get_reduced_costs())
        )

    def _get_reduced_costs(self):
        """The reduced costs/dual values of all variables.

        Returns
        -------
        list
        """
        # Fallback, if nothing faster is available
        return [variable.dual for variable in self.variables]

    def _get_constraint_names(self):
        """The names of model constraints.

        Returns
        -------
        list
        """
        return [constraint.name for constraint in self.constraints]

    @property
    def constraint_values(self):
        """The primal values of all constraints.

        Returns
        -------
        collections.OrderedDict
        """
        return OrderedDict(
            zip(self._get_constraint_names(), self._get_constraint_values())
        )

    def _get_constraint_values(self):
        """The primal values of all constraints.

        Returns
        -------
        list
        """
        # Fallback, if nothing faster is available
        return [constraint.primal for constraint in self.constraints]

    @property
    def shadow_prices(self):
        """The shadow prices of model (dual values of all constraints).

        Returns
        -------
        collections.OrderedDict
        """
        return OrderedDict(
            zip(self._get_constraint_names(), self._get_shadow_prices())
        )

    def _get_shadow_prices(self):
        """The shadow prices of model (dual values of all constraints).

        Returns
        -------
        collections.OrderedDict
        """
        # Fallback, if nothing faster is available
        return [constraint.dual for constraint in self.constraints]

    @property
    def is_integer(self):
        return any(var.type in ("integer", "binary") for var in self.variables)

    def __str__(self):  # pragma: no cover
        if hasattr(self, "to_lp"):
            return self.to_lp()
        self.update()
        return '\n'.join((
            str(self.objective),
            "subject to",
            '\n'.join([str(constr) for constr in self.constraints]),
            'Bounds',
            '\n'.join([str(var) for var in self.variables])
        ))

    def add(self, stuff, sloppy=False):
        """Add variables and constraints.

        Parameters
        ----------
        stuff : iterable, Variable, Constraint
            Either an iterable containing variables and constraints or a single variable or constraint.

        sloppy : bool
            Check constraints for variables that are not part of the model yet.

        Returns
        -------
        None
        """
        if self._pending_modifications.toggle == 'remove':
            self.update()
            self._pending_modifications.toggle = 'add'
        if isinstance(stuff, Iterable):
            for elem in stuff:
                self.add(elem, sloppy=sloppy)
        elif isinstance(stuff, Variable):
            if stuff.__module__ != self.__module__:
                raise TypeError("Cannot add Variable %s of interface type %s to model of type %s." % (
                    stuff, stuff.__module__, self.__module__))
            self._pending_modifications.add_var.append(stuff)
        elif isinstance(stuff, Constraint):
            if stuff.__module__ != self.__module__:
                raise TypeError("Cannot add Constraint %s of interface type %s to model of type %s." % (
                    stuff, stuff.__module__, self.__module__))
            if sloppy is True:
                self._pending_modifications.add_constr_sloppy.append(stuff)
            else:
                self._pending_modifications.add_constr.append(stuff)
        else:
            raise TypeError("Cannot add %s. It is neither a Variable, or Constraint." % stuff)

    def remove(self, stuff):
        """Remove variables and constraints.

        Parameters
        ----------
        stuff : iterable, str, Variable, Constraint
            Either an iterable containing variables and constraints to be removed from the model or a single variable or contstraint (or their names).

        Returns
        -------
        None
        """
        if self._pending_modifications.toggle == 'add':
            self.update()
            self._pending_modifications.toggle = 'remove'
        if isinstance(stuff, str):
            try:
                variable = self.variables[stuff]
                self._pending_modifications.rm_var.append(variable)
            except KeyError:
                try:
                    constraint = self.constraints[stuff]
                    self._pending_modifications.rm_constr.append(constraint)
                except KeyError:
                    raise LookupError(
                        "%s is neither a variable nor a constraint in the current solver instance." % stuff)
        elif isinstance(stuff, Variable):
            self._pending_modifications.rm_var.append(stuff)
        elif isinstance(stuff, Constraint):
            self._pending_modifications.rm_constr.append(stuff)
        elif isinstance(stuff, Iterable):
            for elem in stuff:
                self.remove(elem)
        elif isinstance(stuff, Objective):
            raise TypeError(
                "Cannot remove objective %s. Use model.objective = Objective(...) to change the current objective." % stuff)
        else:
            raise TypeError(
                "Cannot remove %s. It neither a variable or constraint." % stuff)

    def update(self, callback=int):
        """Process all pending model modifications."""
        # print(self._pending_modifications)
        add_var = self._pending_modifications.add_var
        if len(add_var) > 0:
            self._add_variables(add_var)
            self._pending_modifications.add_var = []
        callback()

        add_constr = self._pending_modifications.add_constr
        if len(add_constr) > 0:
            self._add_constraints(add_constr)
            self._pending_modifications.add_constr = []

        add_constr_sloppy = self._pending_modifications.add_constr_sloppy
        if len(add_constr_sloppy) > 0:
            self._add_constraints(add_constr_sloppy, sloppy=True)
            self._pending_modifications.add_constr_sloppy = []

        var_lb = self._pending_modifications.var_lb
        var_ub = self._pending_modifications.var_ub
        if len(var_lb) > 0 or len(var_ub) > 0:
            self._set_variable_bounds_on_problem(var_lb, var_ub)
            self._pending_modifications.var_lb = []
            self._pending_modifications.var_ub = []

        rm_var = self._pending_modifications.rm_var
        if len(rm_var) > 0:
            self._remove_variables(rm_var)
            self._pending_modifications.rm_var = []
        callback()

        rm_constr = self._pending_modifications.rm_constr
        if len(rm_constr) > 0:
            self._remove_constraints(rm_constr)
            self._pending_modifications.rm_constr = []

    def optimize(self):
        """
        Solve the optimization problem using the relevant solver back-end.
        The status returned by this method tells whether an optimal solution was found,
        if the problem is infeasible etc. Consult optlang.statuses for more elaborate explanations
        of each status.

        The objective value can be accessed from 'model.objective.value', while the solution can be
        retrieved by 'model.primal_values'.

        Returns
        -------
        status: str
            Solution status.
        """
        self.update()
        status = self._optimize()
        if status != OPTIMAL and self.configuration.presolve == "auto":
            self.configuration.presolve = True
            status = self._optimize()
            self.configuration.presolve = "auto"
        self._status = status
        return status

    def _optimize(self):
        raise NotImplementedError(
            "You're using the high level interface to optlang. Problems cannot be optimized in this mode. Choose from one of the solver specific interfaces.")

    def _set_variable_bounds_on_problem(self, var_lb, var_ub):
        """"""
        pass

    def _add_variable(self, variable):
        self._add_variables([variable])

    def _add_variables(self, variables):
        for variable in variables:
            self._variables.append(variable)
            self._variables_to_constraints_mapping[variable.name] = set([])
            variable.problem = self

    def _remove_variables(self, variables):
        for variable in variables:
            try:
                self._variables[variable.name]
            except KeyError:
                raise LookupError("Variable %s not in solver" % variable.name)

        constraint_ids = set()
        for variable in variables:
            constraint_ids.update(self._variables_to_constraints_mapping[variable.name])
            del self._variables_to_constraints_mapping[variable.name]
            variable.problem = None
            del self._variables[variable.name]

        replacements = dict([(variable, 0) for variable in variables])
        for constraint_id in constraint_ids:
            constraint = self._constraints[constraint_id]
            constraint._expression = constraint._expression.xreplace(replacements)
        if self.objective is not None:
            self.objective._expression = self.objective._expression.xreplace(replacements)

    def _remove_variable(self, variable):
        self._remove_variables([variable])

    def _add_constraint(self, constraint, sloppy=False):
        self._add_constraints([constraint], sloppy=sloppy)

    def _add_constraints(self, constraints, sloppy=False):
        for constraint in constraints:
            constraint_id = constraint.name
            if sloppy is False:
                variables = constraint.variables
                if constraint.indicator_variable is not None:
                    variables.add(constraint.indicator_variable)
                missing_vars = [var for var in variables if var.problem is not self]
                if len(missing_vars) > 0:
                    self._add_variables(missing_vars)
                for var in variables:
                    try:
                        self._variables_to_constraints_mapping[var.name].add(constraint_id)
                    except KeyError:
                        self._variables_to_constraints_mapping[var.name] = set([constraint_id])
            self._constraints.append(constraint)
            constraint._problem = self

    def _remove_constraints(self, constraints):
        keys = [constraint.name for constraint in constraints]
        if len(constraints) > 350:  # Need to figure out a good threshold here
            self._constraints = self._constraints.fromkeys(set(self._constraints.keys()).difference(set(keys)))
            for co in constraints:
                co.problem = None
        else:
            for constraint in constraints:
                try:
                    del self._constraints[constraint.name]
                except KeyError:
                    raise LookupError("Constraint %s not in solver" % constraint)
                else:
                    constraint.problem = None

    def _remove_constraint(self, constraint):
        self._remove_constraints([constraint])

    def to_json(self):
        """
        Returns a json-compatible object from the model that can be saved using the json module.
        Variables, constraints and objective contained in the model will be saved. Configurations
        will not be saved.

        Example
        --------
        >>> import json
        >>> with open("path_to_file.json", "w") as outfile:
        >>>     json.dump(model.to_json(), outfile)
        """
        json_obj = {
            "name": self.name,
            "variables": [var.to_json() for var in self.variables],
            "constraints": [const.to_json() for const in self.constraints],
            "objective": self.objective.to_json()
        }
        return json_obj

    @classmethod
    def from_json(cls, json_obj):
        """
        Constructs a Model from the provided json-object.

        Example
        --------
        >>> import json
        >>> with open("path_to_file.json") as infile:
        >>>     model = Model.from_json(json.load(infile))
        """
        model = cls()
        model._init_from_json(json_obj)
        return model

    def _init_from_json(self, json_obj):
        self.name = json_obj["name"]
        interface = self.interface
        variables = [interface.Variable.from_json(var_json) for var_json in json_obj["variables"]]
        var_dict = {var.name: var for var in variables}
        constraints = [interface.Constraint.from_json(constraint_json, var_dict) for constraint_json in json_obj["constraints"]]
        objective = interface.Objective.from_json(json_obj["objective"], var_dict)
        self.add(variables)
        self.add(constraints)
        self.objective = objective
        self.update()

    def __getstate__(self):
        state = self.to_json()
        try:
            state["configuration"] = self.interface.Configuration._get_parameters(self.configuration)
        except NotImplementedError:  # solver-agnostic interface
            pass
        return state

    def __setstate__(self, state):
        self.__init__()
        self._init_from_json(state)
        if "configuration" in state:  # absent when pickled by optlang <= 1.9.1
            self.configuration = self.interface.Configuration(problem=self, **state["configuration"])


if __name__ == '__main__':
    # Example workflow

    x1 = Variable('x1', lb=0)
    x2 = Variable('x2', lb=0)
    x3 = Variable('x3', lb=0)
    c1 = Constraint(x1 + x2 + x3, ub=100)
    c2 = Constraint(10 * x1 + 4 * x2 + 5 * x3, ub=600)
    c3 = Constraint(2 * x1 + 2 * x2 + 6 * x3, ub=300)
    obj = Objective(10 * x1 + 6 * x2 + 4 * x3, direction='max')
    model = Model(name='Simple model')
    model.objective = obj
    model.add([c1, c2, c3])

    try:
        sol = model.optimize()
    except NotImplementedError as e:
        print(e)

    print(model)
    print(model.variables)

    # model.remove(x1)

    model.interface = optlang.glpk_interface
