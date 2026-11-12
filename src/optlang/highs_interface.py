"""
optlang interface for the HiGHS solver, using the highspy Python bindings.

Supports LP, QP (Quadratic Programming) and MILP (Mixed Integer Linear
Programming).
- Continuous, integer and binary variables. Note: mixing integer/binary
  variables with a quadratic objective (MIQP) is not supported by HiGHS;
  use a linear objective whenever the model contains integer/binary
  variables.
- Linear constraints.
- Linear or Quadratic objectives (continuous models only, see above).

Install with:
    pip install highspy
"""

import logging
from collections import defaultdict

import numpy as np

try:
    import highspy
except ImportError:
    raise ImportError("The highs_interface requires highspy: pip install highspy")

from optlang import interface
from optlang import symbolics
from optlang.exceptions import ContainerAlreadyContains

log = logging.getLogger(__name__)


# Keys are the strings returned by Highs.modelStatusToString() (verified
# against highspy 1.15.1). Mirrors hybrid_interface._STATUS_MAP for the shared
# statuses; where the two differ, the more specific optlang status is used here.
_HIGHS_STATUS_TO_STATUS = {
    "Optimal": interface.OPTIMAL,
    "Infeasible": interface.INFEASIBLE,
    "Unbounded": interface.UNBOUNDED,
    "Primal infeasible or unbounded": interface.INFEASIBLE_OR_UNBOUNDED,
    "Time limit reached": interface.TIME_LIMIT,
    "Iteration limit reached": interface.ITERATION_LIMIT,
    "Memory limit reached": interface.MEMORY_LIMIT,
    # MIP-specific early-termination statuses (only ever returned once a
    # model has integer/binary variables and HiGHS' branch-and-bound runs):
    "Solution limit reached": interface.SOLUTION_LIMIT,
    "Bound on objective reached": interface.SUBOPTIMAL,
    "Target for objective reached": interface.SUBOPTIMAL,
    "Interrupted by user": interface.ABORTED,
    "Interrupted by HiGHS": interface.ABORTED,
    "Load error": interface.SPECIAL,
    "Model error": interface.SPECIAL,
    "Presolve error": interface.SPECIAL,
    "Solve error": interface.SPECIAL,
    "Postsolve error": interface.SPECIAL,
    "Empty": interface.UNDEFINED,
    "Not Set": interface.UNDEFINED,
    "Unknown": interface.UNDEFINED,
}

_STATUSES_WITH_USABLE_SOLUTION = frozenset([
    interface.OPTIMAL, interface.SUBOPTIMAL, interface.TIME_LIMIT, interface.ITERATION_LIMIT,
    interface.INFEASIBLE # needed for cobrapy, see cobra/util/solver.has_primals
])


def _split_terms(expression, _expanded=False):
    """Split a linear/quadratic sympy/symengine expression in a single pass.

    Returns ``(linear, quadratic, constant)`` where

    * ``linear`` is ``{var_name: coeff}``,
    * ``quadratic`` is ``{(name1, name2): coeff}`` with the names sorted (a
      square ``c*x**2`` is stored as ``(x, x): c``),
    * ``constant`` is the constant offset.

    The fast path handles the shapes that show up in practice: numbers,
    ``x``, ``c*x``, ``c*x*y`` and ``c*x**2`` (as terms of an ``Add`` or on
    their own). If a term has any other shape (e.g. an unexpanded product) the
    expression is expanded once and parsed again; if that still fails, or the
    degree is above 2, a ``ValueError`` is raised. That makes this function
    double as the linear/quadratic validity check.
    """
    terms = expression.args if getattr(expression, "is_Add", False) else (expression,)
    linear = defaultdict(float)
    quadratic = defaultdict(float)
    constant = 0.0

    for term in terms:
        if term.is_Number:
            constant += float(term)
            continue

        coeff = 1.0
        names = []
        for factor in (term.args if term.is_Mul else (term,)):
            if factor.is_Number:
                coeff *= float(factor)
            elif factor.is_Symbol:
                names.append(factor.name)
            elif (factor.is_Pow and factor.args[0].is_Symbol
                  and factor.args[1].is_Number and float(factor.args[1]) == 2.0):
                names.extend((factor.args[0].name,) * 2)
            else:
                break
        else:  # every factor was recognised
            n = len(names)
            if n == 0:
                constant += coeff
            elif n == 1:
                linear[names[0]] += coeff
            elif n == 2:
                quadratic[tuple(sorted(names))] += coeff
            else:
                raise ValueError("Degree > 2 detected: %s" % term)
            continue

        # Unrecognised factor: expand once and retry, otherwise give up.
        if _expanded:
            raise ValueError("Unsupported term in expression: %s" % term)
        return _split_terms(expression.expand(), _expanded=True)

    return dict(linear), dict(quadratic), constant


def _linear_expression_to_dict(expression):
    """Turn a linear expression into ({var_name: coeff}, constant)."""
    linear, quadratic, constant = _split_terms(expression)
    if quadratic:
        raise ValueError("Expression is not linear.")
    return linear, constant


def _separate_linear_and_quadratic_from_expr(expression):
    """Return (linear_coeffs, quadratic_coeffs, constant) of an expression."""
    return _split_terms(expression)


class Variable(interface.Variable):
    def __init__(self, name, *args, **kwargs):
        var_type = kwargs.get("type", "continuous")
        if var_type not in ("continuous", "integer", "binary"):
            raise ValueError(
                "This HiGHS interface only supports continuous, integer "
                "and binary variables (type=%r is not one of these)." % (var_type,)
            )
        super(Variable, self).__init__(name, **kwargs)
        # The column index of this variable in the HiGHS problem it currently
        # belongs to (None if it isn't attached to a model). New columns are
        # always appended at the end, so Model._add_variables can just read
        # off problem.getNumCol() - 1 to set this; on removal, the remaining
        # variables' indices are shifted down to match HiGHS's own reindexing
        # (see Model._remove_variables). This makes a separate
        # name -> index map on the Model unnecessary.
        self._solver_index = None

    @interface.Variable.lb.setter
    def lb(self, value):
        interface.Variable.lb.fset(self, value)
        if self.problem is not None:
            self.problem._highs_set_col_bounds(self)

    @interface.Variable.ub.setter
    def ub(self, value):
        interface.Variable.ub.fset(self, value)
        if self.problem is not None:
            self.problem._highs_set_col_bounds(self)

    @interface.Variable.name.setter
    def name(self, value):
        # Reimplemented rather than delegating to interface.Variable.name's
        # base setter (interface.Variable.name.fset): that base setter also
        # updates problem._variables_to_constraints_mapping, which this
        # interface no longer maintains -- HiGHS's own row/column data is the
        # sole source of truth for constraint/variable membership (see
        # Model._add_variables/_add_constraints/_remove_variables below).
        #
        # getattr(self, 'problem', None), not self.problem: sympy's
        # Symbol.__new__ (via __xnew__) assigns .name during construction,
        # before Variable.__init__ has set self.problem at all - a direct
        # attribute access would raise AttributeError at that point.
        if len(value) < 1:
            raise ValueError('Variable name must not be empty string')
        for char in value:
            if char.isspace():
                raise ValueError(
                    'Variable names cannot contain whitespace characters. "%s" contains whitespace character "%s".' % (
                        value, char))
        old_name = getattr(self, 'name', None)
        self._name = value
        problem = getattr(self, 'problem', None)
        if problem is not None and value != old_name:
            problem.variables.update_key(old_name)
        if problem is not None:
            problem._highs_set_col_name(self)

    @interface.Variable.type.setter
    def type(self, value):
        # The base setter (interface.Variable.type.fset) may itself go
        # through self.lb/self.ub (for type == 'integer', via the property
        # setters above - which already sync bounds to HiGHS on their own)
        # or bypass them entirely and poke self._lb/self._ub directly (for
        # type == 'binary'). Either way, once the base setter returns,
        # self.lb/self.ub/self.type all already reflect the final, settled
        # state, so _highs_set_col_type below can simply push that settled
        # state (bounds + integrality) to HiGHS in one go - no further
        # self.lb=/self.ub=/self.type= assignments happen here, so there is
        # no risk of this setter (transitively) re-entering itself.
        interface.Variable.type.fset(self, value)
        if self.problem is not None:
            self.problem._highs_set_col_type(self)

    def set_bounds(self, lb, ub):
        super(Variable, self).set_bounds(lb, ub)
        if self.problem is not None:
            self.problem._highs_set_col_bounds(self)

    @property
    def primal(self):
        if self.problem is None:
            return None
        return self.problem._variable_primal(self)

    @property
    def dual(self):
        if self.problem is None:
            return None
        return self.problem._variable_dual(self)


class Constraint(interface.Constraint):
    """
    Unlike the earlier version of this interface, this Constraint does not keep
    its own copy of the linear coefficients around. Once the constraint is part
    of a model, HiGHS's own row data is the single source of truth - writes go
    straight through to the solver (mirroring the GLPK interface's design), so
    there's no local coefficient cache that can silently drift from it.

    .expression is the one exception: rebuilding it from HiGHS's row data on
    every access is expensive, so it's cached in `_expression` and only
    rebuilt when `_expression_expired` is set. Anything that changes what a
    constraint's row actually contains - a direct coefficient write, or a
    referenced variable disappearing out from under it - is responsible for
    setting that flag.

    `_initial_coeffs` is only a bootstrap value: it is parsed once at
    construction time so `_add_constraints` has something to build the HiGHS
    row from, and is never consulted again once that row exists.

    `_constant` remains local state (HiGHS rows don't carry a constant offset
    themselves, so it has to be folded into the row bounds), but a bare
    constant can never reference a variable, so it can never go stale the way
    a coefficient cache can.
    """

    _INDICATOR_CONSTRAINT_SUPPORT = False

    def __init__(self, expression, sloppy=False, *args, **kwargs):
        # A bare number (e.g. the `Constraint(0, ...)` bootstrap used to set
        # up a constraint that will be filled in afterwards via
        # set_linear_coefficients({var: coeff, ...})) needs no symbolic
        # parsing at all: capture it as a plain float and skip straight to
        # the trivial coefficients below.
        _constant = float(expression) if isinstance(expression, (int, float)) else None
        if _constant is not None:
            sloppy = True
            expression = symbolics.Real(expression)

        self._expression_expired = False
        super(Constraint, self).__init__(expression, sloppy=sloppy, *args, **kwargs)

        if _constant is not None:
            self._initial_coeffs, self._constant = {}, _constant
        else:
            # Parsing doubles as the linearity check (raises ValueError for
            # anything non-linear), so no separate is_Linear pass is needed.
            try:
                self._initial_coeffs, self._constant = _linear_expression_to_dict(self.expression)
            except ValueError as err:
                raise ValueError(
                    "The HiGHS interface only supports linear constraints. "
                    "%s is not linear." % self
                ) from err
        # The row index of this constraint in the HiGHS problem it currently
        # belongs to (None if it isn't attached to a model, or not yet added
        # to the model's HiGHS instance). See Variable._solver_index for why
        # this replaces a name -> index map kept on the Model.
        self._solver_index = None

    @interface.Constraint.lb.setter
    def lb(self, value):
        interface.Constraint.lb.fset(self, value)
        if self.problem is not None:
            self.problem._highs_set_row_bounds(self)

    @interface.Constraint.ub.setter
    def ub(self, value):
        interface.Constraint.ub.fset(self, value)
        if self.problem is not None:
            self.problem._highs_set_row_bounds(self)

    @interface.Constraint.name.setter
    def name(self, value):
        interface.Constraint.name.fset(self, value)
        if getattr(self, 'problem', None) is not None:
            self.problem._highs_set_row_name(self)

    @property
    def primal(self):
        if self.problem is None:
            return None
        return self.problem._constraint_primal(self)

    @property
    def dual(self):
        if self.problem is None:
            return None
        return self.problem._constraint_dual(self)

    def _get_expression(self):
        """Reconstruct the expression by reading HiGHS's live row data if necessary."""
        if not self._expression_expired:
            return self._expression
        elif self.problem is not None: # constraint may be pending removal
            self.problem.update()

        if self.problem is None:
            raise ValueError("Constraint expression cannot be constructed.")

        row_index = self._solver_index
        if row_index is None:
            # Constructed, but not yet added to the model's HiGHS instance -
            # report the bootstrap coefficients that _add_constraints will use.
            terms = []
            for var_name, coeff in self._initial_coeffs.items():
                var = self.problem.variables.get(var_name)
                if var is not None:
                    terms.append(coeff * var)
        else:
            _, idx, val = self.problem.problem.getRowEntries(row_index)
            terms = []
            for i, v in zip(idx, val):
                var_name = self.problem.problem.variableName(int(i))
                var = self.problem.variables.get(var_name)
                if var is not None:
                    terms.append(float(v) * var)

        if self._constant != 0:
            # symbolics.add() uses SymPy's low-level Add._from_args(), which
            # (unlike Add(*args)) does not auto-sympify its arguments - a bare
            # Python float here breaks it (AttributeError: 'float' object has
            # no attribute 'is_commutative'), so sympify explicitly.
            terms.append(symbolics.Real(self._constant))

        self._expression =  symbolics.add(terms) if terms else symbolics.sympify(0)
        self._expression_expired = False
        return self._expression

    def set_linear_coefficients(self, coefficients, sloppy=False):
        """
        Sets the linear coefficients of the constraint directly, bypassing SymPy
        parsing. Writes go straight through to HiGHS's row - there's no local
        cache to keep in sync, so this can never drift from the solver's state.

        Args:
            coefficients (dict): A dictionary of {var: coeff}.
            sloppy (bool): If True, skip re-resolving each variable through
                the model (self.problem.variables.get(var.name)) and pass
                `coefficients` straight through to HiGHS. Only safe when the
                variables in `coefficients` are already known to belong to
                this constraint's model (e.g. they came from model.variables
                in the first place) - saves a dict lookup per variable, which
                matters when this is called in a hot loop.
        """
        if self.problem is None:
            raise Exception("Can't change coefficients if constraint is not associated with a model.")

        if self._solver_index is None:
            # Not yet added to the model - update the bootstrap coefficients
            # that will be used to build the row.
            for var, coeff in coefficients.items():
                self._initial_coeffs[var.name] = coeff
            return

        if sloppy:
            self.problem._highs_set_coefficients(self, coefficients)
            self._expression_expired = True
            return

        model_coeffs = {}
        for var, coeff in coefficients.items():
            var_obj = self.problem.variables.get(var.name)
            if var_obj is not None:
                model_coeffs[var_obj] = coeff

        self.problem._highs_set_coefficients(self, model_coeffs)
        self._expression_expired = True

    def get_linear_coefficients(self, variables):
        """
        Get coefficients of linear terms in constraint.

        Like everything else on this class, this reads straight from HiGHS's
        row data when the constraint is attached to a model (via the same
        helper _get_expression uses), rather than from any local cache.

        Parameters
        ----------
        variables : iterable
            An iterable of Variable objects

        Returns
        -------
        Coefficients : dict
            {var1: coefficient, var2: coefficient ...}. Variables with no
            entry in the constraint get a coefficient of 0.
        """
        if self.problem is None:
            raise Exception(
                "Can't get coefficients from solver if constraint is not associated with a model.")
        coeffs, _, _ = self.problem._constraint_to_coeffs(self)
        return {variable: coeffs.get(variable.name, 0) for variable in variables}


class Objective(interface.Objective):
    """
    The linear part of the objective follows the same ground-truth design as
    Constraint: once attached to a model, HiGHS's own cost vector (readable via
    Highs.getObjective()) is authoritative, and coefficient writes go straight
    through to Highs.changeColCost. There's no local linear-coefficient cache
    left to go stale.

    The quadratic part (_quadratic_coeffs) is still cached locally. highspy's
    high-level wrapper (highs.py) has no Hessian read-back equivalent to
    getObjective()/getRowEntries() - passHessian is write-only from what's
    exposed there. Until that's confirmed one way or another against the
    compiled _core module, the Hessian remains the one piece still built from
    a local cache rather than queried live, and _remove_variables still has to
    prune it explicitly when a variable disappears.
    """

    def __init__(self, expression, sloppy=False, *args, **kwargs):
        # Same reasoning as Constraint.__init__: a bare number (typically
        # `Objective(0, ...)` followed by
        # set_coefficients(linear={var: coeff, ...})) needs no symbolic
        # parsing - capture it as a plain float up front.
        _constant = float(expression) if isinstance(expression, (int, float)) else None
        if _constant is not None:
            expression = symbolics.Real(expression)
            sloppy = True

        kwargs['sloppy'] = sloppy
        super(Objective, self).__init__(expression, *args, **kwargs)

        # Bootstrap values only: _initial_linear_coeffs seeds HiGHS's cost
        # vector the first time this Objective is attached to a model (see
        # Model._update_linear_objective) and is never consulted again after
        # that. _quadratic_coeffs remains the ongoing source of truth for the
        # Hessian (see class docstring). Parsing doubles as the
        # linear/quadratic validity check (raises ValueError otherwise).
        if _constant is not None:
            self._initial_linear_coeffs, self._quadratic_coeffs, self._constant = {}, {}, _constant
        else:
            try:
                self._initial_linear_coeffs, self._quadratic_coeffs, self._constant = \
                    _separate_linear_and_quadratic_from_expr(self.expression)
            except ValueError as err:
                raise ValueError(
                    "The HiGHS interface only supports linear or quadratic objectives."
                ) from err

    def _get_expression(self):
        """Reconstruct the expression, reading the linear part live from HiGHS."""
        if self.problem is None:
            return self._expression

        obj_expr, _ = self.problem.problem.getObjective()
        linear_coeffs = {}
        for i, v in zip(obj_expr.idxs, obj_expr.vals):
            var_name = self.problem.problem.variableName(int(i))
            linear_coeffs[var_name] = float(v)

        terms = []

        for (v1_name, v2_name), coeff in self._quadratic_coeffs.items():
            v1 = self.problem.variables.get(v1_name)
            v2 = self.problem.variables.get(v2_name)
            if v1 is not None and v2 is not None:
                terms.append(coeff * v1 * v2)

        for var_name, coeff in linear_coeffs.items():
            var = self.problem.variables.get(var_name)
            if var is not None:
                terms.append(coeff * var)

        if self._constant != 0:
            # See Constraint._get_expression - symbolics.add() needs an
            # already-sympified argument, not a bare Python float.
            terms.append(symbolics.sympify(self._constant))

        return symbolics.add(terms) if terms else symbolics.sympify(0)

    @interface.Objective.expression.getter
    def expression(self):
        return self._get_expression()

    @interface.Objective.direction.setter
    def direction(self, value):
        interface.Objective.direction.fset(self, value)
        if self.problem is not None:
            self.problem._highs_set_objective_sense(value)

    @property
    def value(self):
        if self.problem is None:
            return None
        return self.problem._get_objective_value()

    def set_linear_coefficients(self, coefficients):
        self.set_coefficients(linear=coefficients)

    def set_coefficients(self, linear=None, quadratic=None, constant=None):
        """
        Sets the objective coefficients directly, bypassing SymPy parsing.

        Args:
            linear (dict, optional): Dictionary of {Variable: coeff} for linear terms.
            quadratic (dict, optional): Dictionary of {(Variable, Variable): coeff}
                                       for quadratic terms.
            constant (float, optional): Constant term of the objective.
        """
        if self.problem is None:
            raise Exception("Can't change coefficients if objective is not associated with a model.")

        # Linear coefficients go straight through to HiGHS's cost vector -
        # only the named variables are touched, so this is naturally a merge,
        # not a replace, with no local bookkeeping required. (This matters in
        # practice: callers like cobrapy build up an objective by calling this
        # incrementally, one or two variables at a time, across many calls,
        # and expect earlier calls' coefficients to survive.)
        if linear is not None:
            self.problem._highs_set_objective_coefficients(linear)

        # Quadratic coefficients are different: _quadratic_coeffs is still a
        # local cache standing in for the whole Hessian (see class
        # docstring), and passing quadratic={} is the only way to express
        # "there are now fewer quadratic terms than before" - including zero
        # of them, e.g. when moving from a QP back to a plain LP. So unlike
        # linear, an explicitly-given quadratic dict *replaces* the cache
        # outright rather than merging into it.
        if quadratic is not None:
            self._quadratic_coeffs = {
                (v1.name, v2.name): coeff for (v1, v2), coeff in quadratic.items()
            }
            self.problem._update_quadratic_objective()

        if constant is not None:
            self._constant = float(constant)
            self.problem.problem.changeObjectiveOffset(float(constant))

    def get_linear_coefficients(self, variables):
        """
        Get coefficients of linear terms in objective.

        Reads the linear part straight from HiGHS's live cost vector when
        attached to a model, the same way _get_expression does. The
        quadratic part is intentionally not reflected here since this only
        reports linear coefficients (see set_linear_coefficients/
        set_coefficients for the quadratic equivalent).

        Parameters
        ----------
        variables : iterable
            An iterable of Variable objects

        Returns
        -------
        Coefficients : dict
            {var1: coefficient, var2: coefficient ...}. Variables with no
            linear entry in the objective get a coefficient of 0.
        """
        if self.problem is None:
            raise Exception(
                "Can't get coefficients from solver if objective is not associated with a model.")
        obj_expr, _ = self.problem.problem.getObjective()
        linear_coeffs = {
            self.problem.problem.variableName(int(i)): float(v)
            for i, v in zip(obj_expr.idxs, obj_expr.vals)
        }
        return {variable: linear_coeffs.get(variable.name, 0) for variable in variables}


class Tolerances(object):
    def __init__(self, configuration):
        self._configuration = configuration
        self._feasibility = 1e-7
        self._optimality = 1e-7
        # HiGHS always solves QPs with its interior-point method, which has its
        # own convergence tolerance distinct from the simplex-oriented
        # primal/dual feasibility tolerances above. Exposed separately since
        # loosening it is often the fix for QPs that fail to converge for
        # purely numerical/scaling reasons.
        self._ipm_optimality = 1e-8
        # Relative MIP optimality gap (HiGHS option 'mip_rel_gap'). Only
        # meaningful once a model has integer/binary variables and its
        # branch-and-bound actually runs; applying it unconditionally to
        # every model (see _apply_tolerances) is harmless for pure LP/QP
        # problems, since HiGHS simply ignores it there.
        self._mip_gap = 1e-4

    @property
    def feasibility(self):
        return self._feasibility

    @feasibility.setter
    def feasibility(self, value):
        self._feasibility = value
        self._configuration._apply_tolerances()

    @property
    def optimality(self):
        return self._optimality

    @optimality.setter
    def optimality(self, value):
        self._optimality = value
        self._configuration._apply_tolerances()

    @property
    def ipm_optimality(self):
        return self._ipm_optimality

    @ipm_optimality.setter
    def ipm_optimality(self, value):
        self._ipm_optimality = value
        self._configuration._apply_tolerances()

    @property
    def mip_gap(self):
        return self._mip_gap

    @mip_gap.setter
    def mip_gap(self, value):
        self._mip_gap = value
        self._configuration._apply_tolerances()

    def to_dict(self):
        return {
            "feasibility": self.feasibility,
            "optimality": self.optimality,
            "ipm_optimality": self.ipm_optimality,
            "mip_gap": self.mip_gap,
        }


class Configuration(interface.MathematicalProgrammingConfiguration):
    # Allowed values as in hybrid_interface (_LP_METHODS), plus "hipo".
    lp_methods = ("auto", "simplex", "interior point", "hipo")
    # HiGHS solves QPs with its active-set solver; the only other choice is HiPO.
    qp_methods = ("auto", "active set", "hipo")

    # optlang name -> value of the HiGHS "solver" option
    _LP_METHOD_TO_HIGHS = {
        "auto": "choose", "simplex": "simplex", "interior point": "ipm", "hipo": "hipo",
    }
    _QP_METHOD_TO_HIGHS = {"auto": "choose", "active set": "qpasm", "hipo": "hipo"}

    def __init__(self, verbosity=0, timeout=None, presolve=True,
                 problem=None, lp_method="auto", qp_method="auto", *args, **kwargs):
        self.problem = problem
        self._lp_method = self._check_method(lp_method, self.lp_methods, "lp_method")
        self._qp_method = self._check_method(qp_method, self.qp_methods, "qp_method")
        self._verbosity = verbosity
        self._timeout = timeout
        self._presolve = presolve
        self._tolerances = Tolerances(self)
        for key, val in kwargs.get("tolerances", {}).items():
            if hasattr(self._tolerances, key):
                setattr(self._tolerances, key, val)
        super(Configuration, self).__init__(*args, **kwargs)
        self.problem = problem
        if self.problem is not None:
            self._apply_all()

    def _apply_all(self):
        h = self.problem.problem
        if h is None:
            return
        h.setOptionValue("output_flag", self._verbosity != 0)
        if self._timeout is not None:
            h.setOptionValue("time_limit", float(self._timeout))
        h.setOptionValue("presolve", "on" if self._presolve else "off")
        self._apply_tolerances()
        self._apply_solver_method()

    @staticmethod
    def _check_method(value, allowed, name):
        if value not in allowed:
            raise ValueError(
                "%s must be one of %s (got %r)." % (name, ", ".join(map(repr, allowed)), value))
        return value

    @staticmethod
    def _check_hipo_available(name):
        # HiPO is an optional part of HiGHS; some builds (e.g. the PyPI wheels)
        # reject solver="hipo". Probe on a scratch instance so the error shows
        # up when the option is set, not in the middle of a solve.
        probe = highspy.Highs()
        probe.setOptionValue("output_flag", False)
        if probe.setOptionValue("solver", "hipo") != highspy.HighsStatus.kOk:
            raise ValueError(
                "%s='hipo' requested, but this HiGHS build does not include HiPO." % name)

    def _apply_solver_method(self):
        """Push lp_method or qp_method to HiGHS' single 'solver' option.

        HiGHS has one "solver" option for both LP and QP, so which of the two
        settings is active depends on whether the objective currently has a
        quadratic part. Called on every change of either setting and again
        right before each solve (the objective may have switched type).
        """
        if self.problem is None or self.problem.problem is None:
            return
        if self.problem._last_objective_was_quadratic:
            method, table, name = self._qp_method, self._QP_METHOD_TO_HIGHS, "qp_method"
        else:
            method, table, name = self._lp_method, self._LP_METHOD_TO_HIGHS, "lp_method"
        status = self.problem.problem.setOptionValue("solver", table[method])
        if status != highspy.HighsStatus.kOk:
            raise ValueError("HiGHS rejected %s=%r." % (name, method))

    @property
    def lp_method(self):
        """Algorithm used for LPs: 'auto', 'simplex', 'interior point' or 'hipo'."""
        return self._lp_method

    @lp_method.setter
    def lp_method(self, value):
        self._check_method(value, self.lp_methods, "lp_method")
        if value == "hipo":
            self._check_hipo_available("lp_method")
        self._lp_method = value
        self._apply_solver_method()

    @property
    def qp_method(self):
        """Algorithm used for QPs: 'auto', 'active set' or 'hipo'."""
        return self._qp_method

    @qp_method.setter
    def qp_method(self, value):
        self._check_method(value, self.qp_methods, "qp_method")
        if value == "hipo":
            self._check_hipo_available("qp_method")
        self._qp_method = value
        self._apply_solver_method()

    def _apply_tolerances(self):
        if self.problem is not None and self.problem.problem is not None:
            h = self.problem.problem
            h.setOptionValue("primal_feasibility_tolerance", self._tolerances.feasibility)
            h.setOptionValue("dual_feasibility_tolerance", self._tolerances.feasibility)
            h.setOptionValue("ipm_optimality_tolerance", self._tolerances.ipm_optimality)
            h.setOptionValue("mip_rel_gap", self._tolerances.mip_gap)

    @property
    def tolerances(self):
        return self._tolerances

    @tolerances.setter
    def tolerances(self, value):
        if isinstance(value, Tolerances):
            self._tolerances = value

    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value):
        self._verbosity = value
        if self.problem is not None and self.problem.problem is not None:
            self.problem.problem.setOptionValue("output_flag", value != 0)

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        self._timeout = value
        if self.problem is not None and self.problem.problem is not None:
            if value is not None:
                self.problem.problem.setOptionValue("time_limit", float(value))

    @property
    def presolve(self):
        return self._presolve

    @presolve.setter
    def presolve(self, value):
        self._presolve = value
        if self.problem is not None and self.problem.problem is not None:
            self.problem.problem.setOptionValue("presolve", "on" if value else "off")


class _HighsBackedContainer(object):
    """
    Shared base for the model.variables / model.constraints containers. Their
    ordering and indexing are read directly from the live HiGHS instance -
    the same "ask the solver" principle used throughout this interface for
    Constraint/Objective coefficients - instead of a private _object_list /
    _indices that could drift out of sync with the solver's own column/row
    order (e.g. after HiGHS shifts indices down on _remove_variables's
    deleteVars call).

    A name -> object dict is still kept, since HiGHS itself has no notion of
    the optlang Variable/Constraint wrapper objects, only of names and
    column/row numbers - "how many are there" and "what order are they in"
    come from HiGHS, "which Python object goes with this name" comes from
    this dict.

    This deliberately does not subclass optlang.container.Container. That
    class carries its own private _object_list/_indices bookkeeping (exactly
    what this class exists to avoid), plus API surface this interface never
    touches: __setitem__, fromkeys, and the py2-era iterkeys/itervalues/
    iteritems/has_key aliases. Implementing just the subset actually used by
    optlang.interface.Model and this file - append/extend, __contains__/
    __iter__/__getitem__/__delitem__/__len__, get/keys/values/items, and
    update_key (needed by Variable.name's and Constraint.name's setters when
    an attached object is renamed) - keeps this class's behavior easy to
    verify against what's really exercised.

    Subclasses only need to implement _num() and _name_at(idx).
    """

    def __init__(self, model, iterable=()):
        self._model = model
        self._dict = {}
        for item in iterable:
            self.append(item)

    def _num(self):
        raise NotImplementedError("Subclasses must implement _num().")

    def _name_at(self, idx):
        raise NotImplementedError("Subclasses must implement _name_at().")

    def _live_names(self):
        return [self._name_at(i) for i in range(self._num())]

    def __len__(self):
        return self._num()

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self._dict
        name = getattr(item, "name", None)
        return name is not None and name in self._dict and item is self._dict[name]

    def __iter__(self):
        for name in self._live_names():
            obj = self._dict.get(name)
            if obj is not None:
                yield obj

    def __getitem__(self, item):
        if isinstance(item, slice):
            names = self._live_names()[item]
            return [self._dict[name] for name in names if name in self._dict]
        if isinstance(item, int):
            names = self._live_names()
            return self._dict[names[item]]
        return self._dict[item]  # by name

    def __delitem__(self, key):
        name = key if isinstance(key, str) else self[key].name
        del self._dict[name]

    def keys(self):
        return list(self._live_names())

    def values(self):
        return list(self)

    def items(self):
        return [(name, self._dict[name]) for name in self._live_names() if name in self._dict]

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    def append(self, value):
        if not hasattr(value, "name"):
            raise AttributeError(
                'Object %s does not have a "name" attribute and cannot be stored.' % value)
        name = value.name
        if name in self._dict:
            raise ContainerAlreadyContains(
                "Container '%s' already contains an object with name '%s'." % (self, value.name))
        self._dict[name] = value

    def extend(self, values):
        for value in values:
            self.append(value)

    def update_key(self, key):
        """Re-files an already-stored item under its current .name.

        Called by Variable.name's and Constraint.name's setters (in
        optlang.interface) right after an attached object's name changes, so
        this dict's key stays in sync with the object it points at.
        """
        item = self._dict.get(key)
        if item is None:
            return
        name = item.name
        if key != name:
            self._dict[name] = item
            del self._dict[key]

    def __getstate__(self):
        return list(self)

    def __setstate__(self, obj_list):
        # Unlike optlang.container.Container, this can't just replay
        # self.__init__(obj_list): a live HiGHS-backed container needs its
        # owning Model, which isn't part of the pickled state.
        # Reconstructing the Model rebuilds this instead.
        raise NotImplementedError(
            "%s is backed by a live HiGHS instance and cannot be unpickled on "
            "its own - restore the owning Model instead." % type(self).__name__
        )

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError("'%s' object has no attribute %s" % (self, name))

    def __dir__(self):
        attributes = list(self.__class__.__dict__.keys())
        attributes.extend(self._dict.keys())
        return attributes


class HighsVariablesContainer(_HighsBackedContainer):
    """model.variables, backed by HiGHS's live column order/names."""

    def _num(self):
        return self._model.problem.getNumCol()

    def _name_at(self, idx):
        return self._model.problem.variableName(idx)


class HighsConstraintsContainer(_HighsBackedContainer):
    """model.constraints, backed by HiGHS's live row order/names."""

    def _num(self):
        return self._model.problem.getNumRow()

    def _name_at(self, idx):
        _, name = self._model.problem.getRowName(idx)
        return name


class Model(interface.Model):
    def __init__(self, problem=None, *args, **kwargs):
        self.problem = None
        self._has_solution = False
        self._solution_col_value = None
        self._solution_col_dual = None
        self._solution_row_value = None
        self._solution_row_dual = None
        self._objective_value = None
        self._last_objective_was_quadratic = False
        super(Model, self).__init__(problem=problem, *args, **kwargs)

    def _initialize_problem(self):
        self.problem: highspy.Highs = highspy.Highs()
        self.problem.setOptionValue("output_flag", False)
        # Replace the base class's plain Containers (created before this
        # point, so still empty here) with ones backed by this Highs
        # instance's live column/row order.
        self._variables = HighsVariablesContainer(self)
        self._constraints = HighsConstraintsContainer(self)

    def _initialize_model_from_problem(self, problem: highspy.Highs):
        """
        Copies an already-populated highspy.Highs instance, reconstructing
        optlang Variable/Constraint/Objective wrappers for its current
        columns, rows and cost vector.
        """
        if not isinstance(problem, highspy.Highs):
            raise TypeError(
                "Provided problem must be a highspy.Highs instance, not %s." % type(problem))

        # self.problem = problem
        # self._variables = HighsVariablesContainer(self)
        # self._constraints = HighsConstraintsContainer(self)

        # lp = problem.getLp()

        self._initialize_problem() # create a new highspy.Highs problem
        self.problem.passModel(problem.getModel()) # copy the source problem
        lp = self.problem.getLp()
        problem = self.problem

        inf = highspy.kHighsInf

        # Variables: one wrapper per existing column, bounds, integrality
        # and index taken straight from the live problem.
        integrality = getattr(lp, "integrality_", None)
        for i in range(problem.getNumCol()):
            name = problem.variableName(i)
            lb = float(lp.col_lower_[i])
            ub = float(lp.col_upper_[i])
            is_integer = (integrality is not None and len(integrality) > i and
                          int(integrality[i]) == int(highspy.HighsVarType.kInteger))
            if is_integer:
                var_type = "binary" if (lb == 0.0 and ub == 1.0) else "integer"
            else:
                var_type = "continuous"
            variable = Variable(
                name,
                lb=None if lb <= -inf else lb,
                ub=None if ub >= inf else ub,
                type=var_type,
            )
            # Raw assignment, not variable.problem = self: mirrors
            # _add_variables/the rest of this file, and (unlike Constraint's
            # or Objective's `problem`) Variable.problem is a plain
            # attribute with no side effects, so either would work - this
            # just keeps the pattern identical across all three.
            variable._solver_index = i
            variable.problem = self
            self._variables.append(variable)
            self._variables_to_constraints_mapping[name] = set()

        # Constraints: one wrapper per existing row. Built the same way
        # Constraint.__init__ always is elsewhere in this file - constructed
        # unattached, with _solver_index and _problem only patched in
        # afterwards. Constraint._get_expression touches self._solver_index,
        # so constructing it directly with problem=self here (before that
        # attribute exists) would blow up; this order avoids that.
        for i in range(problem.getNumRow()):
            _, name = problem.getRowName(i)
            _, row_lb, row_ub, _ = problem.getRow(i)
            _, idx, _ = problem.getRowEntries(i)
            # terms = [float(v) * self._variables[problem.variableName(int(j))] for j, v in zip(idx, val)]
            # expression = symbolics.add(terms) if terms else symbolics.sympify(0)
            # constraint = Constraint(
            constraint = Constraint(
                0, # do not set up explicit expression here, can be constructed on demand
                lb=None if row_lb <= -inf else float(row_lb),
                ub=None if row_ub >= inf else float(row_ub),
                name=name,
                sloppy=True,
            )
            constraint._expression_expired = True
            constraint._solver_index = i
            constraint._problem = self
            self._constraints.append(constraint)
            for j in idx:
                var_name = problem.variableName(int(j))
                self._variables_to_constraints_mapping[var_name].add(name)

        # Objective: linear part only, see docstring. Same construction
        # order rationale as Constraint above - Objective._get_expression
        # reads self._quadratic_coeffs, which does not exist until after
        # __init__ returns, so this too is built unattached first.
        obj_expr, sense = problem.getObjective()
        linear_coeffs = {}
        terms = []
        for i, v in zip(obj_expr.idxs, obj_expr.vals):
            var_name = problem.variableName(int(i))
            linear_coeffs[var_name] = float(v)
            terms.append(float(v) * self._variables[var_name])
        constant = float(obj_expr.constant or 0.0)
        if constant:
            terms.append(symbolics.Real(constant))
        expression = symbolics.add(terms) if terms else symbolics.sympify(0)
        direction = "max" if sense == highspy.ObjSense.kMaximize else "min"

        objective = Objective(expression, direction=direction, sloppy=True)
        objective._initial_linear_coeffs = linear_coeffs
        objective._quadratic_coeffs = {}
        objective._constant = constant
        objective._problem = self
        self._objective = objective

    def _build_hessian_data(self, quadratic_coeffs, num_col):
        """
        Helper to build Hessian matrix data (start, index, value) for HiGHS.
        Returns (q_start, q_index, q_value) or (None, None, None) if empty.
        """
        if not quadratic_coeffs:
            return None, None, None

        q_start = np.zeros(num_col + 1, dtype=np.int32)
        q_index_list = []
        q_value_list = []
        q_col_entries = [[] for _ in range(num_col)]
        
        for (v1_name, v2_name), coeff in quadratic_coeffs.items():
            v1 = self._variables.get(v1_name)
            v2 = self._variables.get(v2_name)
            if v1 is None or v2 is None:
                continue
            i = v1._solver_index
            j = v2._solver_index
            
            # Lower Triangular (i >= j)
            if i < j:
                i, j = j, i
                
            val = coeff
            if i == j:
                val = 2.0 * coeff
            
            q_col_entries[j].append((i, val))
            
        # Sort and build CSC arrays
        pos = 0
        for j in range(num_col):
            q_start[j] = pos
            q_col_entries[j].sort(key=lambda x: x[0])
            for (row_idx, value) in q_col_entries[j]:
                q_index_list.append(row_idx)
                q_value_list.append(value)
                pos += 1
        q_start[num_col] = pos
        
        return q_start, np.array(q_index_list, dtype=np.int32), np.array(q_value_list, dtype=np.double)

    def _update_linear_objective(self):
        """
        Fully (re)seeds the linear part (costs, offset, sense) of the objective
        in HiGHS. This only runs when the whole Objective object is (re)assigned
        via the `model.objective = ...` setter, so wiping all costs first and
        seeding from the new objective's bootstrap coefficients is intentional -
        it's a full replace, not an incremental update. Incremental updates go
        through Objective.set_coefficients / _highs_set_objective_coefficients
        instead, which only ever touch the specific columns being changed.
        """
        if self.problem is None or self.objective is None:
            return

        n = self.problem.getNumCol()
        if n > 0:
            self.problem.changeColsCost(n, np.arange(n, dtype=np.int32), np.zeros(n, dtype=np.double))

        # Re-derive from the objective's own (unattached-side) expression
        # rather than trusting _initial_linear_coeffs/_constant verbatim:
        # those are snapshotted once at Objective.__init__ time, but
        # +=/-=/*= (OptimizationExpression's __iadd__/__isub__/__imul__)
        # mutate _expression directly and have no way to know they should
        # keep that snapshot in sync. Re-parsing here, at the one place the
        # snapshot is actually consumed, keeps a mutated-before-attachment
        # Objective (e.g. `obj = Objective(x); obj += 2 * y; model.objective
        # = obj`) working correctly without having to intercept every
        # mutating operator individually.
        linear_coeffs, quadratic_coeffs, obj_constant = \
            _separate_linear_and_quadratic_from_expr(self.objective._expression)
        self.objective._initial_linear_coeffs = linear_coeffs
        # Refresh the Hessian snapshot too, so e.g. `obj += y**2` before
        # attachment is not lost (_update_quadratic_objective reads this).
        self.objective._quadratic_coeffs = quadratic_coeffs
        self.objective._constant = obj_constant

        for var_name, coeff in linear_coeffs.items():
            var = self._variables.get(var_name)
            if var is not None:
                self.problem.changeColCost(var._solver_index, float(coeff))
        
        self.problem.changeObjectiveOffset(float(obj_constant))
        self._highs_set_objective_sense(self.objective.direction)

    def _update_quadratic_objective(self):
        """Updates the quadratic part (Hessian) of the objective in HiGHS."""
        if self.problem is None or self.objective is None:
            return

        # Same rationale as the _highs_set_* helpers below: quadratic terms
        # may reference variables added but not yet flushed.
        self.update()

        quadratic_coeffs = self.objective._quadratic_coeffs
        num_col = self.problem.getNumCol()
        
        q_start, q_index, q_value = self._build_hessian_data(quadratic_coeffs, num_col)
        
        if q_start is not None:
            hessian = highspy.HighsHessian()
            hessian.dim_ = num_col
            hessian.format_ = highspy.HessianFormat.kTriangular
            hessian.start_ = q_start
            hessian.index_ = q_index
            hessian.value_ = q_value
            self.problem.passHessian(hessian)
        else:
            # Clear the Hessian if no quadratic terms
            empty_hessian = highspy.HighsHessian()
            empty_hessian.dim_ = num_col
            empty_hessian.format_ = highspy.HessianFormat.kTriangular
            empty_hessian.start_ = np.zeros(num_col + 1, dtype=np.int32)
            empty_hessian.index_ = np.array([], dtype=np.int32)
            empty_hessian.value_ = np.array([], dtype=np.double)
            self.problem.passHessian(empty_hessian)
            
        self._set_objective_is_quadratic(len(quadratic_coeffs) > 0)

    def _set_objective_is_quadratic(self, is_quadratic):
        """Record whether the objective has a quadratic part.

        HiGHS has a single "solver" option for LP and QP, so whenever the
        objective flips between the two, the matching method (lp_method or
        qp_method) has to be pushed to HiGHS. This is the only place the flag
        is written, so no re-apply is needed at solve time.
        """
        changed = is_quadratic != self._last_objective_was_quadratic
        self._last_objective_was_quadratic = is_quadratic
        # configuration doesn't exist yet if the objective is passed to
        # Model.__init__; Configuration.__init__ applies everything itself then.
        configuration = getattr(self, "configuration", None)
        if changed and configuration is not None:
            configuration._apply_solver_method()

    def _sync_objective_to_solver(self):
        """Sync the current objective's internal state to the HiGHS solver."""
        self._update_linear_objective()
        self._update_quadratic_objective()

    @interface.Model.objective.setter
    def objective(self, value):
        interface.Model.objective.fset(self, value)
        if self.problem is not None:
            self._sync_objective_to_solver()

    def _add_variables(self, variables):
        # Same as interface.Model._add_variables, minus its
        # _variables_to_constraints_mapping bookkeeping: HiGHS's own row data
        # is the sole source of truth for which constraints reference which
        # variable (see Constraint._get_expression / get_linear_coefficients,
        # and _remove_variables below), so no separate mapping is kept here.
        if variables:
            self._invalidate_solution()
        for variable in variables:
            self._variables.append(variable)
            variable.problem = self

        inf = highspy.kHighsInf
        for variable in variables:
            lb = -inf if variable.lb is None else variable.lb
            ub = inf if variable.ub is None else variable.ub
            vtype = (highspy.HighsVarType.kInteger if variable.type in ("integer", "binary")
                     else highspy.HighsVarType.kContinuous)
            # name=... is required here: every read path (Constraint/Objective
            # .expression, _constraint_to_coeffs) maps a
            # HiGHS column index back to an optlang variable name via
            # variableName(idx), which raises if the column was never named.
            self.problem.addVariable(lb, ub, type=vtype, name=variable.name)
            # New columns are always appended at the end, so the newly added
            # variable's index is simply the last column.
            variable._solver_index = self.problem.getNumCol() - 1

    def _add_constraints(self, constraints, sloppy=False):
        # Same as interface.Model._add_constraints, minus its
        # _variables_to_constraints_mapping bookkeeping -- not needed here
        # for the same reason as in _add_variables above.
        if constraints:
            self._invalidate_solution()
        for constraint in constraints:
            if sloppy is False:
                variables = constraint.variables
                if constraint.indicator_variable is not None:
                    variables.add(constraint.indicator_variable)
                missing_vars = [var for var in variables if var.problem is not self]
                if len(missing_vars) > 0:
                    self._add_variables(missing_vars)
            self._constraints.append(constraint)
            constraint._problem = self

        inf = highspy.kHighsInf
        for constraint in constraints:
            coeffs, lb, ub = self._constraint_to_coeffs(constraint)
            row_lb = -inf if lb is None else lb
            row_ub = inf if ub is None else ub
            indices = np.array([self._variables[name]._solver_index for name in coeffs.keys()], dtype=np.int32)
            values = np.array(list(coeffs.values()), dtype=np.double)
            self.problem.addRow(row_lb, row_ub, len(indices), indices, values)
            # New rows are always appended at the end, so the newly added
            # constraint's index is simply the last row.
            row_index = self.problem.getNumRow() - 1
            constraint._solver_index = row_index
            # addRow (unlike addVariable) has no name kwarg, so name the row
            # in a separate call. Needed by variableName's row counterpart
            # (getRowName) used in _constraint_to_coeffs and
            # HighsConstraintsContainer.
            self.problem.passRowName(row_index, constraint.name)

    @staticmethod
    def _rows_referencing_columns(highs: "highspy.Highs", col_indices) -> set:
        """Returns a sorted list of unique row indices containing any of the given variable indices."""
        matching_rows = set()

        for col in col_indices:
            # getColEntries abstracts away CSR/CSC storage formats
            status, indices, _ = highs.getColEntries(col)

            if status == highspy.HighsStatus.kOk:
                matching_rows.update(indices)

        return matching_rows

    def _remove_variables(self, variables):
        if variables:
            self._invalidate_solution()
        for variable in variables:
            try:
                self._variables[variable.name]
            except KeyError:
                raise LookupError("Variable %s not in solver" % variable.name)

        removed_names = set(variable.name for variable in variables)
        removed_indices = sorted(variable._solver_index for variable in variables)
        old_num_col = self.problem.getNumCol()

        # Find exactly which constraints reference any of the columns about
        # to be deleted -- must happen before deleteVars, since that call
        # rewrites the matrix and invalidates these column indices. This
        # replaces the old _variables_to_constraints_mapping lookup below.
        affected_rows = self._rows_referencing_columns(self.problem, removed_indices)

        # Physically delete the columns from HiGHS, not just from our own
        # bookkeeping. This is essential under the ground-truth design:
        # Constraint/Objective now read their linear coefficients straight
        # from HiGHS's live row/cost data rather than from a local cache, so
        # if the column itself is left in place, every row that referenced it
        # keeps reporting it - even though our _variables container has
        # already forgotten about it. Deleting the column here lets HiGHS
        # itself cascade the removal through every row and the objective,
        # exactly like GLPK's glp_del_cols.
        self.problem.deleteVars(len(removed_indices), removed_indices)

        # HiGHS shifts every remaining column's index down to close the gaps
        # left by the deleted ones - shift the remaining variables' own
        # _solver_index attributes to match. But if every removed column was
        # already at the tail (the common case: variables added and then
        # removed again within the same `with model:` block, e.g. slack
        # variables in core.make_scenario_feasible), there are no gaps to
        # close below them and nothing to shift - skip the loop entirely.
        # (Sorted, unique indices whose smallest value equals
        # old_num_col - len(removed_indices) can only be the top
        # len(removed_indices) indices; there's no other way to fit that
        # many distinct values into that range.)
        if removed_indices[0] != old_num_col - len(removed_indices):
            for variable in self._variables:
                if variable.name in removed_names:
                    continue
                shift = sum(1 for r in removed_indices if r < variable._solver_index)
                variable._solver_index -= shift

        # Constraint caches its expression (_expression_expired) -- a
        # constraint referencing a removed variable has a stale cache the
        # moment that variable's column disappears from its row, so mark it
        # expired. affected_rows (computed above, before deleteVars) already
        # pins down exactly which rows those were; resolve each row index
        # back to its constraint via getRowName + self._constraints (an O(1)
        # dict lookup by name) rather than scanning every constraint in the
        # model, and rather than any _variables_to_constraints_mapping.
        for row_index in affected_rows:
            _, row_name = self.problem.getRowName(row_index)
            constraint = self._constraints.get(row_name)
            if constraint is not None:
                constraint._expression_expired = True

        for variable in variables:
            variable.problem = None
            del self._variables[variable.name]

        # The objective's quadratic part is still cached locally (see
        # Objective's docstring - no live Hessian read-back is available yet),
        # so it's the one remaining piece that needs explicit pruning: a
        # lingering quadratic term referencing a removed variable would break
        # _build_hessian_data the next time the objective is rebuilt.
        if self.objective is not None and self.objective._quadratic_coeffs:
            self.objective._quadratic_coeffs = {
                k: v for k, v in self.objective._quadratic_coeffs.items()
                if k[0] not in removed_names and k[1] not in removed_names
            }
            # Deliberately not _update_quadratic_objective(): it calls
            # self.update(), which would re-enter this method while the
            # removal is still pending.
            self._set_objective_is_quadratic(bool(self.objective._quadratic_coeffs))

    def _remove_constraints(self, constraints):
        if constraints:
            self._invalidate_solution()
        for constraint in constraints:
            try:
                self._constraints[constraint.name]
            except KeyError:
                raise LookupError("Constraint %s not in solver" % constraint)

        removed_names = set(constraint.name for constraint in constraints)
        removed_indices = sorted(constraint._solver_index for constraint in constraints)
        old_num_row = self.problem.getNumRow()

        # Mirrors _remove_variables: physically delete the rows from HiGHS
        # and let it cascade the reindexing, rather than rebuilding the
        # whole LP from scratch. Removing rows doesn't touch any column, so
        # unlike variable removal there's no Hessian/objective cache to
        # prune here - constraints don't appear in any other object's cache.
        self.problem.deleteRows(len(removed_indices), removed_indices)

        # Same end-of-range shortcut as _remove_variables: if every removed
        # row was already at the tail, nothing below them moves.
        if removed_indices[0] != old_num_row - len(removed_indices):
            for constraint in self._constraints:
                if constraint.name in removed_names:
                    continue
                shift = sum(1 for r in removed_indices if r < constraint._solver_index)
                constraint._solver_index -= shift

        for constraint in constraints:
            constraint.problem = None
            del self._constraints[constraint.name]

    def _constraint_bounds(self, constraint: Constraint):
        lb, ub = constraint.lb, constraint.ub
        if lb is not None:
            lb = lb - constraint._constant
        if ub is not None:
            ub = ub - constraint._constant
        return lb, ub

    def _constraint_to_coeffs(self, constraint: Constraint):
        lb, ub = self._constraint_bounds(constraint)

        row_index = constraint._solver_index
        if row_index is not None:
            # The row already exists in HiGHS, which is the source of truth
            # for its own coefficients - read them back directly rather than
            # trusting any local copy (there isn't one anymore anyway).
            _, idx, val = self.problem.getRowEntries(row_index)
            coeffs = {
                self.problem.variableName(int(i)): float(v)
                for i, v in zip(idx, val)
            }
        else:
            # Constraint not yet added to this model - use its bootstrap
            # coefficients (this is the path _add_constraints takes).
            coeffs = dict(constraint._initial_coeffs)

        return coeffs, lb, ub

    def _highs_set_col_bounds(self, variable):
        # Flush any pending add()s first: a variable can have its lb/ub
        # setter (or set_bounds) called right after being added to a model,
        # before update() has run to assign it a _solver_index. Without this,
        # changeColBounds would be called with index None. update() is a
        # cheap no-op when there is nothing pending, so this costs nothing
        # in the common case where the variable was already committed.
        self.update()
        inf = highspy.kHighsInf
        lb = -inf if variable.lb is None else variable.lb
        ub = inf if variable.ub is None else variable.ub
        self.problem.changeColBounds(variable._solver_index, lb, ub)

    def _highs_set_col_type(self, variable):
        # Flush any pending add()s first - same rationale as
        # _highs_set_col_bounds: a variable's type can be changed right
        # after it was added to a model, before update() has assigned it a
        # _solver_index. update() only ever touches pending
        # add/remove-variable/constraint bookkeeping here (variable type
        # changes never go through self._pending_modifications - they are
        # written straight through below, exactly like bounds/coefficients
        # elsewhere in this file), so this can never trigger another call
        # back into this method or into the type setter that called us.
        inf = highspy.kHighsInf
        # Push bounds too: interface.Variable.type.fset's 'binary' branch
        # sets self._lb/self._ub directly (bypassing the lb/ub property
        # setters and therefore _highs_set_col_bounds), so HiGHS would
        # otherwise miss that change. Re-sending the (by now settled)
        # bounds here is a cheap no-op for every other case.
        lb = -inf if variable.lb is None else variable.lb
        ub = inf if variable.ub is None else variable.ub
        self.problem.changeColBounds(variable._solver_index, lb, ub)
        vtype = (highspy.HighsVarType.kInteger if variable.type in ("integer", "binary")
                 else highspy.HighsVarType.kContinuous)
        self.problem.changeColIntegrality(variable._solver_index, vtype)

    def _highs_set_row_bounds(self, constraint):
        # See _highs_set_col_bounds -- same rationale, for constraint rows.
        self.update()
        inf = highspy.kHighsInf
        lb, ub = self._constraint_bounds(constraint)
        row_lb = -inf if lb is None else lb
        row_ub = inf if ub is None else ub
        self.problem.changeRowBounds(constraint._solver_index, row_lb, row_ub)

    def _highs_set_col_name(self, variable):
        # Same flush rationale as _highs_set_col_bounds.
        self.update()
        self.problem.passColName(variable._solver_index, variable.name)

    def _highs_set_row_name(self, constraint):
        # Same flush rationale as _highs_set_col_bounds.
        self.update()
        self.problem.passRowName(constraint._solver_index, constraint.name)

    def _highs_set_coefficients(self, constraint, coefficients):
        # Same rationale as _highs_set_col_bounds: coefficients may reference
        # a variable that was just add()ed but not yet update()d, so its
        # _solver_index would still be None. Constraint.set_linear_coefficients
        # already resolves variables through the auto-updating `variables`
        # property in its default (non-sloppy) path, making this redundant
        # there -- but the sloppy=True path bypasses that resolution
        # entirely and calls straight through to here, so this is the one
        # place that also has to hold for that path to be safe by default.
        self.update()
        i = constraint._solver_index
        for var, coeff in coefficients.items():
            j = var._solver_index
            self.problem.changeCoeff(i, j, float(coeff))

    def _highs_set_objective_coefficients(self, coefficients):
        # Unlike Constraint.set_linear_coefficients' default path, Objective's
        # linear coefficient path does not resolve variables through the
        # auto-updating `variables` property before reaching here -- so this
        # is the one place that needs to force freshness for it.
        self.update()
        for var, coeff in coefficients.items():
            self.problem.changeColCost(var._solver_index, float(coeff))

    def _highs_set_objective_sense(self, direction):
        self.problem.changeObjectiveSense(
            highspy.ObjSense.kMaximize if direction == "max" else highspy.ObjSense.kMinimize
        )

    def _optimize(self):
        self._clear_cached_solution()
        h = self.problem
        h.run()

        status_str = h.modelStatusToString(h.getModelStatus())
        status = _HIGHS_STATUS_TO_STATUS.get(status_str, interface.UNDEFINED)

        self._has_solution = status in _STATUSES_WITH_USABLE_SOLUTION

        return status

    def _clear_cached_solution(self):
        """Drop the Python-side copies of the last solution (rebuilt lazily)."""
        self._solution_col_value = None
        self._solution_col_dual = None
        self._solution_row_value = None
        self._solution_row_dual = None
        self._objective_value = None

    def _invalidate_solution(self):
        """Forget the last solution after the model's shape has changed.

        The cached solution vectors are indexed by column/row position. Adding
        columns/rows leaves them too short for the new objects' _solver_index,
        and removing columns/rows shifts every later index down, so a cached
        vector would silently belong to a different variable/constraint. Call
        this from every method that adds or removes columns or rows.
        """
        self._has_solution = False
        self._clear_cached_solution()

    @property
    def is_integer(self):
        return any(int_type != highspy.HighsVarType.kContinuous for int_type in self.problem.getLp().integrality_)

    def _col_values(self):
        # highspy builds a fresh Python list on every attribute access (O(n)),
        # so convert once and keep it.
        if self._solution_col_value is None:
            self._solution_col_value = self.problem.getSolution().col_value
        return self._solution_col_value

    def _col_duals(self):
        if self._solution_col_dual is None:
            self._solution_col_dual = self.problem.getSolution().col_dual
        return self._solution_col_dual

    def _row_values(self):
        if self._solution_row_value is None:
            self._solution_row_value = self.problem.getSolution().row_value
        return self._solution_row_value

    def _row_duals(self):
        if self._solution_row_dual is None:
            self._solution_row_dual = self.problem.getSolution().row_dual
        return self._solution_row_dual

    def _require_dual_solution(self):
        if (self.problem.getInfo().dual_solution_status
                != highspy.SolutionStatus.kSolutionStatusFeasible):
            raise ValueError("Dual values unavailable/not well-defined for integer problems")

    # bulk getters
    def _get_primal_values(self):
        return self._col_values() if self._has_solution else None

    def _get_reduced_costs(self):
        if not self._has_solution:
            return None
        self._require_dual_solution()
        return self._col_duals()

    def _get_constraint_values(self):
        return self._row_values() if self._has_solution else None

    def _get_shadow_prices(self):
        if not self._has_solution:
            return None
        self._require_dual_solution()
        return self._row_duals()

    # single-value accessors
    def _variable_primal(self, variable):
        if not self._has_solution:
            return None
        return self._col_values()[variable._solver_index]

    def _variable_dual(self, variable):
        if not self._has_solution:
            return None
        self._require_dual_solution()
        return self._col_duals()[variable._solver_index]

    def _constraint_primal(self, constraint):
        if not self._has_solution:
            return None
        return self._row_values()[constraint._solver_index]

    def _constraint_dual(self, constraint):
        if not self._has_solution:
            return None
        self._require_dual_solution()
        return self._row_duals()[constraint._solver_index]

    def _get_objective_value(self):
        if not self._has_solution:
            return None
        if self._objective_value is None:
            info = self.problem.getInfo()
            self._objective_value = float(info.objective_function_value)
        return self._objective_value


__all__ = ["Variable", "Constraint", "Objective", "Configuration", "Model"]
