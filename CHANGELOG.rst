=======

Next Release
-----
* validate GLPK optimal MIP primal equalities and bounds, retry invalid solutions
  without presolve, and report undefined if the retry is still invalid

1.9.1
-----
* expressions for Constraint in the matrix interface are now assembled from only non-zero
  coefficients, speeding up JSON conversion as a side effect
* fixed a typo in the matrix interface

1.9.0
-----
* enable Python 3.14
* bump OSQP to 1.0+

1.8.3
-----
* fix the objective offset test for compatibility with Debian sid
* enable compatibility with Gurobi 12.0
* fix an issue where the removal of more than 350 constraints
  would lead to dangling references
* add support for Python 3.13

1.8.2
-----
* fix the feasibility check in the hybrid solver
* make optlang compatible with numpy>=2.0

1.8.1
-----
* update versioneer to support newer Python versions

1.8.0
-----
* add a generic matrix interface to allow easy addition of new solvers
  that expect an immutable problem in standard form as input
* replace the OSQP interface with a hybrid interface that uses HIGHS for (MI)LPs and
  OSQP for QPs
* `osqp_interface` is now deprecated, will import the hybrid interface when used, and
  will be removed entirely soon

1.7.0
-----
* remove deprecated numpy type casts
* The symbolics module now has consistent exports
* When sympy is used the internal Symbol class now derives from sympy.core.Dummy. This
  circumvents the hack in place to make Symbols unique and makes optlang work with
  sympy>=1.12 again.
* Updated the scipy and the jsonschema tests to work with newer versions of those packages.
* Package version dependencies are now more specific.
* Tests are run for sympy and symengine now.
* Updated support Python versions to >=3.8.


1.6.1
-----
* fix the Gurobi version check to allow 10.0

1.6.0
-----
* fixes problem scaling for GLPK
* fixes scaling output in GLPK that could not be turned off
* Major performance improvements in the Gurobi interface when getting primals,
  shadow prices and reduced costs
* Now only supports gurobipy>=9.5
* Tests are now run with the Gurobi and CPLEX community editions
* Drop support for Python 2 and add support for Python 3.10

1.5.2
-----
* Gurobi can now serialize its configuration correctly. This also fixes pickling of Gurobi models.
* Fix the shim for integrality tolerance in OSQP which makes it easier to clone OSQP Models to other solvers.
* Fix an issue where one could not rename variables in Gurobi version 9.

1.5.1
-----
* GLPK now respects `Configuration.tolerances.integrality` again

1.5.0
-----
* removed support for Python 3.5
* added support for Python 3.9
* enabled code coverage in tox
* support symengine 0.7.0
* add [OSQP](https://github.com/oxfordcontrol/osqp) as additional solver
* add [cbc](https://github.com/coin-or/python-mip) as additional solver

1.4.7
-----
* fix: except AttributeError when setting tolerance on cloned Configuration
* enable cloning models between solver interfaces that do not support the same set of tolerance parameters
