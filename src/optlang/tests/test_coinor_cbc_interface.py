# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

import copy
import json
import os
import pickle
import unittest

import optlang.interface


try:
    import mip
except ImportError as e:

    if str(e).find('mip') >= 0:
        class TestMissingDependency(unittest.TestCase):

            @unittest.skip('Missing dependency - ' + str(e))
            def test_fail(self):
                pass
    else:
        raise

else:

    from optlang import coinor_cbc_interface
    from optlang.tests import abstract_test_cases

    TESTMODELPATH = os.path.join(os.path.dirname(__file__), 'data/coli_core.json')

    def same_ex(ex1, ex2):
        """Compare to expressions for mathematical equality."""
        return ex1.simplify() == ex2.simplify()

    class VariableTestCase(abstract_test_cases.AbstractVariableTestCase):
        interface = coinor_cbc_interface

        def test_get_primal(self):
            self.assertEqual(self.var.primal, None)
            with open(TESTMODELPATH) as infile:
                model = self.interface.Model.from_json(json.load(infile))

            model.optimize()
            self.assertEqual(model.status, optlang.interface.OPTIMAL)
            for var in model.variables:
                self.assertTrue(var.lb <= round(var.primal, 6) <= var.ub, (var.lb, var.primal, var.ub))

        @unittest.skip("COIN-OR Cbc doesn't support variable name change")
        def test_changing_variable_names_is_reflected_in_the_solver(self):
            pass

        @unittest.skip("COIN-OR Cbc doesn't support variable name change")
        def test_change_name(self):
            pass

        def test_set_wrong_type_raises(self):
            self.assertRaises(ValueError, self.interface.Variable, name="test", type="mayo")
            self.assertRaises(Exception, setattr, self.var, 'type', 'ketchup')
            self.model.add(self.var)
            self.model.update()
            self.assertRaises(ValueError, setattr, self.var, "type", "mustard")

        def test_change_type(self):
            self.var.type = "continuous"
            self.assertEqual(self.var.lb, None)
            self.assertEqual(self.var.ub, None)
            self.var.type = "integer"
            self.assertEqual(self.var.lb, None)
            self.assertEqual(self.var.ub, None)
            self.var.type = "binary"
            self.assertEqual(self.var.lb, 0)
            self.assertEqual(self.var.ub, 1)
            self.var.type = "integer"
            self.assertEqual(self.var.lb, 0)
            self.assertEqual(self.var.ub, 1)
            self.var.type = "continuous"
            self.assertEqual(self.var.lb, 0)
            self.assertEqual(self.var.ub, 1)
            self.var.lb = -1.4
            self.var.ub = 1.6
            self.var.type = "integer"
            self.assertEqual(self.var.lb, -1)
            self.assertEqual(self.var.ub, 2)


    class ConstraintTestCase(abstract_test_cases.AbstractConstraintTestCase):
        interface = coinor_cbc_interface

        def test_get_primal(self):

            with open(TESTMODELPATH) as infile:
                self.model = self.interface.Model.from_json(json.load(infile))
            self.assertEqual(self.constraint.primal, None)
            self.model.optimize()
            for c in self.model.constraints:
                if c.lb is not None:
                    self.assertTrue(c.lb <= round(c.primal, 6))
                if c.ub is not None:
                    self.assertTrue(round(c.primal, 6) <= c.ub)

        @unittest.skip("COIN-OR Cbc doesn't support constraint name change")
        def test_change_constraint_name(self):
            pass

        @unittest.skip("TODO: Currently not supported")
        def test_indicator_constraint_support(self):
            pass


    class ObjectiveTestCase(abstract_test_cases.AbstractObjectiveTestCase):
        interface = coinor_cbc_interface

        def setUp(self):
            with open(TESTMODELPATH) as infile:
                self.model = self.interface.Model.from_json(json.load(infile))
            self.obj = self.model.objective

        def test_change_direction(self):
            from mip import MAXIMIZE, MINIMIZE
            self.obj.direction = "min"
            self.assertEqual(self.obj.direction, "min")
            self.assertEqual(self.model.problem.sense, MINIMIZE)

            self.obj.direction = "max"
            self.assertEqual(self.obj.direction, "max")
            self.assertEqual(self.model.problem.sense, MAXIMIZE)


    class ConfigurationTestCase(abstract_test_cases.AbstractConfigurationTestCase):
        interface = coinor_cbc_interface


    class ModelTestCase(abstract_test_cases.AbstractModelTestCase):
        interface = coinor_cbc_interface

        def setUp(self):
            with open(TESTMODELPATH) as infile:
                self.model = self.interface.Model.from_json(json.load(infile))

        def test_pickle_ability(self):
            self.model.optimize()
            value = self.model.objective.value
            pickle_string = pickle.dumps(self.model)
            from_pickle = pickle.loads(pickle_string)
            from_pickle.optimize()
            self.assertAlmostEqual(value, from_pickle.objective.value)
            self.assertEqual([(var.lb, var.ub, var.name, var.type) for var in from_pickle.variables.values()],
                             [(var.lb, var.ub, var.name, var.type) for var in self.model.variables.values()])
            self.assertEqual([(constr.lb, constr.ub, constr.name) for constr in from_pickle.constraints],
                             [(constr.lb, constr.ub, constr.name) for constr in self.model.constraints])

        def test_pickle_keeps_configuration(self):
            configuration = self.model.configuration
            configuration.presolve = True
            configuration.timeout = 60
            configuration.max_nodes = 10
            configuration.threads = 2
            configuration.tolerances.feasibility = 1e-8
            configuration.tolerances.integrality = 1e-8
            from_pickle = pickle.loads(pickle.dumps(self.model)).configuration
            self.assertEqual(from_pickle.presolve, True)
            self.assertEqual(from_pickle.timeout, 60)
            self.assertEqual(from_pickle.max_nodes, 10)
            self.assertEqual(from_pickle.threads, 2)
            self.assertEqual(from_pickle.tolerances.to_dict(), configuration.tolerances.to_dict())

        def test_config_gets_copied_too(self):
            self.assertEqual(self.model.configuration.verbosity, 0)
            self.model.configuration.verbosity = 3
            model_copy = copy.copy(self.model)
            self.assertEqual(model_copy.configuration.verbosity, 3)

        def test_init_from_existing_problem(self):
            self.assertEqual(len(self.model.variables), len(self.model.problem.vars))
            # Divide by 2 because upper and lower constraints are represented seperately
            self.assertEqual(len(self.model.constraints), len(self.model.problem.constrs) / 2)
            self.assertEqual(self.model.variables.keys(),
                             [var.name[2:] for var in self.model.problem.vars])
            # Collect _lower and _upper constraints
            constrs= []
            for con in self.model.constraints:
                constrs.append(con.constraint_name(True))
                constrs.append(con.constraint_name(False))

            self.assertEqual(constrs, [constr.name for constr in self.model.problem.constrs])

        def test_add_non_cplex_conform_variable(self):
            var = self.interface.Variable('12x!!@#5_3', lb=-666, ub=666)
            self.model.add(var)
            self.assertTrue(var in self.model.variables.values())
            self.assertEqual(self.model.variables['12x!!@#5_3'].lb, -666)
            self.assertEqual(self.model.variables['12x!!@#5_3'].ub, 666)
            repickled = pickle.loads(pickle.dumps(self.model))
            var_from_pickle = repickled.variables['12x!!@#5_3']
            self.assertTrue('v_' + var_from_pickle.name in [var.name for var in self.model.problem.vars])

        @unittest.skip("COIN-OR Cbc doesn't support constraint name change")
        def test_change_constraint_name(self):
            pass

        def test_clone_model_with_lp(self):
            self.assertEqual(self.model.configuration.verbosity, 0)
            self.model.configuration.verbosity = 3
            self.model.optimize()
            opt = self.model.objective.value
            cloned_model = self.interface.Model.clone(self.model, use_lp=True)
            self.assertEqual(cloned_model.configuration.verbosity, 3)
            self.assertEqual(len(cloned_model.variables), len(self.model.variables))
            for var in self.model.variables:
                self.assertTrue(var.name in cloned_model.variables)
                var_clone = cloned_model.variables[var.name]
                self.assertEqual(var_clone.lb, var.lb)
                self.assertEqual(var_clone.ub, var.ub)
            self.assertEqual(len(cloned_model.constraints), len(self.model.constraints))
            for con in self.model.constraints:
                self.assertTrue(con.name in cloned_model.constraints)
                con_clone = cloned_model.constraints[con.name]
                self.assertEqual(con_clone.lb, con.lb)
                self.assertEqual(con_clone.ub, con.ub)
            cloned_model.optimize()
            self.assertAlmostEqual(cloned_model.objective.value, opt)

        def test_clone_small_model_with_lp(self):
            x1 = self.interface.Variable('x1', lb=0)
            x2 = self.interface.Variable('x2', lb=0)
            x3 = self.interface.Variable('x3', lb=0)

            # A constraint is constructed from an expression of variables and a lower and/or upper bound (lb and ub).
            c1 = self.interface.Constraint(x1 + x2 + x3, ub=100, name='c1')
            c2 = self.interface.Constraint(10 * x1 + 4 * x2 + 5 * x3, ub=600, name='c2')
            c3 = self.interface.Constraint(2 * x1 + 2 * x2 + 6 * x3, ub=300, name='c3')

            # An objective can be formulated
            obj = self.interface.Objective(10 * x1 + 6 * x2 + 4 * x3, direction='max')

            # Variables, constraints and objective are combined in a Model object, which can subsequently be optimized.
            model = self.interface.Model(name='Simple model')
            model.objective = obj
            model.add([c1, c2, c3])
            model.update()

            self.assertEqual(model.configuration.verbosity, 0)
            model.configuration.verbosity = 3
            model.optimize()
            opt = model.objective.value
            cloned_model = self.interface.Model.clone(model, use_lp=True)
            self.assertEqual(cloned_model.configuration.verbosity, 3)
            self.assertEqual(len(cloned_model.variables), len(model.variables))
            for var in model.variables:
                self.assertTrue(var.name in cloned_model.variables)
                var_clone = cloned_model.variables[var.name]
                self.assertEqual(var_clone.lb, var.lb)
                self.assertEqual(var_clone.ub, var.ub)
            self.assertEqual(len(cloned_model.constraints), len(model.constraints))
            for con in model.constraints:
                self.assertTrue(con.name in cloned_model.constraints)
                con_clone = cloned_model.constraints[con.name]
                self.assertEqual(con_clone.lb, con.lb)
                self.assertEqual(con_clone.ub, con.ub)
            cloned_model.optimize()
            self.assertAlmostEqual(cloned_model.objective.value, opt)

        def test_change_of_constraint_is_reflected_in_low_level_solver(self):
            x = self.interface.Variable('x', lb=0, ub=1, type='continuous')
            y = self.interface.Variable('y', lb=-181133.3, ub=12000., type='continuous')
            z = self.interface.Variable('z', lb=0., ub=10., type='continuous')
            constr1 = self.interface.Constraint(0.3 * x + 0.4 * y + 66. * z, lb=-100, ub=0., name='test')
            self.model.add(constr1)
            self.model.update()
            self.assertEqual(self.model.problem.constr_by_name('c_test_lower').rhs, 100)
            self.assertEqual(self.model.problem.constr_by_name('c_test_upper').rhs, 0)
            constr1.lb = -9
            constr1.ub = 10
            self.assertEqual(self.model.problem.constr_by_name('c_test_lower').rhs, 9)
            self.assertEqual(self.model.problem.constr_by_name('c_test_upper').rhs, 10)
            self.model.optimize()
            constr1.lb = -90
            constr1.ub = 100
            self.assertEqual(self.model.problem.constr_by_name('c_test_lower').rhs, 90)
            self.assertEqual(self.model.problem.constr_by_name('c_test_upper').rhs, 100)

        def test_constraint_set_problem_to_None_caches_the_latest_expression_from_solver_instance(self):
            x = self.interface.Variable('x', lb=-83.3, ub=1324422.)
            y = self.interface.Variable('y', lb=-181133.3, ub=12000.)
            constraint = self.interface.Constraint(0.3 * x + 0.4 * y, lb=-100, name='test')
            self.model.add(constraint)
            z = self.interface.Variable('z', lb=2, ub=5, type='integer')
            constraint += 77. * z
            self.model.remove(constraint)
            self.assertEqual(
                (constraint.expression - (0.4 * y + 0.3 * x + 77.0 * z)).expand() - 0, 0
            )
            self.assertEqual(constraint.lb, -100)
            self.assertEqual(constraint.ub, None)

        def test_change_of_objective_is_reflected_in_low_level_solver(self):
            x = self.interface.Variable('x', lb=-83.3, ub=1324422.)
            y = self.interface.Variable('y', lb=-181133.3, ub=12000.)
            objective = self.interface.Objective(0.3 * x + 0.4 * y, name='test', direction='max')
            self.model.objective = objective
            self.model.update()
            grb_x = self.model.problem.var_by_name('v_' + x.name)
            grb_y = self.model.problem.var_by_name('v_' + y.name)
            expected = {grb_x: 0.3, grb_y: 0.4}

            self.assertEqual(self.model.problem.objective.expr, expected)

            z = self.interface.Variable('z', lb=4, ub=4, type='integer')
            self.model.objective += 77. * z
            self.model.update()
            grb_z = self.model.problem.var_by_name('v_' + z.name)
            expected[grb_z] = 77.

            self.assertEqual(self.model.problem.objective.expr, expected)

        def test_change_variable_bounds(self):
            import random
            inner_prob = self.model.problem
            inner_problem_bounds = [(var.lb, var.ub) for var in inner_prob.vars]
            bounds = [(var.lb, var.ub) for var in self.model.variables.values()]
            self.assertEqual(bounds, inner_problem_bounds)
            for var in self.model.variables.values():
                var.ub = random.uniform(var.lb, 1000)
                var.lb = random.uniform(-1000, var.ub)
            self.model.update()
            inner_problem_bounds_new = [(var.lb, var.ub) for var in inner_prob.vars]
            bounds_new = [(var.lb, var.ub) for var in self.model.variables.values()]
            self.assertNotEqual(bounds, bounds_new)
            self.assertNotEqual(inner_problem_bounds, inner_problem_bounds_new)
            self.assertEqual(bounds_new, inner_problem_bounds_new)

        def test_change_constraint_bounds(self):
            constraint = self.model.constraints[0]
            value = 42
            constraint.ub = value
            self.assertEqual(constraint.ub, value)
            constraint.lb = value
            self.assertEqual(constraint.lb, value)
            name = constraint.name
            self.assertEqual(self.model.problem.constr_by_name('c_' + name + '_upper').rhs, value)
            self.assertEqual(self.model.problem.constr_by_name('c_' + name + '_lower').rhs, -1*value)

        def test_initial_objective(self):
            self.assertIn('BIOMASS_Ecoli_core_w_GAM', self.model.objective.expression.__str__(), )
            self.assertEqual(
                (self.model.objective.expression - (
                    1.0 * self.model.variables.BIOMASS_Ecoli_core_w_GAM -
                    1.0 * self.model.variables.BIOMASS_Ecoli_core_w_GAM_reverse_712e5)).expand() - 0, 0
            )

        def test_change_objective(self):
            v1, v2 = self.model.variables.values()[0:2]

            self.model.objective = self.interface.Objective(1. * v1 + 1. * v2)
            self.assertIn(v1.name, str(self.model.objective))
            self.assertIn(v2.name, str(self.model.objective))
            self.assertTrue(same_ex(self.model.objective._expression, 1.*v1 + 1.*v2))

            self.model.objective = self.interface.Objective(v1 + v2)
            self.assertIn(v1.name, str(self.model.objective))
            self.assertIn(v2.name, str(self.model.objective))
            self.assertTrue(same_ex(self.model.objective._expression, 1.*v1 + 1.*v2))

        def test_iadd_objective(self):
            v2, v3 = self.model.variables.values()[1:3]
            obj_coeff = sorted(self.model.problem.objective.expr.values())
            self.assertEqual(obj_coeff, [-1.0, 1.0])
            self.model.objective += 2. * v2 - 3. * v3
            obj_coeff = sorted(self.model.problem.objective.expr.values())
            self.assertEqual(obj_coeff, [-3.0, -1.0, 1.0, 2.0])

        def test_imul_objective(self):
            self.model.objective *= 2.
            obj_coeff = sorted(self.model.problem.objective.expr.values())
            self.assertEqual(obj_coeff, [-2.0, 2.0])

            v2, v3 = self.model.variables.values()[1:3]

            self.model.objective += 4. * v2 - 3. * v3
            self.model.objective *= 3.
            obj_coeff = sorted(self.model.problem.objective.expr.values())
            self.assertEqual(obj_coeff, [-9.0, -6.0, 6.0, 12.0])

            self.model.objective *= -1
            obj_coeff = sorted(self.model.problem.objective.expr.values())
            self.assertEqual(obj_coeff, [-12.0, -6.0, 6.0, 9.0])

        def test_set_copied_objective(self):
            mip_expr= self.model.problem.objective.expr
            obj_copy = copy.copy(self.model.objective)
            self.model.objective = obj_copy
            self.assertEqual(self.model.objective.direction, "max")
            self.assertEqual(mip_expr, self.model.problem.objective.expr)

        @unittest.skip("CBC-MIP timeout is flaky around 0")
        def test_timeout(self):
            pass

        def test_set_linear_coefficients_objective(self):
            self.model.objective.set_linear_coefficients({self.model.variables.BIOMASS_Ecoli_core_w_GAM: 666.})
            var = self.model.problem.var_by_name('v_' + self.model.variables.BIOMASS_Ecoli_core_w_GAM.name)
            self.assertEqual(self.model.problem.objective.expr[var], 666.)

        def test_set_linear_coefficients_constraint(self):
            constraint = self.model.constraints[0]
            coeff_dict = constraint.expression.as_coefficients_dict()
            self.assertEqual(coeff_dict[self.model.variables.GAPD_reverse_459c1], -1.0)
            constraint.set_linear_coefficients({self.model.variables.GAPD_reverse_459c1: 666.})
            coeff_dict = constraint.expression.as_coefficients_dict()
            self.assertEqual(coeff_dict[self.model.variables.GAPD_reverse_459c1], 666.)

        def test_coinor_cbc_coefficient_dict(self):
            x = self.interface.Variable("x")
            c = self.interface.Constraint(2 ** x, lb=0, sloppy=True)
            obj = self.interface.Objective(2 ** x, sloppy=True)
            model = self.interface.Model()
            self.assertRaises(Exception, setattr, model, "objective", obj)
            self.assertRaises(Exception, model._add_constraint, c)

            c = self.interface.Constraint(0, lb=0)
            obj = self.interface.Objective(0)
            model.add(c)
            model.objective = obj
            self.assertEqual(model.optimize(), optlang.interface.OPTIMAL)

        @unittest.skip('TODO: fix. Not working correctly')
        def test_integer_variable_dual(self):
            from functools import partial
            model = self.interface.Model()
            x = self.interface.Variable("x", lb=0)
            y = self.interface.Variable("y", lb=0)
            c = self.interface.Constraint(x + y, ub=1)
            model.add(c)
            model.objective = self.interface.Objective(x)

            model.optimize()
            self.assertEqual(y.dual, -1)

            x.type = "integer"
            model.optimize()
            # TODO: investigate. abstract test case has y. What should the
            #       behavior be?
            self.assertRaises(ValueError, partial(getattr, x, "dual"))

            x.type = "continuous"
            model.optimize()
            self.assertEqual(y.dual, -1)
            self.assertEqual(x.dual, 0)

        @unittest.skip('TODO: fix. Not working correctly')
        def test_integer_constraint_dual(self):
            pass

        @unittest.skip('TODO: fix. Not working correctly')
        def test_integer_batch_duals(self):
            pass

        def test_relax_with_knapsack(self):

            p = [10, 13, 18, 31, 7, 15]
            w = [11, 15, 20, 35, 10, 33]
            c, I = 47, range(len(w))

            x = [self.interface.Variable(type='binary', name='x{}'.format(i)) for i in I]

            obj = self.interface.Objective(sum(p[i] * x[i] for i in I), direction='max')

            c1 = self.interface.Constraint(sum(w[i] * x[i] for i in I), ub=c)

            model = self.interface.Model(name='knapsack')
            model.objective = obj
            model.add([c1])

            model.configuration.relax = True

            status = model.optimize()
            self.assertTrue(model.problem.relax)
            self.assertEqual(model.status, 'optimal')
            self.assertTrue(model.objective.value >= 41.0)


        def test_max_nodes_max_solutions_with_knapsack(self):

            p = [10, 13, 18, 31, 7, 15]
            w = [11, 15, 20, 35, 10, 33]
            c, I = 47, range(len(w))

            x = [self.interface.Variable(type='binary', name='x{}'.format(i)) for i in I]

            obj = self.interface.Objective(sum(p[i] * x[i] for i in I), direction='max')

            c1 = self.interface.Constraint(sum(w[i] * x[i] for i in I), ub=c)

            model = self.interface.Model(name='knapsack')
            model.objective = obj
            model.add([c1])

            model.configuration.max_nodes = 0
            model.configuration.max_solutions = 0
            status = model.optimize()
            self.assertEqual(model.problem.max_nodes, 0)
            self.assertEqual(model.problem.max_solutions, 0)
            self.assertEqual(model.status, 'feasible')

            model.configuration.max_solutions = 10
            status = model.optimize()
            self.assertEqual(model.problem.max_solutions, 10)
            self.assertEqual(model.status, 'optimal')

        def test_threads_cuts_emphasis_with_knapsack(self):

            p = [10, 13, 18, 31, 7, 15]
            w = [11, 15, 20, 35, 10, 33]
            c, I = 47, range(len(w))

            x = [self.interface.Variable(type='binary', name='x{}'.format(i)) for i in I]

            obj = self.interface.Objective(sum(p[i] * x[i] for i in I), direction='max')

            c1 = self.interface.Constraint(sum(w[i] * x[i] for i in I), ub=c)

            model = self.interface.Model(name='knapsack')
            model.objective = obj
            model.add([c1])

            model.configuration.threads = -1
            model.configuration.cuts = 1
            model.configuration.emphasis = 2

            status = model.optimize()
            self.assertEqual(model.problem.threads, -1)
            self.assertEqual(model.problem.cuts, 1)
            self.assertEqual(model.problem.emphasis, 2)
            self.assertEqual(model.status, 'optimal')



    class MIPExamples(unittest.TestCase):
        interface = coinor_cbc_interface

        def test_constant_objective(self):
            x1 = self.interface.Variable('x1', lb=0, ub=5)
            c1 = self.interface.Constraint(x1, lb=-10, ub=10, name='c1')
            obj = self.interface.Objective(1)
            model = self.interface.Model()
            model.objective = obj
            model.add(c1)
            model.optimize()

            self.assertEqual(model.status, 'optimal')
            self.assertEqual(model.objective.value, 1.0)

        def test_knapsack(self):

            p = [10, 13, 18, 31, 7, 15]
            w = [11, 15, 20, 35, 10, 33]
            c, I = 47, range(len(w))

            x = [self.interface.Variable(type='binary', name='x{}'.format(i)) for i in I]

            obj = self.interface.Objective(sum(p[i] * x[i] for i in I), direction='max')

            c1 = self.interface.Constraint(sum(w[i] * x[i] for i in I), ub=c)

            model = self.interface.Model(name='knapsack')
            model.objective = obj
            model.add([c1])

            status = model.optimize()

            self.assertEqual(model.status, 'optimal')
            self.assertEqual(model.objective.value, 41.0)

            primal_values = [val for val in model.primal_values.values()]
            self.assertEqual(primal_values, [1, 0, 0, 1, 0, 0])

            selected = [i for i in I if x[i].primal >= 0.99]
            self.assertEqual(selected, [0, 3])

        def test_travelling_salesman(self):
            from itertools import product

            # names of places to visit
            places = ['Antwerp', 'Bruges', 'C-Mine', 'Dinant', 'Ghent',
                      'Grand-Place de Bruxelles', 'Hasselt', 'Leuven',
                      'Mechelen', 'Mons', 'Montagne de Bueren', 'Namur',
                      'Remouchamps', 'Waterloo']

            # distances in an upper triangular matrix
            dists = [[83, 81, 113, 52, 42, 73, 44, 23, 91, 105, 90, 124, 57],
                     [161, 160, 39, 89, 151, 110, 90, 99, 177, 143, 193, 100],
                     [90, 125, 82, 13, 57, 71, 123, 38, 72, 59, 82],
                     [123, 77, 81, 71, 91, 72, 64, 24, 62, 63],
                     [51, 114, 72, 54, 69, 139, 105, 155, 62],
                     [70, 25, 22, 52, 90, 56, 105, 16],
                     [45, 61, 111, 36, 61, 57, 70],
                     [23, 71, 67, 48, 85, 29],
                     [74, 89, 69, 107, 36],
                     [117, 65, 125, 43],
                     [54, 22, 84],
                     [60, 44],
                     [97],
                     []]

            # number of nodes and list of vertices
            n, V = len(dists), set(range(len(dists)))

            # distances matrix
            c = [[0 if i == j
                  else dists[i][j-i-1] if j > i
                  else dists[j][i-j-1]
                  for j in V] for i in V]

            # binary variables indicating if arc (i,j) is used on the route or not
            x = [[self.interface.Variable(type='binary', name='x_i={}_j={}_arc'.format(i, j)) for j in V] for i in V]

            # continuous variable to prevent subtours: each city will have a
            # different sequential id in the planned route except the first one
            y = [self.interface.Variable(name='x{}'.format(i)) for i in V]

            # objective function: minimize the distance
            obj = self.interface.Objective(sum(c[i][j]*x[i][j] for i in V for j in V), direction='min')

            # constraint : leave each city only once
            cons = []
            for i in V:
                cons.append(self.interface.Constraint(sum(x[i][j] for j in V - {i}), lb=1, ub=1))

            # constraint : enter each city only once
            for i in V:
                cons.append(self.interface.Constraint(sum(x[j][i] for j in V - {i}), lb=1, ub=1))

            # subtour elimination
            for (i, j) in product(V - {0}, V - {0}):
                if i != j:
                    cons.append(self.interface.Constraint(y[i] - (n+1)*x[i][j] - y[j], lb=-1*n))

            model = self.interface.Model(name='travelling_salesman')
            model.objective = obj
            model.add(cons)
            model.optimize()

            self.assertEqual(model.status, 'optimal')
            self.assertEqual(model.objective.value, 547.0)
