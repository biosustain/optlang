# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

import copy
import json
import os
import pickle
import sys
import unittest

import optlang.interface


try:
    import scipy
except ImportError as e:
    class TestMissingDependency(unittest.TestCase):

        @unittest.skip('Missing dependency - ' + str(e))
        def test_fail(self):
            pass

    sys.exit(0)

from optlang import scipy_interface
from optlang.tests import abstract_test_cases


TESTMODELPATH = os.path.join(os.path.dirname(__file__), 'data/coli_core.json')


class VariableTestCase(abstract_test_cases.AbstractVariableTestCase):
    interface = scipy_interface

    def test_get_primal(self):
        self.assertEqual(self.var.primal, None)
        with open(TESTMODELPATH) as infile:
            model = self.interface.Model.from_json(json.load(infile))

        model.optimize()
        self.assertEqual(model.status, optlang.interface.OPTIMAL)
        for var in model.variables:
            self.assertTrue(var.lb <= round(var.primal, 6) <= var.ub, (var.lb, var.primal, var.ub))

    def test_changing_variable_names_is_reflected_in_the_solver(self):
        self.skipTest("NA")

    def test_set_wrong_type_raises(self):
        self.assertRaises(ValueError, self.interface.Variable, name="test", type="mayo")
        self.assertRaises(Exception, setattr, self.var, 'type', 'ketchup')
        self.model.add(self.var)
        self.model.update()
        self.assertRaises(ValueError, setattr, self.var, "type", "mustard")

    @unittest.skip("Scipy doesn't support duals")
    def test_get_dual(self):
        pass

    def test_change_type(self):
        self.var.type = "continuous"
        self.assertRaises(ValueError, setattr, self.var, "type", "integer")
        self.assertRaises(ValueError, setattr, self.var, "type", "binary")


class ConstraintTestCase(abstract_test_cases.AbstractConstraintTestCase):
    interface = scipy_interface

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

    @unittest.skip("NA")
    def test_get_dual(self):
        pass

    @unittest.skip("NA")
    def test_indicator_constraint_support(self):
        pass


class ObjectiveTestCase(abstract_test_cases.AbstractObjectiveTestCase):
    interface = scipy_interface

    def setUp(self):
        with open(TESTMODELPATH) as infile:
            self.model = self.interface.Model.from_json(json.load(infile))
        self.obj = self.model.objective

    def test_change_direction(self):
        self.obj.direction = "min"
        self.assertEqual(self.obj.direction, "min")
        self.assertEqual(self.model.problem.direction, "min")

        self.obj.direction = "max"
        self.assertEqual(self.obj.direction, "max")
        self.assertEqual(self.model.problem.direction, "max")


class ConfigurationTestCase(abstract_test_cases.AbstractConfigurationTestCase):
    interface = scipy_interface


class ModelTestCase(abstract_test_cases.AbstractModelTestCase):
    interface = scipy_interface

    def setUp(self):
        with open(TESTMODELPATH) as infile:
            self.model = self.interface.Model.from_json(json.load(infile))

    @unittest.skip("NA")
    def test_clone_model_with_lp(self):
        pass

    @unittest.skip("NA")
    def test_optimize_milp(self):
        pass

    @unittest.skip("Not implemented yet")
    def test_pickle_empty_model(self):
        pass

    @unittest.skip("Not implemented yet")
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
        self.model.configuration.verbosity = 3
        self.model.configuration.tolerance = 1e-6
        from_pickle = pickle.loads(pickle.dumps(self.model)).configuration
        self.assertEqual(from_pickle.verbosity, 3)
        self.assertEqual(from_pickle.tolerance, 1e-6)

    @unittest.skip("Not implemented yet")
    def test_config_gets_copied_too(self):
        self.assertEquals(self.model.configuration.verbosity, 0)
        self.model.configuration.verbosity = 3
        model_copy = copy.copy(self.model)
        self.assertEquals(model_copy.configuration.verbosity, 3)

    @unittest.skip("Not implemented yet")
    def test_init_from_existing_problem(self):
        pass

    @unittest.skip("Not implemented yet")
    def test_add_non_cplex_conform_variable(self):
        var = self.interface.Variable('12x!!@#5_3', lb=-666, ub=666)
        self.assertEqual(var.index, None)
        self.model.add(var)
        self.assertTrue(var in self.model.variables.values())
        self.assertEqual(self.model.variables['12x!!@#5_3'].lb, -666)
        self.assertEqual(self.model.variables['12x!!@#5_3'].ub, 666)
        # repickled = pickle.loads(pickle.dumps(self.model))
        # var_from_pickle = repickled.variables['12x!!@#5_3']
        # self.assertEqual(var_from_pickle.name, glp_get_col_name(repickled.problem, var_from_pickle.index))

    def test_add_constraints(self):
        x = self.interface.Variable('x', lb=0, ub=1, type='continuous')
        y = self.interface.Variable('y', lb=-181133.3, ub=12000., type='continuous')
        z = self.interface.Variable('z', lb=0., ub=10., type='continuous')
        constr1 = self.interface.Constraint(0.3 * x + 0.4 * y + 66. * z, lb=-100, ub=0., name='test')
        constr2 = self.interface.Constraint(2.333 * x + y + 3.333, ub=100.33, name='test2')
        constr3 = self.interface.Constraint(2.333 * x + y + z, lb=-300)
        constr4 = self.interface.Constraint(x, lb=-300, ub=-300)
        constr5 = self.interface.Constraint(3 * x)
        self.model.add(constr1)
        self.model.add(constr2)
        self.model.add(constr3)
        self.model.add([constr4, constr5])
        self.assertIn(constr1.name, self.model.constraints)
        self.assertIn(constr2.name, self.model.constraints)
        self.assertIn(constr3.name, self.model.constraints)
        self.assertIn(constr4.name, self.model.constraints)
        self.assertIn(constr5.name, self.model.constraints)

    @unittest.skip("")
    def test_change_of_constraint_is_reflected_in_low_level_solver(self):
        pass

    @unittest.skip("")
    def test_constraint_set_problem_to_None_caches_the_latest_expression_from_solver_instance(self):
        pass

    @unittest.skip("")
    def test_change_of_objective_is_reflected_in_low_level_solver(self):
        pass

    @unittest.skip("")
    def test_change_variable_bounds(self):
        pass

    @unittest.skip("")
    def test_change_constraint_bounds(self):
        pass

    def test_initial_objective(self):
        self.assertIn('BIOMASS_Ecoli_core_w_GAM', self.model.objective.expression.__str__(), )
        self.assertEqual(
            (self.model.objective.expression - (
                1.0 * self.model.variables.BIOMASS_Ecoli_core_w_GAM -
                1.0 * self.model.variables.BIOMASS_Ecoli_core_w_GAM_reverse_712e5)).expand() - 0, 0
        )

    @unittest.skip("Not implemented yet")
    def test_iadd_objective(self):
        pass

    @unittest.skip("Not implemented yet")
    def test_imul_objective(self):
        pass

    @unittest.skip("Not implemented yet")
    def test_set_copied_objective(self):
        obj_copy = copy.copy(self.model.objective)
        self.model.objective = obj_copy
        self.assertEqual(self.model.objective.direction, "max")
        self.assertEqual(
            (self.model.objective.expression - (1.0 * self.model.variables.R_Biomass_Ecoli_core_w_GAM)).expand() - 0, 0
        )

    @unittest.skip("NA")
    def test_timeout(self):
        self.model.configuration.timeout = 0
        status = self.model.optimize()
        self.assertEqual(status, 'time_limit')

    @unittest.skip("Not implemented yet")
    def test_set_linear_coefficients_objective(self):
        self.model.objective.set_linear_coefficients({self.model.variables.R_TPI: 666.})
        # self.assertEqual(glp_get_obj_coef(self.model.problem, self.model.variables.R_TPI.index), 666.)

    @unittest.skip("")
    def test_instantiating_model_with_different_solver_problem_raises(self):
        self.assertRaises(TypeError, self.interface.Model, problem='Chicken soup')

    @unittest.skip("Not implemented yet")
    def test_set_linear_coefficients_constraint(self):
        pass

    def test_shadow_prices(self):
        self.model.optimize()
        self.assertRaises(NotImplementedError, getattr, self.model, "shadow_prices")

    def test_reduced_costs(self):
        self.model.optimize()
        self.assertRaises(NotImplementedError, getattr, self.model, "reduced_costs")

    def test_remove_constraints(self):
        x = self.interface.Variable('x', type='continuous')
        y = self.interface.Variable('y', lb=-181133.3, ub=12000., type='continuous')
        z = self.interface.Variable('z', lb=4, ub=4, type='continuous')
        constr1 = self.interface.Constraint(0.3 * x + 0.4 * y + 66. * z, lb=-100, ub=0., name='test')
        self.assertEqual(constr1.problem, None)
        self.model.add(constr1)
        self.model.update()
        self.assertEqual(constr1.problem, self.model)
        self.assertIn(constr1, self.model.constraints)
        self.model.remove(constr1.name)
        self.model.update()
        self.assertEqual(constr1.problem, None)
        self.assertNotIn(constr1, self.model.constraints)

    def test_add_nonlinear_constraint_raises(self):
        x = self.interface.Variable('x', type='continuous')
        y = self.interface.Variable('y', lb=-181133.3, ub=12000., type='continuous')
        z = self.interface.Variable('z', lb=3, ub=3, type='continuous')
        with self.assertRaises(ValueError):
            constraint = self.interface.Constraint(0.3 * x + 0.4 * y ** x + 66. * z, lb=-100, ub=0., name='test')
            self.model.add(constraint)
            self.model.update()

    def test_change_objective(self):
        v1, v2 = self.model.variables.values()[0:2]
        self.model.objective = self.interface.Objective(1. * v1 + 1. * v2)
        self.assertIn(v1.name, str(self.model.objective))
        self.assertIn(v2.name, str(self.model.objective))

    def test_change_variable_type(self):
        self.assertRaises(ValueError, setattr, self.model.variables[-1], "type", "integer")

    def test_add_integer_var(self):
        self.assertRaises(ValueError, self.interface.Variable, 'int_var', lb=-13, ub=499., type='integer')

    def test_scipy_coefficient_dict(self):
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

    def test_deepcopy(self):
        self.skipTest("Not implemented yet")

    def test_is_integer(self):
        self.skipTest("No integers with scipy")

    def test_integer_variable_dual(self):
        self.skipTest("No duals with scipy")

    def test_integer_constraint_dual(self):
        self.skipTest("No duals with scipy")

    def test_integer_batch_duals(self):
        self.skipTest("No duals with scipy")

    def test_large_objective(self):
        self.skipTest("Quite slow and not necessary")

    def test_binary_variables(self):
        self.skipTest("No integers with scipy")

    def test_implicitly_convert_milp_to_lp(self):
        self.skipTest("No integers with scipy")
