# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

import copy
import os
import pickle
import pytest
import random
import sys
import unittest


try:  # noqa: C901
    import cplex
except ImportError as e:

    if str(e).find('cplex') >= 0:
        class TestMissingDependency(unittest.TestCase):

            @unittest.skip('Missing dependency - ' + str(e))
            def test_fail(self):
                pass
    else:
        raise
else:

    from optlang import cplex_interface
    from optlang.cplex_interface import Constraint, Model, Objective, Variable
    from optlang.exceptions import SolverError
    from optlang.interface import INFEASIBLE, OPTIMAL
    from optlang.tests import abstract_test_cases

    CplexSolverError = cplex.exceptions.CplexSolverError

    random.seed(666)
    TESTMODELPATH = os.path.join(os.path.dirname(__file__), 'data/model.lp')
    TESTMILPMODELPATH = os.path.join(os.path.dirname(__file__), 'data/simple_milp.lp')
    CONVEX_QP_PATH = os.path.join(os.path.dirname(__file__), 'data/qplib_3256.lp')
    NONCONVEX_QP_PATH = os.path.join(os.path.dirname(__file__), 'data/qplib_1832.lp')


    class VariableTestCase(abstract_test_cases.AbstractVariableTestCase):
        __test__ = True

        interface = cplex_interface

        def test_get_primal(self):
            self.assertEqual(self.var.primal, None)
            problem = cplex.Cplex()
            problem.read(TESTMODELPATH)
            model = Model(problem=problem)
            model.optimize()
            self.assertEqual(model.status, 'optimal')
            self.assertAlmostEqual(model.objective.value, 0.8739215069684305)
            print([var.primal for var in model.variables])
            for i, j in zip([var.primal for var in model.variables],
                            [0.8739215069684306, -16.023526143167608, 16.023526143167604, -14.71613956874283,
                             14.71613956874283, 4.959984944574658, 4.959984944574657, 4.959984944574658,
                             3.1162689467973905e-29, 2.926716099010601e-29, 0.0, 0.0, -6.112235045340358e-30,
                             -5.6659435396316186e-30, 0.0, -4.922925402711085e-29, 0.0, 9.282532599166613, 0.0,
                             6.00724957535033, 6.007249575350331, 6.00724957535033, -5.064375661482091,
                             1.7581774441067828, 0.0, 7.477381962160285, 0.0, 0.22346172933182767, 45.514009774517454,
                             8.39, 0.0, 6.007249575350331, 0.0, -4.541857463865631, 0.0, 5.064375661482091, 0.0, 0.0,
                             2.504309470368734, 0.0, 0.0, -22.809833310204958, 22.809833310204958, 7.477381962160285,
                             7.477381962160285, 1.1814980932459636, 1.496983757261567, -0.0, 0.0, 4.860861146496815,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.064375661482091, 0.0, 5.064375661482091, 0.0, 0.0,
                             1.496983757261567, 10.000000000000002, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, -29.175827135565804,
                             43.598985311997524, 29.175827135565804, 0.0, 0.0, 0.0, -1.2332237321082153e-29,
                             3.2148950476847613, 38.53460965051542, 5.064375661482091, 0.0, -1.2812714099825612e-29,
                             -1.1331887079263237e-29, 17.530865429786694, 0.0, 0.0, 0.0, 4.765319193197458,
                             -4.765319193197457, 21.79949265599876, -21.79949265599876, -3.2148950476847613, 0.0,
                             -2.281503094067127, 2.6784818505075303, 0.0]):
                self.assertAlmostEqual(i, j)

        def test_changing_variable_names_is_reflected_in_the_solver(self):
            model = Model(problem=cplex.Cplex(TESTMODELPATH))
            for i, variable in enumerate(model.variables):
                old_name = variable.name
                variable.name = "var" + str(i)
                self.assertEqual(variable.name, "var" + str(i))
                self.assertEqual(model.problem.variables.get_names(i), "var" + str(i))
                self.assertIn("var" + str(i), model._variables_to_constraints_mapping)
                self.assertNotIn(old_name, model._variables_to_constraints_mapping)

        def test_cplex_setting_bounds(self):
            problem = cplex.Cplex()
            problem.read(TESTMODELPATH)
            model = Model(problem=problem)
            var = model.variables[0]
            var.lb = 1
            self.assertEqual(var.lb, 1)
            model.update()
            self.assertEqual(model.problem.variables.get_lower_bounds(var.name), 1)
            var.ub = 2
            self.assertEqual(var.ub, 2)
            model.update()
            self.assertEqual(model.problem.variables.get_upper_bounds(var.name), 2)


    class ConstraintTestCase(abstract_test_cases.AbstractConstraintTestCase):
        interface = cplex_interface

        def test_set_linear_coefficients(self):
            self.model.add(self.constraint)
            self.constraint.set_linear_coefficients({Variable('chip'): 33., self.model.variables.R_PGK: -33})
            sparse_pair = self.model.problem.linear_constraints.get_rows(self.constraint.name)
            names = [self.model.problem.variables.get_names(i) for i in sparse_pair.ind]
            self.assertEqual(dict(zip(names, sparse_pair.val)),
                             dict([('R_PGK', -33.0), ('chap', 1.0), ('chip', 33.0)]))

        def test_get_primal(self):
            self.assertEqual(self.constraint.primal, None)
            self.model.optimize()
            self.assertEqual(self.model.status, 'optimal')
            self.assertAlmostEqual(self.model.objective.value, 0.8739215069684305)
            print([constraint.primal for constraint in self.model.constraints])
            for i, j in zip([constraint.primal for constraint in self.model.constraints],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             4.048900234729145e-15, 0.0, 0.0, 0.0, -3.55971196577979e-16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 2.5546369406238147e-17, 0.0, -5.080374405378186e-29, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
                self.assertAlmostEqual(i, j)

        def test_cplex_indicator_constraint_expression(self):
            model = self.interface.Model()
            x = self.interface.Variable("x")
            y = self.interface.Variable("y", type="binary")
            c = self.interface.Constraint(x, lb=0, indicator_variable=y, active_when=1)
            self.assertEqual(c.expression - x, 0)

            model.add(c)
            model.update()
            self.assertEqual(c.expression - x, 0)

    class ObjectiveTestCase(abstract_test_cases.AbstractObjectiveTestCase):
        interface = cplex_interface

        def setUp(self):
            problem = cplex.Cplex()
            problem.read(TESTMODELPATH)
            self.model = Model(problem=problem)
            self.obj = self.model.objective

        def test_change_direction(self):
            self.obj.direction = "min"
            self.assertEqual(self.obj.direction, "min")
            self.assertEqual(self.model.problem.objective.get_sense(), self.model.problem.objective.sense.minimize)

            self.obj.direction = "max"
            self.assertEqual(self.obj.direction, "max")
            self.assertEqual(self.model.problem.objective.get_sense(), self.model.problem.objective.sense.maximize)


    class ModelTestCase(abstract_test_cases.AbstractModelTestCase):
        interface = cplex_interface

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

        def test_config_gets_copied_too(self):
            self.assertEqual(self.model.configuration.verbosity, 0)
            self.model.configuration.verbosity = 3
            model_copy = copy.copy(self.model)
            self.assertEqual(model_copy.configuration.verbosity, 3)

        def test_pickle_keeps_configuration(self):
            configuration = self.model.configuration
            configuration.lp_method = "dual"
            configuration.qp_method = "barrier"
            configuration.solution_target = "convex"
            configuration.presolve = True
            configuration.timeout = 60
            configuration.tolerances.feasibility = 1e-8
            configuration.tolerances.optimality = 1e-8
            configuration.tolerances.integrality = 1e-8
            from_pickle = pickle.loads(pickle.dumps(self.model)).configuration
            self.assertEqual(from_pickle.lp_method, "dual")
            self.assertEqual(from_pickle.qp_method, "barrier")
            self.assertEqual(from_pickle.solution_target, "convex")
            self.assertEqual(from_pickle.presolve, True)
            self.assertEqual(from_pickle.timeout, 60)
            self.assertEqual(from_pickle.tolerances.to_dict(), configuration.tolerances.to_dict())

        def test_load_model_pickled_by_previous_version(self):
            # checks the compatibility with models pickled by optlang <= 1.9.1, which stored the configuration itself
            self.model.configuration.timeout = 60
            state = self.model.__getstate__()
            state = {'cplex_binary': state['cplex_binary'], 'status': state['status'],
                     'config': self.model.configuration}
            model = self.interface.Model.__new__(self.interface.Model)
            model.__setstate__(pickle.loads(pickle.dumps(state)))
            self.assertEqual(model.configuration.timeout, 60)
            self.assertEqual(len(model.variables), len(self.model.variables))

        def test_pickle_keeps_parameters_set_on_problem(self):
            parameters = self.model.problem.parameters
            parameters.threads.set(2)
            parameters.mip.tolerances.mipgap.set(0.05)
            from_pickle = pickle.loads(pickle.dumps(self.model)).problem.parameters
            self.assertEqual(from_pickle.threads.get(), 2)
            self.assertEqual(from_pickle.mip.tolerances.mipgap.get(), 0.05)

        def test_init_from_existing_problem(self):
            inner_prob = self.model.problem
            self.assertEqual(len(self.model.variables), inner_prob.variables.get_num())
            self.assertEqual(len(self.model.constraints),
                             inner_prob.linear_constraints.get_num() + inner_prob.quadratic_constraints.get_num())
            self.assertEqual(self.model.variables.keys(), inner_prob.variables.get_names())
            self.assertEqual(self.model.constraints.keys(), inner_prob.linear_constraints.get_names())

        def test_add_non_cplex_conform_variable(self):
            var = Variable('12x!!@#5_3', lb=-666, ub=666)
            self.model.add(var)
            self.assertTrue(var in self.model.variables.values())
            self.assertEqual(var.name, self.model.problem.variables.get_names(len(self.model.variables) - 1))
            self.assertEqual(self.model.variables['12x!!@#5_3'].lb, -666)
            self.assertEqual(self.model.variables['12x!!@#5_3'].ub, 666)
            repickled = pickle.loads(pickle.dumps(self.model))
            print(repickled.variables)
            var_from_pickle = repickled.variables['12x!!@#5_3']
            self.assertEqual(var_from_pickle.name, self.model.problem.variables.get_names()[-1])

        def test_cplex_remove_variable(self):
            var = self.model.variables[0]
            self.assertEqual(var.problem, self.model)
            self.model.remove(var)
            self.model.update()
            self.assertNotIn(var.name, self.model.problem.variables.get_names())
            self.assertEqual(var.problem, None)

        @unittest.skip('Skipping for now')
        def test_add_quadratic_constraints(self):
            x = Variable('x', lb=-83.3, ub=1324422., type='binary')
            y = Variable('y', lb=-181133.3, ub=12000., type='continuous')
            z = Variable('z', lb=0.000003, ub=0.000003, type='integer')
            constr1 = Constraint(0.3 * x * y + 0.4 * y ** 2 + 66. * z, lb=-100, ub=0., name='test')
            constr2 = Constraint(2.333 * x * x + y + 3.333, ub=100.33, name='test2')
            constr3 = Constraint(2.333 * x + y ** 2 + z + 33, ub=100.33, lb=-300)
            self.model.add(constr1)
            self.model.add(constr2)
            self.model.add(constr3)
            self.assertIn(constr1, self.model.constraints)
            self.assertIn(constr2, self.model.constraints)
            self.assertIn(constr3, self.model.constraints)
            cplex_lines = [line.strip() for line in str(self.model).split('\n')]
            self.assertIn('test:       0.4 y + 66 z + 0.3 x - Rgtest  = -100', cplex_lines)
            self.assertIn('test2:      y + 2.333 x <= 96.997', cplex_lines)
            # Dummy_21:   y + z + 2.333 x - RgDummy_21  = -300
            self.assertRegexpMatches(str(self.model), '\s*Dummy_\d+:\s*y \+ z \+ 2\.333 x - .*  = -300')
            print(self.model)

        def test_change_of_constraint_is_reflected_in_low_level_solver(self):
            x = Variable('x', lb=-83.3, ub=1324422.)
            y = Variable('y', lb=-181133.3, ub=12000.)
            constraint = Constraint(0.3 * x + 0.4 * y, lb=-100, name='test')
            self.model.add(constraint)
            self.assertEqual(
                (self.model.constraints['test'].expression - (0.4 * y + 0.3 * x)).expand() - 0,
                0
            )

            # self.assertEqual(self.model.constraints['test'].__str__(), 'test: -100 <= 0.4*y + 0.3*x')
            self.assertEqual(self.model.problem.linear_constraints.get_coefficients([('test', 'x'), ('test', 'y')]),
                             [0.3, 0.4])
            z = Variable('z', lb=3, ub=4, type='integer')
            constraint += 77. * z
            self.assertEqual(
                self.model.problem.linear_constraints.get_coefficients([('test', 'x'), ('test', 'y'), ('test', 'z')]),
                [0.3, 0.4, 77.])
            self.assertEqual(
                (self.model.constraints['test'].expression - (0.4 * y + 0.3 * x + 77.0 * z)).expand() - 0,
                0
            )

        def test_constraint_set_problem_to_None_caches_the_latest_expression_from_solver_instance(self):
            x = Variable('x', lb=-83.3, ub=1324422.)
            y = Variable('y', lb=-181133.3, ub=12000.)
            constraint = Constraint(0.3 * x + 0.4 * y, lb=-100, name='test')
            self.model.add(constraint)
            z = Variable('z', lb=2, ub=5, type='integer')
            constraint += 77. * z
            self.model.remove(constraint)
            self.assertEqual(constraint.name, "test")
            self.assertEqual(constraint.lb, -100)
            self.assertEqual(constraint.ub, None)
            self.assertEqual((constraint.expression - (0.4 * y + 0.3 * x + 77.0 * z)).expand(), 0)

        def test_change_of_objective_is_reflected_in_low_level_solver(self):
            x = Variable('x', lb=-83.3, ub=1324422.)
            y = Variable('y', lb=-181133.3, ub=12000.)
            objective = Objective(0.3 * x + 0.4 * y, name='obj', direction='max')
            self.model.objective = objective
            for variable in self.model.variables:
                coeff = self.model.problem.objective.get_linear(variable.name)
                if variable.name == 'x':
                    self.assertEqual(coeff, 0.3)
                elif variable.name == 'y':
                    self.assertEqual(coeff, 0.4)
                else:
                    self.assertEqual(coeff, 0.)
            z = Variable('z', lb=0.000003, ub=0.000003, type='continuous')
            objective += 77. * z
            for variable in self.model.variables:
                coeff = self.model.problem.objective.get_linear(variable.name)
                if variable.name == 'x':
                    self.assertEqual(coeff, 0.3)
                elif variable.name == 'y':
                    self.assertEqual(coeff, 0.4)
                elif variable.name == 'z':
                    self.assertEqual(coeff, 77.)
                else:
                    self.assertEqual(coeff, 0.)

        def test_change_variable_bounds(self):
            inner_prob = self.model.problem
            inner_problem_bounds = list(
                zip(inner_prob.variables.get_lower_bounds(), inner_prob.variables.get_upper_bounds())
            )
            bounds = [(var.lb, var.ub) for var in self.model.variables.values()]
            self.assertEqual(bounds, inner_problem_bounds)
            for var in self.model.variables.values():
                var.lb = random.uniform(-1000, 1000)
                var.ub = random.uniform(var.lb, 1000)
            self.model.update()
            inner_problem_bounds_new = list(
                zip(inner_prob.variables.get_lower_bounds(), inner_prob.variables.get_upper_bounds())
            )
            bounds_new = [(var.lb, var.ub) for var in self.model.variables.values()]
            self.assertNotEqual(bounds, bounds_new)
            self.assertNotEqual(inner_problem_bounds, inner_problem_bounds_new)
            self.assertEqual(bounds_new, inner_problem_bounds_new)

        @unittest.skip("The cplex get_types function seems to be broken.")
        def test_cplex_change_variable_type(self):
            for variable in self.model.variables:
                variable.type = 'integer'
            # There seems to be a bug in the get_types function TODO Fix this
            # It doesn't behave according to cplex docs
            self.assertEqual(set(self.model.problem.variables.get_types()), {'I'})

        def test_change_constraint_bounds(self):
            constraint = self.model.constraints[0]
            value = 42
            constraint.ub = value
            self.assertEqual(constraint.ub, value)
            constraint.lb = value
            self.assertEqual(constraint.lb, value)
            self.assertEqual(self.model.problem.linear_constraints.get_senses(constraint.name), "E")
            self.assertEqual(self.model.problem.linear_constraints.get_range_values(constraint.name), 0)

        def test_iadd_objective(self):
            v2, v3 = self.model.variables.values()[1:3]
            self.model.objective += 2. * v2 - 3. * v3
            obj_coeff = self.model.problem.objective.get_linear()
            self.assertEqual(obj_coeff,
                             [1.0, 2.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0]
                             )

        def test_imul_objective(self):
            self.model.objective *= 2.
            obj_coeff = self.model.problem.objective.get_linear()
            print(obj_coeff)
            self.assertEqual(obj_coeff,
                             [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0]
                             )

        def test_set_copied_objective(self):
            obj_copy = copy.copy(self.model.objective)
            self.model.objective = obj_copy
            self.assertEqual(self.model.objective.direction, "max")
            self.assertEqual(self.model.objective.expression, 1.0 * self.model.variables["R_Biomass_Ecoli_core_w_GAM"])

        @pytest.mark.xfail(sys.platform == "win32", reason="buggy with windows clocks")
        def test_timeout(self):
            self.model.configuration.timeout = 0
            status = self.model.optimize()
            self.assertEqual(status, 'time_limit')

        def test_set_linear_coefficients_objective(self):
            self.model.objective.set_linear_coefficients({self.model.variables.R_TPI: 666.})
            self.assertEqual(self.model.problem.objective.get_linear(self.model.variables.R_TPI.name), 666.)

        def test_set_linear_coefficients_constraint(self):
            constraint = self.model.constraints.M_atp_c
            coeff_dict = constraint.expression.as_coefficients_dict()
            self.assertEqual(coeff_dict[self.model.variables.R_Biomass_Ecoli_core_w_GAM], -59.8100000000000)
            constraint.set_linear_coefficients({self.model.variables.R_Biomass_Ecoli_core_w_GAM: 666.})
            coeff_dict = constraint.expression.as_coefficients_dict()
            self.assertEqual(coeff_dict[self.model.variables.R_Biomass_Ecoli_core_w_GAM], 666.)

        def test_cplex_change_objective_can_handle_removed_vars(self):
            self.model.objective = Objective(self.model.variables[0])
            self.model.remove(self.model.variables[0])
            self.model.update()
            self.model.objective = Objective(self.model.variables[1] ** 2)
            self.model.remove(self.model.variables[1])
            self.model.update()
            self.model.objective = Objective(self.model.variables[2])

        @pytest.mark.skipif(os.environ.get("CI", "false") == "true", reason="not supported on community edition")
        def test_large_objective(self):
            super().test_large_objective()


    class ConfigurationTestCase(abstract_test_cases.AbstractConfigurationTestCase):

        interface = cplex_interface

        def setUp(self):
            self.model = Model()
            self.configuration = self.model.configuration

        def test_lp_method(self):
            for option in cplex_interface._LP_METHODS:
                self.configuration.lp_method = option
                self.assertEqual(self.configuration.lp_method, option)
                self.assertEqual(self.model.problem.parameters.lpmethod.get(),
                                 getattr(self.model.problem.parameters.lpmethod.values, option))

            self.assertRaises(ValueError, setattr, self.configuration, "lp_method", "weird_stuff")

        def test_qp_method(self):
            for option in cplex_interface._QP_METHODS:
                self.configuration.qp_method = option
                self.assertEqual(self.configuration.qp_method, option)
                self.assertEqual(self.model.problem.parameters.qpmethod.get(),
                                 getattr(self.model.problem.parameters.qpmethod.values, option))

            self.assertRaises(ValueError, setattr, self.configuration, "qp_method", "weird_stuff")
            self.configuration.solution_target = None
            if hasattr(self.model.problem.parameters, "optimalitytarget"):
                self.assertEqual(self.model.problem.parameters.optimalitytarget.get(),
                                 self.model.problem.parameters.optimalitytarget.default())
            else:
                self.assertEqual(self.model.problem.parameters.solutiontarget.get(),
                                 self.model.problem.parameters.solutiontarget.default())

        def test_solution_method(self):
            for option in cplex_interface._SOLUTION_TARGETS:
                self.configuration.solution_target = option
                self.assertEqual(self.configuration.solution_target, option)
                if hasattr(self.model.problem.parameters, "optimalitytarget"):
                    self.assertEqual(self.model.problem.parameters.optimalitytarget.get(),
                                     cplex_interface._SOLUTION_TARGETS.index(option))
                else:
                    self.assertEqual(self.model.problem.parameters.solutiontarget.get(),
                                     cplex_interface._SOLUTION_TARGETS.index(option))

            self.assertRaises(ValueError, setattr, self.configuration, "solution_target", "weird_stuff")

        def test_verbosity(self):
            for i in range(4):
                self.model.configuration.verbosity = i
                self.assertEqual(self.model.configuration.verbosity, i)
            self.assertRaises(ValueError, setattr, self.configuration, "verbosity", 8)

        def test_presolve(self):
            for presolve in (True, False, "auto"):
                self.configuration.presolve = presolve
                self.assertEqual(self.configuration.presolve, presolve)

            self.assertRaises(ValueError, setattr, self.configuration, "presolve", "trump")


    class QuadraticProgrammingTestCase(abstract_test_cases.AbstractQuadraticProgrammingTestCase):
        def setUp(self):
            self.model = Model()
            self.x1 = Variable("x1", lb=0)
            self.x2 = Variable("x2", lb=0)
            self.c1 = Constraint(self.x1 + self.x2, lb=1)
            self.model.add([self.x1, self.x2, self.c1])

        def test_convex_obj(self):
            model = self.model
            obj = Objective(self.x1 ** 2 + self.x2 ** 2, direction="min")
            model.objective = obj
            model.optimize()
            self.assertAlmostEqual(model.objective.value, 0.5)
            self.assertAlmostEqual(self.x1.primal, 0.5)
            self.assertAlmostEqual(self.x2.primal, 0.5)

            obj_2 = Objective(self.x1, direction="min")
            model.objective = obj_2
            model.optimize()
            self.assertAlmostEqual(model.objective.value, 0.0)
            self.assertAlmostEqual(self.x1.primal, 0.0)
            self.assertGreaterEqual(self.x2.primal, 1.0)

        def test_non_convex_obj(self):
            model = self.model
            obj = Objective(self.x1 * self.x2, direction="min")
            model.objective = obj
            model.configuration.solution_target = "convex"
            self.assertRaises(SolverError, model.optimize)
            model.configuration.solution_target = "local"
            model.configuration.qp_method = "barrier"
            model.optimize()
            self.assertAlmostEqual(model.objective.value, 0)
            model.configuration.solution_target = "global"
            model.optimize()
            self.assertAlmostEqual(model.objective.value, 0)

            obj_2 = Objective(self.x1, direction="min")
            model.objective = obj_2
            model.optimize()
            self.assertAlmostEqual(model.objective.value, 0.0)
            self.assertAlmostEqual(self.x1.primal, 0.0)
            self.assertGreaterEqual(self.x2.primal, 1.0)

        @unittest.skip("")
        def test_qp_convex(self):
            problem = cplex.Cplex()
            problem.read(CONVEX_QP_PATH)
            model = Model(problem=problem)
            self.assertEqual(len(model.variables), 651)
            self.assertEqual(len(model.constraints), 501)
            for constraint in model.constraints:
                self.assertTrue(constraint.is_Linear, "%s should be linear" % (str(constraint.expression)))
                self.assertFalse(constraint.is_Quadratic, "%s should not be quadratic" % (str(constraint.expression)))

            self.assertTrue(model.objective.is_Quadratic, "objective should be quadratic")
            self.assertFalse(model.objective.is_Linear, "objective should not be linear")

            model.optimize()
            self.assertAlmostEqual(model.objective.value, 32.2291282)

        @unittest.skip("Solving this is slow")
        def test_qp_non_convex(self):
            problem = cplex.Cplex()
            problem.read(NONCONVEX_QP_PATH)
            model = Model(problem=problem)
            self.assertEqual(len(model.variables), 31)
            self.assertEqual(len(model.constraints), 1)
            for constraint in model.constraints:
                self.assertTrue(constraint.is_Linear, "%s should be linear" % (str(constraint.expression)))
                self.assertFalse(constraint.is_Quadratic, "%s should not be quadratic" % (str(constraint.expression)))

            self.assertTrue(model.objective.is_Quadratic, "objective should be quadratic")
            self.assertFalse(model.objective.is_Linear, "objective should not be linear")

            model.configuration.solution_target = "convex"
            self.assertRaises(CplexSolverError, model.optimize)

            model.configuration.solution_target = "global"
            model.optimize()
            self.assertAlmostEqual(model.objective.value, 2441.999999971)

        def test_quadratic_objective_expression(self):
            objective = Objective(self.x1 ** 2 + self.x2 ** 2, direction="min")
            self.model.objective = objective
            self.assertEqual((self.model.objective.expression - (self.x1 ** 2 + self.x2 ** 2)).simplify(), 0)

    class InfeasibleMIPTestCase(unittest.TestCase):

        interface = cplex_interface

        def setUp(self):
            model = self.interface.Model()
            x = self.interface.Variable('x', lb=0, ub=10)
            y = self.interface.Variable('y', lb=0, ub=10)
            k = self.interface.Variable('k', type='binary')
            i = self.interface.Variable('i', type='binary')
            constr1 = self.interface.Constraint(1. * x + y, lb=3, name="constr1")
            constr2 = self.interface.Constraint(x - k, ub=-10, name="constr2")
            constr3 = self.interface.Constraint(y - i, ub=-10, name="constr3")
            obj = self.interface.Objective(2 * x + y)
            model.add(x)
            model.add(y)
            model.add(k)
            model.add(i)
            model.add(constr1)
            model.add(constr2)
            model.add(constr3)
            model.objective = obj
            model.optimize()
            self.model = model
            self.continuous_var = x
            self.binary_var = k
            self.constraint = constr1

        def test_infeasible(self):
            self.assertEqual(self.model.status, INFEASIBLE)

        def test_objective_value(self):
            with self.assertRaises(SolverError) as context:
                self.model.objective.value
            self.assertIn("CPLEX Error  1217", str(context.exception))

        def test_variable_primal(self):
            with self.assertRaises(SolverError) as context:
                self.continuous_var.primal
            self.assertIn("CPLEX Error  1217", str(context.exception))

        def test_binary_variable_primal(self):
            with self.assertRaises(SolverError) as context:
                self.binary_var.primal
            self.assertIn("CPLEX Error  1217", str(context.exception))

        def test_constraint_primal(self):
            with self.assertRaises(SolverError) as context:
                self.constraint.primal
            self.assertIn("CPLEX Error  1217", str(context.exception))

        def test_primal_values(self):
            with self.assertRaises(SolverError) as context:
                self.model.primal_values
            self.assertIn("CPLEX Error  1217", str(context.exception))

        def test_constraint_values(self):
            with self.assertRaises(SolverError) as context:
                self.model.constraint_values
            self.assertIn("CPLEX Error  1217", str(context.exception))


    class UnsolvedTestCase(unittest.TestCase):

        interface = cplex_interface

        def setUp(self):
            model = self.interface.Model()
            x = self.interface.Variable('x', lb=0, ub=10)
            constr1 = self.interface.Constraint(1. * x, lb=3, name="constr1")
            obj = self.interface.Objective(2 * x)
            model.add(x)
            model.add(constr1)
            model.objective = obj
            self.model = model
            self.continuous_var = x
            self.constraint = constr1

        def test_status(self):
            self.assertIs(self.model.status, None)

        def test_objective_value(self):
            with self.assertRaises(SolverError) as context:
                self.model.objective.value
            self.assertIn("CPLEX Error  1217", str(context.exception))

        def test_variable_primal(self):
            with self.assertRaises(SolverError) as context:
                self.continuous_var.primal
            self.assertIn("CPLEX Error  1217", str(context.exception))

        def test_variable_dual(self):
            with self.assertRaises(SolverError) as context:
                self.continuous_var.dual
            self.assertIn("CPLEX Error  1217", str(context.exception))

        def test_constraint_primal(self):
            with self.assertRaises(SolverError) as context:
                self.constraint.primal
            self.assertIn("CPLEX Error  1217", str(context.exception))

        def test_constraint_dual(self):
            with self.assertRaises(SolverError) as context:
                self.constraint.dual
            self.assertIn("CPLEX Error  1217", str(context.exception))

        def test_primal_values(self):
            with self.assertRaises(SolverError) as context:
                self.model.primal_values
            self.assertIn("CPLEX Error  1217", str(context.exception))

        def test_constraint_values(self):
            with self.assertRaises(SolverError) as context:
                self.model.constraint_values
            self.assertIn("CPLEX Error  1217", str(context.exception))

        def test_reduced_costs(self):
            with self.assertRaises(SolverError) as context:
                self.model.reduced_costs
            self.assertIn("CPLEX Error  1217", str(context.exception))

        def test_shadow_prices(self):
            with self.assertRaises(SolverError) as context:
                self.model.shadow_prices
            self.assertIn("CPLEX Error  1217", str(context.exception))
