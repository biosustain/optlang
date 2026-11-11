# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

import copy
import pickle
import sys
from unittest import TestCase

from optlang.exceptions import ContainerAlreadyContains
from optlang.interface import Constraint, Model, Objective, Variable


class TestModel(TestCase):
    def setUp(self):
        self.model = Model()
        x = Variable('x', lb=0, ub=10)
        y = Variable('y', lb=0, ub=10)
        constr = Constraint(1. * x + y, lb=3, name="constr1")
        obj = Objective(2 * x + y)
        self.model.add(x)
        self.model.add(y)
        self.model.add(constr)
        self.model.objective = obj

    def test_read_only_attributes(self):
        self.assertRaises(AttributeError, setattr, self.model, 'variables', 'Foo')
        self.assertRaises(AttributeError, setattr, self.model, 'constraints', 'Foo')
        self.assertRaises(AttributeError, setattr, self.model, 'status', 'Foo')

    def test_pickle_and_copy(self):
        for model in (pickle.loads(pickle.dumps(self.model)), copy.copy(self.model), copy.deepcopy(self.model)):
            self.assertEqual(list(model.variables.keys()), ["x", "y"])
            self.assertEqual(list(model.constraints.keys()), ["constr1"])

    def test_create_empty_model(self):
        model = Model()
        self.assertEqual(len(model.constraints), 0)
        self.assertEqual(len(model.variables), 0)
        self.assertEqual(model.objective.expression - 0, 0)

    def test_add_variable(self):
        var = Variable('z')
        self.model.add(var)
        self.assertTrue(var in self.model.variables.values())
        self.assertEqual(self.model.variables.values().count(var), 1)
        self.assertEqual(self.model.variables['z'].problem, var.problem)
        var = Variable('asdfasdflkjasdflkjasdlkjsadflkjasdflkjasdlfkjasdlfkjasdlkfjasdf', lb=-13)
        self.model.add(var)
        self.assertTrue(var in self.model.variables.values())
        self.assertEqual(self.model.variables['z'].lb, None)
        self.assertEqual(self.model.variables['z'].ub, None)
        self.assertEqual(self.model.variables['asdfasdflkjasdflkjasdlkjsadflkjasdflkjasdlfkjasdlfkjasdlkfjasdf'].lb,
                         -13)
        self.assertEqual(self.model.variables['asdfasdflkjasdflkjasdlkjsadflkjasdflkjasdlfkjasdlfkjasdlkfjasdf'].ub,
                         None)

    def test_add_variable_twice_raises(self):
        var = Variable('x')
        self.model.add(var)
        self.assertRaises(ContainerAlreadyContains, self.model.update)

    def test_remove_add_variable(self):
        var = self.model.variables[0]
        self.model.remove(var)
        self.model.add(var)
        self.model.update()

    def test_remove_add_remove_variable(self):
        var = self.model.variables[0]
        self.model.remove(var)
        self.model.add(var)
        # self.assertRaises(ContainerAlreadyContains, self.model.remove, var)

    def test_add_existing_variable(self):
        var = self.model.variables[0]
        self.model.add(var)
        self.assertRaises(Exception, self.model.update)

    def test_remove_constraint(self):
        self.model.remove('constr1')
        self.assertEqual(list(self.model.constraints), [])

    def test_add_remove_constraint(self):
        c = Constraint(self.model.variables.x + self.model.variables.y, lb=10)
        self.model.add(c)
        self.assertEqual(list(self.model.constraints), [self.model.constraints['constr1'], c])
        self.model.remove(c)
        self.model.update()
        self.assertEqual(list(self.model.constraints), [self.model.constraints['constr1']])

    def test_add_remove_collection(self):
        c = Constraint(self.model.variables.x + self.model.variables.y, lb=10)
        c2 = Constraint(3. * self.model.variables.x + self.model.variables.y, lb=10)
        self.model.add([c, c2])
        self.assertEqual(list(self.model.constraints), [self.model.constraints['constr1'], c, c2])
        self.model.remove([c, 'constr1', c2])
        self.assertEqual(list(self.model.constraints), [])

    def test_removing_objective_raises(self):
        self.assertRaises(TypeError, self.model.remove, self.model.objective)

    def test_removing_crap_raises(self):
        self.assertRaises(TypeError, self.model.remove, dict)

    def test_remove_variable_str(self):
        var = self.model.variables['y']
        self.model.remove(var.name)
        self.assertNotIn(var, self.model.variables.values())
        self.assertEqual(var.problem, None)
        self.assertEqual(self.model.objective.direction, "max")
        self.assertEqual(
            (self.model.objective.expression - (2.0 * self.model.variables.x)).expand() - 0, 0
        )
        self.assertEqual(len(self.model.variables), 1)
        self.assertEqual(len(self.model.constraints), 1)
        self.assertEqual((self.model.variables[0].lb, self.model.variables[0].ub), (0, 10))
        self.assertEqual(self.model.constraints[0].lb, 3)
        self.assertEqual(
            (self.model.constraints[0].expression - (1.0 * self.model.variables.x)).expand() - 0, 0
        )
        # self.assertEqual(self.model.__str__(), 'Maximize\n2.0*x\nsubject to\nconstr1: 3 <= 1.0*x\nBounds\n0 <= x <= 10')

    def test_number_objective(self):
        self.model.objective = Objective(0.)
        self.assertIn('Maximize\n0', self.model.objective.__str__())

    def test_add_differing_interface_type_raises(self):
        from optlang import glpk_interface as glpk
        x, y = glpk.Variable('x'), glpk.Variable('y')
        constraint = glpk.Constraint(x + y)
        objective = glpk.Objective(1. * x)
        self.assertRaises(TypeError, self.model.add, x)
        self.assertRaises(TypeError, self.model.add, constraint)
        self.assertRaises(TypeError, self.model.add, objective)

    def test_setting_binary_bounds(self):
        self.model.variables['x'].type = 'binary'
        self.assertEqual(self.model.variables['x'].lb, 0)
        self.assertEqual(self.model.variables['x'].ub, 1)

    def test_non_integer_bound_on_integer_variable_raises(self):
        self.model.variables['x'].type = 'integer'
        self.assertRaises(ValueError, setattr, self.model.variables['x'], 'lb', 3.3)

    def test_non_0_or_1_bound_on_binary_variable_raises(self):
        self.model.variables['x'].type = 'integer'
        self.assertRaises(ValueError, setattr, self.model.variables['x'], 'lb', 3.3)

    def test_false_objective_direction_raises(self):
        self.assertRaises(ValueError, setattr, self.model.objective, 'direction', 'neither_min_nor_max')

    def test_all_entities_point_to_correct_model(self):
        for variable in self.model.variables.values():
            self.assertEqual(variable.problem, self.model)
        for constraint in self.model.constraints:
            self.assertEqual(constraint.problem, self.model)
        self.assertEqual(self.model.objective.problem, self.model)

    def test_variable_independence(self):
        model = Model()
        x = Variable('x', lb=0, ub=20)
        self.assertNotEqual(id(x), id(self.model.variables['x']))
        y = Variable('y', lb=0, ub=10)
        constr = Constraint(1. * x + y, lb=3, name="constr1")
        model.add(constr)
        self.assertNotEqual(id(self.model.variables['x']), id(model.variables['x']))
        self.assertNotEqual(id(self.model.variables['y']), id(model.variables['y']))
        self.assertNotEqual(self.model.variables['y'].problem, model)
        self.assertNotEqual(self.model.variables['x'].problem, model)
        x.lb = -10
        self.assertNotEqual(self.model.variables['x'].lb, model.variables['x'].lb)

    def test_primal_and_dual_values(self):
        model = self.model
        self.assertTrue(all([primal is None for primal in model.primal_values.values()]))
        self.assertTrue(all([constraint_primal is None for constraint_primal in model.constraint_values.values()]))
        self.assertTrue(all([sp is None for sp in model.shadow_prices.values()]))
        self.assertTrue(all([rc is None for rc in model.reduced_costs.values()]))

    def test_interface(self):
        self.assertEqual(self.model.interface, sys.modules["optlang.interface"])


class TestVariable(TestCase):
    def setUp(self):
        self.x = Variable("x")

    def test_init_variable(self):
        self.assertRaises(ValueError, Variable, '')

    def test_set_wrong_bounds_on_binary_raises(self):
        self.assertRaises(ValueError, Variable, 'x', lb=-33, ub=0.3, type='binary')
        x = Variable('x', type='binary')
        self.assertRaises(ValueError, setattr, x, 'lb', -3)
        self.assertRaises(ValueError, setattr, x, 'ub', 3)

    def test_set_wrong_bounds_on_integer_raises(self):
        self.assertRaises(ValueError, Variable, 'x', lb=-33, ub=0.3, type='integer')
        x = Variable('x', type='integer')
        self.assertRaises(ValueError, setattr, x, 'lb', -3.3)
        self.assertRaises(ValueError, setattr, x, 'ub', 3.3)

    def test_primal_and_dual(self):
        x = self.x
        self.assertTrue(x.primal is None)
        self.assertTrue(x.dual is None)

    def test_change_name(self):
        x = self.x
        x.name = "xx"
        self.assertEqual(x.name, "xx")


class TestConstraint(TestCase):
    def setUp(self):
        self.a = Variable("a")
        self.b = Variable("b")
        self.c = Variable("c")
        self.d = Variable("d")

    def test_is_linear_and_quadratic(self):
        c1 = Constraint(self.a)
        self.assertTrue(c1.is_Linear)
        self.assertFalse(c1.is_Quadratic)
        c2 = Constraint(self.a * self.b ** self.c)
        self.assertFalse(c2.is_Linear)
        self.assertFalse(c2.is_Quadratic)
        c3 = Constraint(self.a * self.b * self.c + self.d)
        self.assertFalse(c3.is_Linear)
        self.assertFalse(c3.is_Quadratic)
        c4 = Constraint(self.a + self.b ** 3)
        self.assertFalse(c4.is_Linear)
        self.assertFalse(c4.is_Quadratic)
