import pylp
import unittest
import logging

logging.basicConfig(level=logging.INFO)


class TestSolvers(unittest.TestCase):

    def simple_solver_test(self, preference):

        num_vars = 10
        special_var = 5

        solver = pylp.LinearSolver(
            num_vars,
            pylp.VariableType.Binary,
            {
                special_var: pylp.VariableType.Continuous
            },
            preference)

        objective = pylp.LinearObjective()
        for i in range(num_vars):
            objective.set_coefficient(i, 1.0)
        objective.set_coefficient(special_var, 0.5)

        constraint = pylp.LinearConstraint()
        for i in range(num_vars):
            constraint.set_coefficient(i, 1.0)
        constraint.set_relation(pylp.Relation.Equal)
        constraint.set_value(1.0)

        solver.set_objective(objective)
        solver.add_constraint(constraint)

        solution, msg = solver.solve()

        self.assertEqual(solution[5], 1)

    def test_any(self):
        self.simple_solver_test(pylp.Preference.Any)

    def test_gurobi(self):
        self.simple_solver_test(pylp.Preference.Gurobi)
        
if __name__ == '__main__':
    obj = TestSolvers()
    obj.test_gurobi()
