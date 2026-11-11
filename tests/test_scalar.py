import unittest

from sympy import Rational

from relu_rlct import certify_scalar_relu, solve_shallow_relu


class ScalarCertificateTests(unittest.TestCase):
    def test_agrees_with_polynomial_solver_for_dependent_gate_columns(self):
        X = [[-2], [-1], [1], [2]]
        incoming = [[1, 0], [2, 0], [-1, 0]]
        outgoing = [[1], [-2], [3]]
        result = certify_scalar_relu(X, incoming, outgoing)
        general = solve_shallow_relu(X, incoming, outgoing)
        self.assertEqual((result.status, result.rlct, result.multiplicity),
                         ("certified", Rational(2), 1))
        self.assertEqual((result.rlct, result.multiplicity), (general.rlct, general.multiplicity))

    def test_checks_chart_hypotheses(self):
        for X, outgoing in (([[0], [1]], [[1]]), ([[-1], [1]], [[0]])):
            with self.subTest(X=X, outgoing=outgoing):
                result = certify_scalar_relu(X, [[1, 0]], outgoing)
                self.assertEqual(result.status, "unsupported")
                self.assertIsNone(result.rlct)
        with self.assertRaises(ValueError):
            certify_scalar_relu([[1.0]], [[1, 0]], [[1]])
