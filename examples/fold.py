"""Certify a biased two-output fold from the finite samples and weights."""

from sympy import Matrix, Rational

from relu_rlct import solve_shallow_relu


def main():
    groups = [(-2, -1, 0), (Rational(5, 2), 3, Rational(7, 2)), (6, 7, 8)]
    points = [(x, 1) for group in groups for x in group]
    points.extend((2 * middle, 2) for _, middle, _ in groups)
    incoming = Matrix([[1, -1, Rational(1, 10)], [-1, 2, Rational(2, 10)],
                       [1, -4, Rational(3, 10)], [-1, 5, Rational(4, 10)]])
    outgoing = Matrix([[2, 1], [1, 1], [5, 1], [4, 1]])
    result = solve_shallow_relu(points, incoming, outgoing)
    assert (result.status, result.rlct, result.multiplicity) == ("certified", Rational(35, 4), 1)
    print(f"RLCT = {result.rlct}, multiplicity = {result.multiplicity}")
    print(f"Residual Jacobian rank = {result.report.rho}")
    print(result.explanation)


if __name__ == "__main__":
    main()
