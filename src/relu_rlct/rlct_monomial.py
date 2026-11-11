"""Exact squared-loss RLCT and pole multiplicity of a monomial ideal.

Uses the Newton linear program, enumerating rational vertices only in a
bounded-size problem. This never replaces a polynomial by its leading term.
"""

from dataclasses import dataclass
from itertools import combinations
from math import comb

import sympy as s


@dataclass(frozen=True)
class MonomialCertificate:
    status: str
    rlct: s.Rational | None
    multiplicity: int | None
    exponents: tuple[tuple[int, ...], ...] = ()
    optimal_vertices: tuple[tuple[s.Rational, ...], ...] = ()
    explanation: str = ""


def certify_monomial_ideal(generators, variables, *, max_vertex_systems=20000):
    """For I=<x^alpha>, minimize sum(w)/2 with alpha.w>=1, w>=0.

    The multiplicity is one plus the affine dimension of the optimal face.
    Free variables are removed before enumerating candidate vertices. Exceeding
    the computational cap gives unresolved, not an approximate threshold.
    """
    variables = tuple(variables)
    if isinstance(max_vertex_systems, bool) or not isinstance(max_vertex_systems, int) or max_vertex_systems < 0:
        raise ValueError("max_vertex_systems must be a nonnegative integer")
    if len(set(variables)) != len(variables) or any(not isinstance(v, s.Symbol) for v in variables):
        raise ValueError("Coordinates must be distinct symbols")
    generators = tuple(s.expand(g) for g in generators if s.sympify(g) != 0)
    if not generators:
        return MonomialCertificate("certified", s.Rational(0), 1, explanation="Zero ideal contributes no exponent or logarithm.")
    if not variables:
        raise ValueError("A nonzero centered ideal requires variables")
    exponents = []
    for g in generators:
        if g.atoms(s.Float):
            raise ValueError("Exact rational coefficients required")
        try:
            polynomial = s.Poly(g, *variables, domain=s.QQ)
        except (s.PolynomialError, s.polys.polyerrors.CoercionFailed) as exc:
            raise ValueError("Exact rational polynomials required") from exc
        if polynomial.eval(dict.fromkeys(variables, 0)) != 0:
            raise ValueError("Generators must vanish at the origin")
        if len(polynomial.terms()) != 1:
            return MonomialCertificate("unresolved", None, None, explanation="The supplied full ideal generators are not monomials.")
        exponents.append(tuple(polynomial.monoms()[0]))
    # Delete divisibility-redundant monomial generators, not their powers.
    unique = sorted(set(exponents))
    unique = [a for a in unique if not any(b != a and all(x <= y for x, y in zip(b, a)) for b in unique)]
    active = [i for i in range(len(variables)) if any(a[i] for a in unique)]
    compact = [tuple(a[i] for i in active) for a in unique]
    n = len(active)
    constraints = compact + [tuple(int(i == j) for i in range(n)) for j in range(n)]
    constants = [s.Integer(1)]*len(compact)+[s.Integer(0)]*n
    systems = comb(len(constraints), n)
    if systems > max_vertex_systems:
        return MonomialCertificate("unresolved", None, None, tuple(unique), explanation=
                                   f"Exact vertex enumeration needs {systems} systems, above cap {max_vertex_systems}.")
    A = s.Matrix(constraints)
    b = s.Matrix(constants)
    vertices = set()
    for rows in combinations(range(len(constraints)), n):
        square = A[list(rows), :]
        if square.det() == 0:
            continue
        vertex = square.inv()*b[list(rows), :]
        if all(value >= 0 for value in A*vertex-b):
            vertices.add(tuple(vertex))
    if not vertices:
        raise AssertionError("Nonzero centered monomial ideal must have a feasible Newton vertex")
    optimum = min(sum(v) for v in vertices)
    best = sorted(v for v in vertices if sum(v) == optimum)
    dimension = s.Matrix([list(s.Matrix(v)-s.Matrix(best[0])) for v in best[1:]]).rank() if len(best) > 1 else 0
    expanded_vertices = tuple(tuple(v[active.index(i)] if i in active else s.Integer(0)
                                     for i in range(len(variables))) for v in best)
    return MonomialCertificate("certified", optimum/2, dimension+1, tuple(unique), expanded_vertices,
                               "Exact Newton optimum and optimal-face dimension; squared-residual convention.")
