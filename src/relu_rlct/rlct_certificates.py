"""Conservative exact LS certificates for centered polynomial residual maps.

See finite_data_rlct_decision.md. This is a partial solver, not a universal
RLCT package. An unresolved result retains the exact implicit LS equations;
a short vanishing jet is never reported as an identically zero core.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

import sympy as s

if TYPE_CHECKING:
    from .rlct_decision import DecisionReport


@dataclass
class LSCertificate:
    status: str
    method: str
    jacobian_rank: int
    kernel_dimension: int
    rlct: s.Rational | None = None
    multiplicity: int | None = None
    kernel_variables: tuple[s.Symbol, ...] = ()
    regular_variables: tuple[s.Symbol, ...] = ()
    source_change: s.Matrix | None = None
    regular_indices: tuple[int, ...] = ()
    remaining_indices: tuple[int, ...] = ()
    implicit_regular_equations: tuple[s.Expr, ...] = ()
    implicit_remaining_residuals: tuple[s.Expr, ...] = ()
    core_quadratics: tuple[s.Expr, ...] = ()
    definite_combination: tuple[s.Rational, ...] = ()
    core_order: int | None = None
    core_coefficient: s.Rational | None = None
    checked_order: int | None = None
    termination_bound: int | None = None
    transverse_rank: int | None = None
    splitting_order: int | None = None
    splitting_coefficient: s.Rational | None = None
    splitting_remainder_zero: bool | None = None
    critical_curve_certificate: LSCertificate | None = None
    critical_membership: object | None = None
    critical_ideal_variables: tuple[s.Symbol, ...] = ()
    critical_ideal_generators: tuple[s.Expr, ...] = ()
    critical_value_polynomial: s.Expr | None = None
    explanation: str = ""


def _definite_sign(matrix: s.Matrix) -> int:
    """Exact Sylvester criterion; no numerical eigenvalue tolerance."""
    minors = [matrix[:j, :j].det() for j in range(1, matrix.rows + 1)]
    if all(value > 0 for value in minors):
        return 1
    if all((-1)**j * value > 0 for j, value in enumerate(minors, 1)):
        return -1
    return 0


def certify_polynomial_map(
    residuals, variables, *, max_corank_one_order: int = 8,
    quadratic_combination=None, max_critical_membership_variables=10, _degree_limit=None,
) -> LSCertificate:
    """Analyze an exact rational, centered polynomial residual map.

    The corank here is the dimension of the source Jacobian kernel, not the
    number of unused target equations. A supplied quadratic_combination has
    one coefficient per remaining residual after the reported row selection.
    Its definiteness is checked, never trusted. Without it we try individual
    quadratics and, for at most four generators, their signed sums.

    max_corank_one_order limits practical computation. It is NOT a
    determinacy bound. For a one-dimensional kernel the proven worst-case
    bound is the product of the selected regular equations' degrees and the
    largest remaining degree, inclusive. A cap below it can return unresolved.
    For quadratic maps it is at most 2**len(variables).
    """
    variables = tuple(variables)
    residuals = tuple(s.expand(f) for f in residuals)
    if not variables:
        if any(f != 0 or f.atoms(s.Float) for f in residuals):
            raise ValueError("The only centered ideal without variables is the exact zero ideal")
        return LSCertificate("certified", "zero-ideal", 0, 0, s.Rational(0), 1,
                             explanation="A variable-free zero core contributes no exponent or logarithm.")
    if len(set(variables)) != len(variables):
        raise ValueError("Supply distinct coordinate symbols")
    if any(not isinstance(x, s.Symbol) for x in variables):
        raise ValueError("Coordinates must be symbols")
    if not isinstance(max_corank_one_order, int) or max_corank_one_order < 2:
        raise ValueError("max_corank_one_order must be an integer at least two")
    if (isinstance(max_critical_membership_variables, bool)
        or not isinstance(max_critical_membership_variables, int) or max_critical_membership_variables < 0):
        raise ValueError("max_critical_membership_variables must be a nonnegative integer")
    origin = dict.fromkeys(variables, 0)
    degrees = []
    for f in residuals:
        if f.atoms(s.Float):
            raise ValueError("Residuals must use exact rational coefficients, not floats")
        try:
            polynomial = s.Poly(f, *variables, domain=s.QQ)
        except (s.PolynomialError, s.polys.polyerrors.CoercionFailed) as exc:
            raise ValueError("Residuals must be rational polynomials") from exc
        if _degree_limit is not None and polynomial.total_degree() > _degree_limit:
            raise ValueError("Residual degree exceeds the requested degree limit")
        if f.subs(origin) != 0:
            raise ValueError("Residuals must be centered")
        degrees.append(polynomial.total_degree())
    dimension = len(variables)
    F = s.Matrix(len(residuals), 1, residuals)
    J = F.jacobian(variables).subs(origin)
    regular_indices = tuple(J.T.rref()[1])
    rho = len(regular_indices)
    k = dimension - rho
    remaining_indices = tuple(i for i in range(len(residuals)) if i not in regular_indices)
    result = LSCertificate(
        "unresolved", "implicit-ls", rho, k,
        regular_indices=regular_indices, remaining_indices=remaining_indices,
    )

    def solved(method, value, multiplicity=1, explanation=""):
        result.status = "certified"
        result.method = method
        result.rlct = s.Rational(value)
        result.multiplicity = multiplicity
        result.explanation = explanation
        return result

    if k == 0:
        return solved("injective-jacobian", s.Rational(rho, 2))
    if not remaining_indices:
        return solved("submersion", s.Rational(rho, 2), explanation=
                      "All residuals are independent regular coordinates; other source variables are free.")

    JI = J.extract(regular_indices, range(dimension))
    source_pivots = tuple(JI.rref()[1])
    C = s.eye(dimension)[:, list(source_pivots)]
    K = s.Matrix.hstack(*J.nullspace())
    A = JI * C
    Ainv = A.inv() if rho else s.zeros(0, 0)
    # Dummy symbols cannot collide with any caller's source coordinates.
    y = tuple(s.Dummy(f"ls_y{i}") for i in range(rho))
    t = tuple(s.Dummy(f"ls_t{i}") for i in range(k))
    result.regular_variables, result.kernel_variables = y, t
    result.source_change = C.row_join(K)
    kernel_point = K * s.Matrix(t)
    Q = F.subs(dict(zip(variables, kernel_point)), simultaneous=True).applyfunc(s.expand)
    if not any(Q):
        return solved("exact-linear-fiber", s.Rational(rho, 2), explanation=
                      "The entire linear Jacobian kernel is an exact fiber; IFT uniqueness makes the LS core zero.")
    # In a polynomial presentation the exact kernel restriction need not be
    # quadratic. Use only its homogeneous degree-two part for Hessian tests.
    Q = Q.applyfunc(lambda expression: sum(
        coefficient*s.prod(v**power for v, power in zip(t, exponent))
        for exponent, coefficient in s.Poly(expression, *t).terms() if sum(exponent) == 2))
    QI = Q.extract(regular_indices, [0])
    JO = J.extract(remaining_indices, range(dimension))
    QO = Q.extract(remaining_indices, [0])
    g2 = (QO - JO * C * Ainv * QI).applyfunc(s.expand)
    result.core_quadratics = tuple(g2)

    # Keep the full equations defining the unique IFT branch, not just its jet.
    source_point = C * s.Matrix(rho, 1, y) + kernel_point
    transformed = F.subs(dict(zip(variables, source_point)), simultaneous=True).applyfunc(s.expand)
    result.implicit_regular_equations = tuple(transformed[i] for i in regular_indices)
    result.implicit_remaining_residuals = tuple(transformed[i] for i in remaining_indices)

    if k == 1:
        bound = max(2, int(s.prod(degrees[i] for i in regular_indices)
                           * max(degrees[i] for i in remaining_indices)))
        result.termination_bound = bound
        cap = min(max_corank_one_order, bound)
        parameter = t[0]
        phi = s.zeros(rho, 1)
        regular_polys = [s.Poly(transformed[i], *y, parameter) for i in regular_indices]
        remaining_polys = [s.Poly(transformed[i], *y, parameter) for i in remaining_indices]

        def coefficient(polynomial, order):
            value = polynomial.as_expr().subs(dict(zip(y, phi)), simultaneous=True)
            return s.expand(value).coeff(parameter, order)

        for order in range(2, cap + 1):
            rhs = s.Matrix(rho, 1, [coefficient(f, order) for f in regular_polys])
            phi -= (Ainv * rhs) * parameter**order
            coefficients = [coefficient(f, order) for f in remaining_polys]
            if any(value != 0 for value in coefficients):
                result.core_order = order
                result.core_coefficient = next(value for value in coefficients if value != 0)
                result.checked_order = order
                return solved("one-dimensional-core", s.Rational(rho, 2) + s.Rational(1, 2*order),
                              explanation="Exact implicit coefficients find the first nonzero core order.")
        result.checked_order = cap
        if cap == bound:
            return solved("bezout-zero-core", s.Rational(rho, 2), explanation=
                          "Every core coefficient through the inclusive local Bezout bound vanishes.")
        result.explanation = (f"Core vanishes through order {cap}, below the proven bound {bound}; "
                              "no conclusion about higher orders or the RLCT is claimed.")
        return result

    hessians = [s.hessian(f, t) for f in g2]
    if quadratic_combination is not None:
        witness = tuple(s.sympify(a) for a in quadratic_combination)
        if len(witness) != len(hessians) or any(a.is_Rational is not True for a in witness):
            raise ValueError("Quadratic combination must contain one rational coefficient per remaining residual")
        candidates = [witness]
    else:
        candidates = [tuple(int(i == j) for i in range(len(hessians))) for j in range(len(hessians))]
        if len(hessians) <= 4:
            candidates += list(product((-1, 0, 1), repeat=len(hessians)))
    for witness in candidates:
        combined = sum((a*H for a, H in zip(witness, hessians)), s.zeros(k))
        if _definite_sign(combined):
            result.definite_combination = tuple(map(s.Rational, witness))
            return solved("definite-quadratic-combination", s.Rational(rho, 2) + s.Rational(k, 4),
                          explanation="A verified definite quadratic combination gives loss comparable to ||t||^4.")
    if len(remaining_indices) == 1 and hessians[0].det() != 0:
        if _definite_sign(hessians[0]):
            result.definite_combination = (s.Rational(1),)
            return solved("definite-quadratic-combination", s.Rational(rho, 2) + s.Rational(k, 4))
        # Definite cases were handled above. The real analytic Morse lemma
        # applies to the sole FULL residual, not merely a principal two-jet.
        return solved("principal-indefinite-morse", s.Rational(rho + 1, 2),
                      2 if k == 2 else 1,
                      explanation="One full LS residual has a nonsingular indefinite Hessian.")
    if len(remaining_indices) == 1:
        H = hessians[0]
        pivots = tuple(H.rref()[1])
        transverse_rank = len(pivots)
        result.transverse_rank = transverse_rank
        block = H.extract(pivots, pivots)
        sign = _definite_sign(block) if transverse_rank else 0
        if transverse_rank >= 3 and sign == 0:
            return solved("principal-indefinite-block", s.Rational(rho + 1, 2),
                          explanation="An indefinite Hessian block of rank at least three controls every analytic remainder.")
        if transverse_rank == k-1 or (transverse_rank == 2 and sign == 0):
            # Construct the polynomial constrained critical curve of the
            # exact scalar LS residual. These auxiliary equations locate the
            # splitting remainder; their RLCT is NOT the network's RLCT.
            transverse = tuple(s.Dummy(f"split_y{i}") for i in range(transverse_rank))
            parameters = tuple(s.Dummy(f"split_z{i}") for i in range(k-transverse_rank))
            multipliers = tuple(s.Dummy(f"split_mu{i}") for i in range(rho))
            K_hessian = s.Matrix.hstack(*H.nullspace())
            tangent_point = s.eye(k)[:, list(pivots)]*s.Matrix(transverse) + K_hessian*s.Matrix(parameters)
            substitution = dict(zip(t, tangent_point))
            f = s.Matrix(rho, 1, result.implicit_regular_equations)
            p = transformed[remaining_indices[0]]
            # Kill the complete first derivative by a constant residual-row
            # operation. The teacher multipliers are now exactly zero.
            p -= (JO*C*Ainv*f)[0] if rho else 0
            f = f.subs(substitution, simultaneous=True).applyfunc(s.expand)
            p = s.expand(p.subs(substitution, simultaneous=True))
            differentiated = y+transverse
            stationary = [s.expand(s.diff(p, variable)-sum(
                multipliers[a]*s.diff(f[a], variable) for a in range(rho)))
                for variable in differentiated]
            augmented_variables = y+transverse+multipliers+parameters
            critical_equations = tuple(f)+tuple(stationary)
            result.critical_ideal_variables = augmented_variables
            result.critical_ideal_generators = critical_equations
            result.critical_value_polynomial = p
            if len(parameters) > 1:
                if len(augmented_variables) > max_critical_membership_variables:
                    result.explanation = ("The rank-two indefinite exponent is rho/2+1/2, but the full pair "
                                          "is unresolved: critical-ideal membership exceeds the variable cap.")
                    return result
                from .rlct_ideal_reduction import local_ideal_membership
                membership = local_ideal_membership(p, critical_equations, augmented_variables)
                result.critical_membership = membership
                result.splitting_remainder_zero = membership.member
                return solved("principal-rank-two-local-membership", s.Rational(rho+1, 2),
                              2 if membership.member else 1,
                              "The exact local critical-ideal membership test decides the logarithmic factor.")
            curve = certify_polynomial_map(tuple(f)+tuple(stationary)+(p,), augmented_variables,
                                            max_corank_one_order=max_corank_one_order)
            result.critical_curve_certificate = curve
            if curve.kernel_dimension != 1:
                raise AssertionError("Constrained critical curve must have one-dimensional derivative kernel")
            if curve.status != "certified":
                result.explanation = ("Principal one-null-direction core; its critical remainder has not been "
                                      "resolved within the chosen implicit-order cap. " + curve.explanation)
                return result
            order, coefficient_value = curve.core_order, curve.core_coefficient
            result.splitting_order = order
            result.splitting_coefficient = coefficient_value
            result.splitting_remainder_zero = order is None
            if order is None:
                contribution = s.Rational(transverse_rank, 4) if sign else s.Rational(1, 2)
                multiplicity = 2 if sign == 0 and transverse_rank == 2 else 1
            else:
                if order < 3:
                    raise AssertionError("A Hessian-null splitting remainder cannot have order below three")
                multiplicity = 1
                if sign == 0:
                    contribution = s.Rational(1, 2)
                else:
                    radial = s.Rational(transverse_rank, 4)+s.Rational(1, 2*order)
                    same_sign_even = order % 2 == 0 and sign*coefficient_value > 0
                    contribution = radial if same_sign_even else min(s.Rational(1, 2), radial)
            return solved("principal-one-null-direction", s.Rational(rho, 2)+contribution,
                          multiplicity, "Exact constrained critical-curve order and real Morse splitting.")
    result.explanation = ("No implemented exact certificate applies. The implicit equations and two-jet are retained; "
                          "the two-jet is not asserted to be the full LS ideal.")
    return result


def certify_quadratic_map(residuals, variables, **kwargs) -> LSCertificate:
    """Backward-compatible degree-at-most-two entry point with exact validation."""
    return certify_polynomial_map(residuals, variables, _degree_limit=2, **kwargs)


def certify_compiled(report: DecisionReport, **kwargs) -> LSCertificate:
    """Apply the exact certificates to a supported finite-data compiler report."""
    if report.status == "unsupported":
        raise ValueError("Unsupported path chart: " + "; ".join(report.unsupported_reasons))
    return certify_quadratic_map(report.residuals, report.variables, **kwargs)


def analyze_shallow_relu(X, incoming, outgoing, output_bias=None, *,
                         build_transfer=False, **kwargs):
    """Return (exact compiler report, conservative LS/RLCT certificate)."""
    from .rlct_decision import compile_shallow_relu
    report = compile_shallow_relu(X, incoming, outgoing, output_bias,
                                  build_transfer=build_transfer)
    certificate = None if report.status == "unsupported" else certify_compiled(report, **kwargs)
    return report, certificate
