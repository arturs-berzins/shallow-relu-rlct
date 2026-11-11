"""Conservative exact polynomial-ideal preprocessing for local RLCT work.

The input is a centered ideal in a rational polynomial ring. Reduced Groebner
bases preserve the *ideal*, including nonreduced contact. Every affine-linear
generator is then made a regular source coordinate by an invertible rational
linear change. Substitution into the other generators is legitimate because
the eliminated regular coordinate remains an explicit ideal generator.

The result records the exact equivalence (of polynomial ideals if no local
unit step is used, and otherwise of local analytic ideals at the origin)

    I = <regular_coordinates, generators(remaining_variables)>

in the returned source chart. Thus the squared-loss RLCT gains
``regular_count/2`` when an independent method certifies the remaining ideal.
Optional local preprocessing also strips polynomial factors nonzero at the
origin and removes a generator only with an exact ideal-membership witness
after multiplication by such a unit. These steps preserve the local analytic
ideal, not necessarily the global polynomial ideal, and are recorded
separately. No radical or nonlinear implicit solution is computed here. A
smooth nonlinear equation may remain: this is a preprocessing layer, not
complete Lyapunov--Schmidt elimination. Caps preserve a valid partial
reduction. They are not timeouts for individual Groebner computations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import sympy as s


ReductionStatus = Literal["reduced", "zero", "capped"]


@dataclass(frozen=True)
class AffineEliminationStep:
    """One exact source change; all expressions use this step's coordinates.

    ``generator = coefficient * (eliminated_variable - replacement)``.
    The newly named regular coordinate is ``eliminated_variable-replacement``;
    its inverse substitution is ``eliminated_variable = inverse_expression``.
    The fresh SymPy Dummy never collides with an input variable, even if its
    printed name happens to be the same.
    """

    variables_before: tuple[s.Symbol, ...]
    groebner_generators: tuple[s.Expr, ...]
    generator: s.Expr
    eliminated_variable: s.Symbol
    coefficient: s.Rational
    replacement: s.Expr
    regular_coordinate: s.Dummy
    inverse_expression: s.Expr


@dataclass(frozen=True)
class LocalUnitStep:
    """An explicitly certified equivalence of local analytic ideals.

    For ``strip_factor``, ``generator = unit*replacement`` exactly, and the
    remaining generators are unchanged. For ``remove_redundant``, replacement
    is None and ``unit*generator`` belongs to the polynomial ideal generated
    by ``remaining_generators``. In either case ``unit(0) != 0`` is checked.
    Coordinates are those present when the step is performed.
    """

    kind: Literal["strip_factor", "remove_redundant"]
    variables: tuple[s.Symbol, ...]
    generator: s.Expr
    unit: s.Expr
    replacement: s.Expr | None
    remaining_generators: tuple[s.Expr, ...]


@dataclass(frozen=True)
class LocalIdealMembership:
    """Complete membership decision in the rational local ring at the origin.

    When ``member`` is true, ``unit`` is an exact polynomial nonzero at the
    origin such that unit times the tested polynomial belongs to the original
    polynomial ideal. False means the full colon ideal is contained in the
    origin's maximal ideal, not that a finite witness search happened to fail.
    """

    member: bool
    unit: s.Expr | None
    explanation: str


@dataclass(frozen=True)
class IdealReduction:
    """Exact, possibly partial, separation of affine regular coordinates.

    ``source_reconstruction`` is an ordered tuple of (original variable,
    expression) pairs. It expresses every input coordinate in the final
    remaining variables and the distinct regular coordinates from ``steps``.
    Setting those regular coordinates to zero gives the exact eliminated
    fiber substitution. The complete reconstruction is invertible, not merely
    a parametrization of the zero set.

    Empty ``generators`` means the zero residual ideal, including when no
    variables remain. A downstream squared-loss convention may assign this
    core exponent zero and multiplicity one; this module itself returns no
    exponent. ``reduced`` means no affine-linear Groebner generator remains,
    not that the ideal is radical, principal, or otherwise classified.
    """

    status: ReductionStatus
    regular_count: int
    remaining_variables: tuple[s.Symbol, ...]
    generators: tuple[s.Expr, ...]
    degrees: tuple[int, ...]
    steps: tuple[AffineEliminationStep, ...]
    local_steps: tuple[LocalUnitStep, ...]
    equivalence_scope: Literal["polynomial", "local_analytic"]
    source_reconstruction: tuple[tuple[s.Symbol, s.Expr], ...]
    explanation: str

    @property
    def regular_coordinates(self) -> tuple[s.Dummy, ...]:
        return tuple(step.regular_coordinate for step in self.steps)


def _validate(
    generators: Iterable[s.Expr], variables: Iterable[s.Symbol],
) -> tuple[tuple[s.Expr, ...], tuple[s.Symbol, ...]]:
    variables = tuple(variables)
    if any(not isinstance(variable, s.Symbol) for variable in variables):
        raise ValueError("variables must be SymPy Symbols")
    if len(set(variables)) != len(variables):
        raise ValueError("variables must be distinct")
    variable_set = set(variables)
    origin = dict.fromkeys(variables, s.S.Zero)
    validated = []
    for value in generators:
        try:
            expression = s.sympify(value)
        except (TypeError, ValueError, s.SympifyError) as error:
            raise ValueError("generators must be rational polynomials") from error
        if not isinstance(expression, s.Expr):
            raise ValueError("generators must be scalar rational polynomials")
        # QQ conversion can silently rationalize a floating-point coefficient.
        if expression.has(s.Float):
            raise ValueError("floating-point coefficients are unsupported; use exact rationals")
        if expression.free_symbols - variable_set:
            raise ValueError("every generator symbol must be listed in variables")
        try:
            if variables:
                polynomial = s.Poly(expression, *variables, domain=s.QQ)
                expression = polynomial.as_expr()
            elif expression.is_Rational is not True:
                raise ValueError("constant generators must be exact rational numbers")
        except (s.PolynomialError, s.CoercionFailed, s.GeneratorsNeeded) as error:
            raise ValueError("generators must be polynomials with exact rational coefficients") from error
        if expression.subs(origin) != 0:
            raise ValueError("generators must vanish at the origin")
        if expression != 0:
            validated.append(expression)
    return tuple(validated), variables


def local_ideal_membership(
    generator: s.Expr, generators: Iterable[s.Expr], variables: Iterable[s.Symbol],
) -> LocalIdealMembership:
    """Decide h in J localized at the origin, without taking any radical.

    Inputs must be centered polynomials with exact rational coefficients.
    Zero h always belongs, including with no variables; nonzero h does not
    belong to the zero ideal. This procedure has no computational cap and
    therefore returns a genuine membership decision, not an unresolved flag.

    For nonzero h, eliminate a fresh t from <t*J, (1-t)*h> to obtain generators
    of J intersect <h>. Their exact quotients by h generate J:h. Local
    membership holds precisely when this colon ideal contains a polynomial
    nonzero at the origin, equivalently when at least one of these generators
    has nonzero constant term. A positive witness is independently checked
    against a Groebner basis of J. Powers are never discarded.
    """
    generators, variables = _validate(generators, variables)
    target, _ = _validate((generator,), variables)
    if not target:
        return LocalIdealMembership(True, s.S.One, "The zero polynomial belongs to every ideal.")
    h = target[0]
    if not generators:
        return LocalIdealMembership(False, None, "A nonzero polynomial does not belong to the localized zero ideal.")
    origin = dict.fromkeys(variables, s.S.Zero)
    ideal_basis = s.groebner(generators, *variables, domain=s.QQ, order="lex")
    if ideal_basis.reduce(h)[1] == 0:
        return LocalIdealMembership(True, s.S.One, "Membership already holds in the polynomial ideal.")

    auxiliary = s.Dummy("ideal_intersection", real=True)
    intersection = s.groebner(
        tuple(auxiliary*expression for expression in generators)
        + ((1-auxiliary)*h,), auxiliary, *variables, domain=s.QQ, order="lex",
    )
    for polynomial in intersection.polys:
        expression = polynomial.as_expr()
        if expression.has(auxiliary):
            continue
        quotient, remainder = s.div(expression, h, *variables, domain=s.QQ)
        if remainder != 0:
            raise ArithmeticError("intersection generator is not divisible by h")
        if quotient.subs(origin) == 0:
            continue
        unit = s.expand(quotient)
        if ideal_basis.reduce(s.expand(unit*h))[1] != 0:
            raise ArithmeticError("invalid local membership certificate")
        return LocalIdealMembership(
            True, unit, "The exact colon ideal contains the displayed unit; its product with h was verified in J.",
        )
    return LocalIdealMembership(
        False, None, "Every generator of the exact colon ideal vanishes at the origin, so local membership fails.",
    )


def reduce_polynomial_ideal(
    generators: Iterable[s.Expr], variables: Iterable[s.Symbol], *,
    order: str = "lex", max_rounds: int | None = None,
    remove_local_redundancy: bool = True,
    max_local_generators: int | None = 8,
    max_local_variables: int | None = 6,
) -> IdealReduction:
    """Preserve a centered rational ideal while splitting affine generators.

    The supported monomial orders are ``lex``, ``grlex``, and ``grevlex``.
    Variable order is respected throughout; the first variable with nonzero
    coefficient in the first affine-linear basis generator is eliminated.
    Affine source changes divide only by nonzero rational constants. Optional
    local steps divide only by verified units at the origin. The number of
    affine steps is at most the original variable count even without a cap.

    A cap of zero still computes the initial Groebner basis. A ``capped``
    result contains its valid partial decomposition and does not assert that
    all available affine-linear generators have been removed. If no such
    generator remains, the result is ``reduced`` regardless of the cap.

    Local redundancy is tested by the exact colon ideal J:h: construct the
    intersection J intersect <h> by eliminating a fresh variable t from
    <t*J, (1-t)*h>, then divide its generators exactly by h. A colon generator
    nonzero at the origin is a unit witness; membership of its product with h
    in J is independently verified before deletion. The local size caps are
    applied only to this colon computation, after affine elimination. A
    skipped search is reported as ``capped``; all recorded equivalences still
    hold. Unit-factor stripping is not subject to those size caps.
    """
    if order not in ("lex", "grlex", "grevlex"):
        raise ValueError("order must be lex, grlex, or grevlex")
    if max_rounds is not None and (
        isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 0
    ):
        raise ValueError("max_rounds must be a nonnegative integer or None")
    if not isinstance(remove_local_redundancy, bool):
        raise ValueError("remove_local_redundancy must be a boolean")
    for name, cap in (("max_local_generators", max_local_generators),
                      ("max_local_variables", max_local_variables)):
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 0):
            raise ValueError(f"{name} must be a nonnegative integer or None")
    current_generators, current_variables = _validate(generators, variables)
    original_variables = current_variables
    reconstruction = {variable: variable for variable in original_variables}
    steps: list[AffineEliminationStep] = []
    local_steps: list[LocalUnitStep] = []

    while True:
        if not current_generators:
            status: ReductionStatus = "zero"
            explanation = "All remaining residual generators vanish identically; the residual core is zero."
            degrees: tuple[int, ...] = ()
            break
        # Centering and invertible affine elimination rule out a nonzero
        # constant. In particular, this branch always has at least one variable.
        basis = s.groebner(current_generators, *current_variables, order=order, domain=s.QQ)
        current_generators = tuple(polynomial.as_expr() for polynomial in basis.polys)
        degrees = tuple(int(polynomial.total_degree()) for polynomial in basis.polys)
        linear_index = next((index for index, degree in enumerate(degrees) if degree == 1), None)
        if linear_index is None:
            if remove_local_redundancy:
                changed = _strip_one_unit_factor(current_generators, current_variables)
                if changed is not None:
                    current_generators, local_step = changed
                    local_steps.append(local_step)
                    continue
                eligible = (
                    (max_local_generators is None or len(current_generators) <= max_local_generators)
                    and (max_local_variables is None or len(current_variables) <= max_local_variables)
                )
                if not eligible and len(current_generators) > 1:
                    status = "capped"
                    explanation = "Affine elimination is complete; the local-redundancy size cap skipped a colon search."
                    break
                changed = _remove_one_local_redundancy(current_generators, current_variables)
                if changed is not None:
                    current_generators, local_step = changed
                    local_steps.append(local_step)
                    # A Groebner basis may reintroduce the deleted generator.
                    # Preserve this smaller generating set while checking for
                    # additional local redundancies; only recompute after an
                    # actual source elimination or a stripped unit factor.
                    while True:
                        changed = _remove_one_local_redundancy(current_generators, current_variables)
                        if changed is None:
                            break
                        current_generators, local_step = changed
                        local_steps.append(local_step)
                    degrees = tuple(int(s.Poly(expression, *current_variables).total_degree())
                                    for expression in current_generators)
            status = "reduced"
            explanation = "Affine elimination is complete; the exact remaining generators are retained without radicalization."
            break
        if max_rounds is not None and len(steps) >= max_rounds:
            status = "capped"
            explanation = "The affine-elimination cap was reached; the recorded partial equivalence remains exact."
            break

        generator = current_generators[linear_index]
        pivot = next(variable for variable in current_variables if generator.diff(variable) != 0)
        coefficient = s.Rational(generator.diff(pivot))
        replacement = s.expand(pivot-generator/coefficient)
        regular = s.Dummy(f"ideal_regular_{len(steps)}", real=True)
        inverse_expression = regular+replacement
        step = AffineEliminationStep(
            variables_before=current_variables, groebner_generators=current_generators,
            generator=generator, eliminated_variable=pivot, coefficient=coefficient,
            replacement=replacement, regular_coordinate=regular,
            inverse_expression=inverse_expression,
        )
        steps.append(step)
        reconstruction = {
            variable: s.expand(expression.xreplace({pivot: inverse_expression}))
            for variable, expression in reconstruction.items()
        }
        # Polynomial division by pivot-replacement gives the exact identity
        # <pivot-replacement, f> = <pivot-replacement, f|pivot=replacement>.
        # The regular generator is recorded, not discarded as a zero-set trick.
        substituted = tuple(s.expand(expression.xreplace({pivot: replacement}))
                            for expression in current_generators)
        current_generators = tuple(expression for expression in substituted if expression != 0)
        current_variables = tuple(variable for variable in current_variables if variable != pivot)

    return IdealReduction(
        status=status, regular_count=len(steps), remaining_variables=current_variables,
        generators=current_generators, degrees=degrees, steps=tuple(steps),
        local_steps=tuple(local_steps),
        equivalence_scope="local_analytic" if local_steps else "polynomial",
        source_reconstruction=tuple((variable, reconstruction[variable]) for variable in original_variables),
        explanation=explanation,
    )


def _strip_one_unit_factor(
    generators: tuple[s.Expr, ...], variables: tuple[s.Symbol, ...],
) -> tuple[tuple[s.Expr, ...], LocalUnitStep] | None:
    origin = dict.fromkeys(variables, s.S.Zero)
    for index, generator in enumerate(generators):
        coefficient, factors = s.factor_list(generator, *variables)
        unit = s.sympify(coefficient)
        replacement = s.S.One
        nonconstant_unit = False
        for factor, exponent in factors:
            if factor.subs(origin) != 0:
                unit *= factor**exponent
                nonconstant_unit = True
            else:
                replacement *= factor**exponent
        if not nonconstant_unit:
            continue
        unit, replacement = s.expand(unit), s.expand(replacement)
        if s.expand(generator-unit*replacement) != 0 or unit.subs(origin) == 0:
            raise ArithmeticError("invalid local unit-factor certificate")
        remaining = generators[:index]+generators[index+1:]
        return remaining+(replacement,), LocalUnitStep(
            kind="strip_factor", variables=variables, generator=generator,
            unit=unit, replacement=replacement, remaining_generators=remaining,
        )
    return None


def _remove_one_local_redundancy(
    generators: tuple[s.Expr, ...], variables: tuple[s.Symbol, ...],
) -> tuple[tuple[s.Expr, ...], LocalUnitStep] | None:
    if len(generators) <= 1:
        return None
    for index, generator in enumerate(generators):
        remaining = generators[:index]+generators[index+1:]
        membership = local_ideal_membership(generator, remaining, variables)
        if not membership.member:
            continue
        return remaining, LocalUnitStep(
            kind="remove_redundant", variables=variables, generator=generator,
            unit=membership.unit, replacement=None, remaining_generators=remaining,
        )
    return None
