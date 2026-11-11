"""Compose finite-data compilation, exact ideal reduction, and RLCT certificates.

No dispatch uses example names or gate-circuit labels. A failed certificate
leaves an unresolved exact ideal; computational caps never imply vanishing.
"""

from dataclasses import dataclass, field
from typing import Any

import sympy as s

from .rlct_decision import DecisionReport, compile_shallow_relu
from .rlct_certificates import certify_compiled, certify_polynomial_map
from .rlct_ideal_reduction import reduce_polynomial_ideal
from .rlct_monomial import certify_monomial_ideal


@dataclass
class CertificateAttempt:
    presentation: str
    regular_offset: s.Rational
    certificate: Any
    reduction: Any = None


@dataclass
class NetworkRLCT:
    status: str
    report: DecisionReport
    rlct: s.Rational | None = None
    multiplicity: int | None = None
    attempts: list[CertificateAttempt] = field(default_factory=list)
    explanation: str = ""


def solve_shallow_relu(X, incoming, outgoing, output_bias=None, *,
                      max_corank_one_order=8, max_transfer_variables=16,
                      max_monomial_vertex_systems=20000, max_critical_membership_variables=10):
    """A conservative exact rational solver for a teacher-generated finite loss.

    First try the normalized-weight residual. If needed and the transfer
    dimension is within the stated cap, compile the toric-affine ideal, try it
    directly, then apply exact affine/verified-unit preprocessing. Try the full
    monomial ideal and polynomial LS certificates on that reduced ideal.

    The transfer dimension cap is a workload guard, not a mathematical
    restriction. Individual Groebner computations have no runtime guarantee.
    The implicit-order cap also applies to constrained critical curves.
    """
    if isinstance(max_transfer_variables, bool) or not isinstance(max_transfer_variables, int) or max_transfer_variables < 0:
        raise ValueError("max_transfer_variables must be a nonnegative integer")
    report = compile_shallow_relu(X, incoming, outgoing, output_bias)
    result = NetworkRLCT(report.status, report)
    if report.status == "unsupported":
        result.explanation = "; ".join(report.unsupported_reasons)
        return result
    result.status = "unresolved"

    def attempt(presentation, offset, certificate, reduction=None):
        result.attempts.append(CertificateAttempt(presentation, offset, certificate, reduction))
        if certificate.status != "certified":
            return False
        result.status = "certified"
        result.rlct = offset+certificate.rlct
        result.multiplicity = certificate.multiplicity
        result.explanation = f"Certified through {presentation}; exact regular/stable offset {offset}."
        return True

    certificate_options = dict(max_corank_one_order=max_corank_one_order,
                               max_critical_membership_variables=max_critical_membership_variables)
    direct = certify_compiled(report, **certificate_options)
    if attempt("normalized weights", s.S.Zero, direct):
        return result
    transfer_dimension = report.output_dimension*(
        1+report.hidden_width*(report.input_dimension+1)-report.observation_rank)
    if transfer_dimension > max_transfer_variables:
        result.explanation = (f"Direct tests unresolved; transfer dimension {transfer_dimension} exceeds "
                              f"workload cap {max_transfer_variables}. Exact weight residuals are retained.")
        return result
    report = compile_shallow_relu(X, incoming, outgoing, output_bias, build_transfer=True)
    result.report = report
    transfer = report.transfer
    if transfer.variables:
        toric = certify_polynomial_map(transfer.generators, transfer.variables, **certificate_options)
        if attempt("toric transfer", transfer.regular_offset, toric):
            return result
    reduction = reduce_polynomial_ideal(transfer.generators, transfer.variables)
    offset = transfer.regular_offset+s.Rational(reduction.regular_count, 2)
    monomial = certify_monomial_ideal(reduction.generators, reduction.remaining_variables,
                                     max_vertex_systems=max_monomial_vertex_systems)
    if attempt("exact reduced monomial ideal", offset, monomial, reduction):
        return result
    reduced = certify_polynomial_map(reduction.generators, reduction.remaining_variables, **certificate_options)
    if attempt("exact reduced polynomial ideal", offset, reduced, reduction):
        return result
    result.explanation = ("No implemented certificate settles this ideal. All attempted presentations and "
                          "the exact remaining equations are retained; no exponent is inferred from a short jet.")
    return result
