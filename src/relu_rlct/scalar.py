"""Rank certificate for scalar-output networks with a smooth path chart."""

from dataclasses import dataclass

import sympy as sp
from sympy.polys.matrices import DomainMatrix

from .rlct_decision import _rational_matrix


@dataclass(frozen=True)
class ScalarCertificate:
    status: str
    rlct: sp.Rational | None
    multiplicity: int | None
    observation_rank: int | None
    gate_margin: sp.Rational
    explanation: str


def certify_scalar_relu(X, incoming, outgoing):
    """Compute the local pair from the exact gated observation matrix rank.

    X has shape (samples, inputs), incoming has shape (width, inputs + 1)
    with bias last, and outgoing has shape (width, 1). All entries must be
    rational. The output bias is trainable and its marked value is immaterial.
    """
    X = _rational_matrix(X, "X")
    incoming = _rational_matrix(incoming, "incoming")
    outgoing = _rational_matrix(outgoing, "outgoing")
    samples, dimension = X.shape
    width = incoming.rows
    if min(samples, dimension, width) < 1 or incoming.cols != dimension + 1:
        raise ValueError("Expected nonempty X and incoming matrices, with incoming bias last")
    if outgoing.shape != (width, 1):
        raise ValueError("outgoing must have shape (width, 1)")

    augmented = X.row_join(sp.ones(samples, 1))
    preactivation = augmented * incoming.T
    margin = min(map(abs, preactivation))
    if margin == 0:
        return ScalarCertificate("unsupported", None, None, None, margin,
                                 "A sample lies on a teacher gate boundary.")
    if any(weight == 0 for weight in outgoing):
        return ScalarCertificate("unsupported", None, None, None, margin,
                                 "The smooth path chart requires nonzero outgoing coefficients.")

    gates = preactivation.applyfunc(lambda value: 1 if value > 0 else 0)
    blocks = [
        sp.Matrix(samples, dimension + 1, lambda i, k: gates[i, j] * augmented[i, k])
        for j in range(width)
    ]
    observation = sp.Matrix.hstack(sp.ones(samples, 1), *blocks)
    rank = DomainMatrix.from_Matrix(observation).rank()
    return ScalarCertificate("certified", sp.Rational(rank, 2), 1, rank, margin,
                             "The fixed-gate prediction map is linear in signed path coordinates.")
