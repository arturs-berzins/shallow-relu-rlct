"""Teacher-level certificate for shared-hub complete-split-graph cores.

Ordinary shallow ReLU with trainable hidden/output biases and no affine skip.
Targets are the teacher predictions, sample weights are positive, and the
local parameter density is smooth and positive. All parameters vary when
computing the RLCT; the geometric tests concern only the marked teacher.

This checks exact rational data and a supplied neuron grouping. It neither
searches all groupings nor asserts dataset-minimality of an arbitrary input.
See shared_hub_geometric_criterion.md for the general analytic theorem.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import sympy as s


@dataclass(frozen=True)
class SharedHubCertificate:
    status: str
    rlct: Optional[s.Rational]
    multiplicity: Optional[int]
    pair_count: int
    active_pairs: Optional[tuple]
    regular_count: Optional[int]
    normalized_dimension: Optional[int]
    core_coordinate_count: Optional[int]  # Graph vertices, not residual or fiber dimension.
    anchor_aligned: Optional[bool]
    core_edges: Optional[tuple]
    half_fiber_codimension: Optional[s.Rational]
    deficit: Optional[s.Rational]
    explanation: str


def _matrix(value, name):
    result = s.Matrix(value)
    if any(entry.is_Rational is not True for entry in result):
        raise ValueError(f"{name} must contain exact rational numbers, not floats")
    return result


def certify_shared_hub(X, incoming, outgoing, *, anchor=0, hub=1, pairs=None):
    """Certify a local ideal/RLCT from samples, factors, and a neuron grouping.

    X is N x d; incoming is H x (d+1), with hidden bias last; outgoing is
    H x m. Default roles: anchor0, hub1, pairs(2,3),(4,5),.... Other roles
    must be supplied explicitly, and must partition all neurons. The marked
    output bias does not affect this certificate but remains trainable.

    Returned graph vertices 0,...,r-1 index PAIRS, not individual neurons;
    vertex r is the bias-redistribution coordinate when the anchor is
    aligned with the hub. Otherwise that coordinate is regular and absent.
    Missing structural
    hypotheses return unresolved, not a claim of a different RLCT.
    """
    X, incoming, outgoing = (_matrix(value, name) for value, name in
                             ((X, "X"), (incoming, "incoming"), (outgoing, "outgoing")))
    h, d, m = incoming.rows, X.cols, outgoing.cols
    if (X.rows == 0 or d == 0 or m == 0 or incoming.cols != d+1
            or outgoing.rows != h or h < 4 or h % 2):
        raise ValueError("Expected N x d, even H>=4 x (d+1), and H x m matrices")
    if pairs is None:
        pairs = tuple((j, j+1) for j in range(2, h, 2))
    else:
        pairs = tuple(tuple(pair) for pair in pairs)
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("Each supplied pair must contain exactly two indices")
    indices = (anchor, hub)+tuple(j for pair in pairs for j in pair)
    if (any(isinstance(j, bool) or not isinstance(j, int) for j in indices)
            or sorted(indices) != list(range(h))):
        raise ValueError("Anchor, hub, and pairs must partition all neuron indices")
    r, n = len(pairs), len(pairs)+1

    def unresolved(reason):
        return SharedHubCertificate("unresolved", None, None, r, None,
                                    None, None, None, None, None, None, None, reason)

    if min(d, m) < n:
        return unresolved("There are not enough input/output dimensions for independent spokes")
    augmented = X.row_join(s.ones(X.rows, 1))
    margins = augmented*incoming.T
    if any(value == 0 for value in margins):
        return unresolved("A sample lies on a teacher gate boundary")
    gates = margins.applyfunc(lambda value: s.Integer(int(bool(value > 0))))
    if gates[:, anchor]+gates[:, hub] != s.ones(X.rows, 1):
        return unresolved("Anchor and hub signatures must be complementary")
    if any(gates[:, first]+gates[:, second] != gates[:, hub]
           for first, second in pairs):
        return unresolved("Every child pair must partition the hub signature")
    # This is a rank test, not a requirement to observe every cube corner.
    if gates.rank() != r+2:
        return unresolved("The observed hub-positive sign rows do not affinely span all pairs")
    regions = {}
    for sample in range(X.rows):
        regions.setdefault(tuple(gates.row(sample)), []).append(sample)
    if any(augmented.extract(rows, range(d+1)).rank() != d+1
           for rows in regions.values()):
        return unresolved("An observed region lacks full augmented-input column rank")

    u0, f0 = outgoing.row(hub).T, incoming.row(hub)
    u_anchor = outgoing.row(anchor).T
    if u0 == s.zeros(m, 1) or f0 == s.zeros(1, d+1):
        return unresolved("The hub path block must be nonzero")
    if (u_anchor == s.zeros(m, 1)
            or incoming.row(anchor)[:, :d] == s.zeros(1, d)):
        return unresolved("The anchor must have nonzero outgoing vector and spatial normal")
    anchor_aligned = s.Matrix.hstack(u0, u_anchor).rank() == 1

    output_frame, input_frame, active = [u0], [f0], []
    for i, (first, second) in enumerate(pairs):
        U = s.Matrix.hstack(outgoing.row(first).T, outgoing.row(second).T)
        F = s.Matrix.vstack(incoming.row(first), incoming.row(second))
        if U.rank() != 2 or F.rank() != 2:
            return unresolved("Every marked pair jump needs independent incoming and outgoing factors")
        if U.row_join(u0).rank() != 2 or F.col_join(f0).rank() != 2:
            return unresolved("A pair's row or column plane does not contain the hub factor")
        transverse_output = [s.Matrix.hstack(u0, outgoing.row(j).T).rank() == 2
                             for j in (first, second)]
        transverse_input = [s.Matrix.vstack(f0, incoming.row(j)).rank() == 2
                            for j in (first, second)]
        if not all(transverse_input):
            return unresolved("A child incoming line equals the hub line, contrary to strict split gates")
        output_frame.append(outgoing.row((first, second)[transverse_output.index(True)]).T)
        input_frame.append(incoming.row(first))
        if all(transverse_output):
            active.append(i)
    if s.Matrix.hstack(*output_frame).rank() != n:
        return unresolved("The output spokes are not independent modulo the hub line")
    spatial_frame = s.Matrix.vstack(*input_frame)[:, :d]
    if spatial_frame.rank() != n:
        return unresolved("The affine input spokes do not have independent spatial projections")

    active = tuple(active)
    active_set = set(active)
    core_coordinate_count = r+int(anchor_aligned)
    edges = tuple((i, j) for i, j in combinations(range(core_coordinate_count), 2)
                  if i in active_set or j in active_set)
    dimension = h*(d+m)+m
    regular = dimension-core_coordinate_count
    a = len(active)
    if not edges:
        core_exponent, multiplicity = s.Integer(0), 1
    elif core_coordinate_count == 2:
        core_exponent, multiplicity = s.Rational(1, 2), 2
    else:
        core_exponent = min(s.Rational(a, 2), s.Rational(core_coordinate_count, 4))
        multiplicity = 2 if 2*a == core_coordinate_count else 1
    exponent = s.Rational(regular, 2)+core_exponent
    half_codim = s.Rational(regular+min(a, core_coordinate_count-1), 2)
    return SharedHubCertificate(
        "certified", exponent, multiplicity,
        r, active, regular, dimension, core_coordinate_count, anchor_aligned,
        edges, half_codim, half_codim-exponent,
        "All finite-data gate, regional-rank, marked-plane, and independent-spoke "
        "hypotheses checked. Exact LS gives the supplied complete-split-graph "
        "edge ideal, with all original parameters trainable. Minimality is not "
        "asserted for arbitrary accepted teachers; it has a separate family theorem."
    )
