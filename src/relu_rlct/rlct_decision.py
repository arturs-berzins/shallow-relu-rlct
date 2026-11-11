"""Compile a rational finite-data shallow ReLU germ into exact local ideals.

This module is the input/compiler layer, not a universal RLCT solver. It uses
the actual sample observation operator and does not require regional
unisolvency. The architecture has one hidden layer, arbitrary positive input,
hidden, and output dimensions, trainable hidden/output biases, and no skip.

All polynomial coordinates are centered at the teacher. Strict sample gates
and nonzero path blocks justify the smooth path charts and positive smooth
local density. Boundary or zero-block inputs receive an unsupported report.
Full activation rank is only a necessary minimality condition, never a
minimality certificate. Exact rational inputs are an implementation restriction,
not a hypothesis of the underlying real-analytic reduction.

The only RLCT route here is an injective chart Jacobian, which certifies a
regular germ. Every other supported input retains its exact polynomials for a
separate certificate/analyzer. The loss convention is squared empirical error
with arbitrary fixed positive observation weights and positive smooth density.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal

import sympy as s


MinimalityStatus = Literal["not_certified", "not_minimal"]
DecisionStatus = Literal["regular", "unresolved", "unsupported"]


@dataclass(frozen=True)
class AssumptionChecks:
    exact_rational: bool
    strict_cell: bool
    nonzero_path_blocks: bool
    regional_unisolvency: bool
    augmented_activation_full_column_rank: bool
    gate_margin: s.Rational
    input_pivot_indices: tuple[int | None, ...]
    zero_block_indices: tuple[int, ...]
    boundary_entries: tuple[tuple[int, int], ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class TransferPresentation:
    """All Segre minors on the exact affine sample-readout kernel.

    ``kernel_basis`` has shape (1+H*(d+1), nullity(L)). Its first row is the
    output-bias coefficient, followed by one augmented incoming-sized block
    per neuron. Independent copies of this kernel act in every output.

    ``regular_offset`` is the stable-equivalence correction
    (m*rank(L)-H*(m-1)*d)/2. It may be negative. If the transfer ideal's RLCT
    has been independently certified, adding this correction gives the network
    RLCT; the offset is not itself an RLCT certificate. Free neuron scalings
    have already been omitted. Zero minors are omitted from ``generators``.
    """

    variables: tuple[s.Symbol, ...]
    generators: tuple[s.Expr, ...]
    kernel_basis: s.Matrix
    path_blocks: tuple[s.Matrix, ...]
    bias_perturbation: s.Matrix
    regular_offset: s.Rational


@dataclass(frozen=True)
class DecisionReport:
    status: DecisionStatus
    scope: str
    unsupported_reasons: tuple[str, ...]
    assumptions: AssumptionChecks
    input_dimension: int
    output_dimension: int
    hidden_width: int
    sample_count: int
    raw_dimension: int
    effective_dimension: int
    teacher_output: s.Matrix
    gates: s.Matrix
    gate_matrix: s.Matrix
    region_sample_indices: tuple[tuple[int, ...], ...]
    regional_ranks: tuple[int, ...]
    augmented_activation_rank: int
    minimality_status: MinimalityStatus
    minimality_reason: str
    linear_observation_matrix: s.Matrix
    observation_rank: int
    selected_sample_indices: tuple[int, ...]
    residual_target_dimension: int
    variables: tuple[s.Symbol, ...]
    residuals: tuple[s.Expr, ...]
    jacobian: s.Matrix | None
    jacobian_rank: int | None
    kernel_dimension: int | None
    rlct: s.Rational | None
    multiplicity: int | None
    transfer: TransferPresentation | None

    @property
    def rho(self) -> int | None:
        """Rank of the complete weight-to-observation derivative.

        The chart derivative has the same rank as the original all-parameter
        derivative; the H discarded positive-scaling coordinates are free.
        """
        return self.jacobian_rank

    def to_summary(self, *, include_polynomials: bool = False) -> dict[str, Any]:
        """A JSON-compatible report; symbolic values are exact strings."""
        summary: dict[str, Any] = {
            "status": self.status,
            "scope": self.scope,
            "unsupported_reasons": list(self.unsupported_reasons),
            "dimensions": {
                "input": self.input_dimension, "output": self.output_dimension,
                "hidden": self.hidden_width, "samples": self.sample_count,
                "raw": self.raw_dimension, "effective": self.effective_dimension,
                "residual_target": self.residual_target_dimension,
            },
            "assumptions": {
                "exact_rational": self.assumptions.exact_rational,
                "strict_cell": self.assumptions.strict_cell,
                "nonzero_path_blocks": self.assumptions.nonzero_path_blocks,
                "regional_unisolvency": self.assumptions.regional_unisolvency,
                "augmented_activation_full_column_rank":
                    self.assumptions.augmented_activation_full_column_rank,
                "gate_margin": str(self.assumptions.gate_margin),
                "input_pivot_indices": list(self.assumptions.input_pivot_indices),
                "zero_block_indices": list(self.assumptions.zero_block_indices),
                "boundary_entries": [list(entry) for entry in self.assumptions.boundary_entries],
                "notes": list(self.assumptions.notes),
            },
            "gate_matrix": self.gate_matrix.tolist(),
            "regional_ranks": list(self.regional_ranks),
            "augmented_activation_rank": self.augmented_activation_rank,
            "minimality_status": self.minimality_status,
            "minimality_reason": self.minimality_reason,
            "observation_rank": self.observation_rank,
            "selected_sample_indices": list(self.selected_sample_indices),
            "jacobian_rank": self.jacobian_rank,
            "kernel_dimension": self.kernel_dimension,
            "rlct": None if self.rlct is None else str(self.rlct),
            "multiplicity": self.multiplicity,
        }
        # SymPy Integer entries need conversion before json.dumps.
        summary["gate_matrix"] = [[int(value) for value in row]
                                  for row in summary["gate_matrix"]]
        if include_polynomials:
            summary["variables"] = list(map(str, self.variables))
            summary["residuals"] = list(map(str, self.residuals))
            if self.transfer is not None:
                summary["transfer"] = {
                    "variables": list(map(str, self.transfer.variables)),
                    "generators": list(map(str, self.transfer.generators)),
                    "regular_offset": str(self.transfer.regular_offset),
                }
        return summary


def _rational_matrix(value: Any, name: str) -> s.Matrix:
    try:
        matrix = s.Matrix(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular matrix of exact rational numbers") from error
    if any(entry.is_Rational is not True for entry in matrix):
        raise ValueError(f"{name} must contain exact rational numbers; floats and symbolic entries are unsupported")
    return matrix


def _output_bias(value: Any, dimension: int) -> s.Matrix:
    if value is None:
        return s.zeros(dimension, 1)
    try:
        bias = s.Matrix(value)
    except TypeError:
        bias = s.Matrix([value])
    if bias.shape == (1, dimension):
        bias = bias.T
    if bias.shape != (dimension, 1):
        raise ValueError("output_bias must be a vector with one entry per output")
    return _rational_matrix(bias, "output_bias")


def _transfer_presentation(
    observation: s.Matrix, incoming: s.Matrix, outgoing: s.Matrix,
    rank: int, prefix: str,
) -> TransferPresentation:
    H, p = incoming.shape
    m = outgoing.cols
    basis = observation.nullspace()
    kernel = s.Matrix.hstack(*basis) if basis else s.zeros(observation.cols, 0)
    variables = tuple(s.symbols(f"{prefix}_t0:{m*kernel.cols}", real=True))
    # Row = observation-kernel basis index; column = output index.
    parameters = s.Matrix(kernel.cols, m, variables)
    changes = kernel*parameters
    blocks = tuple(outgoing[j, :].T*incoming[j, :]
                   + changes[1+j*p:1+(j+1)*p, :].T for j in range(H))
    generators = []
    for block in blocks:
        for rows in combinations(range(m), 2):
            for columns in combinations(range(p), 2):
                minor = s.expand(block.extract(rows, columns).det())
                if minor != 0:
                    generators.append(minor)
    return TransferPresentation(
        variables=variables, generators=tuple(generators), kernel_basis=kernel,
        path_blocks=blocks, bias_perturbation=changes[0, :].T,
        regular_offset=s.Rational(m*rank-H*(m-1)*(p-1), 2),
    )


def compile_shallow_relu(
    X: Any, incoming: Any, outgoing: Any, output_bias: Any = None, *,
    prefix: str = "z", build_transfer: bool = False,
) -> DecisionReport:
    """Construct an exact local residual ideal from a rational ReLU teacher.

    Shapes are X=(N,d), incoming=(H,d+1), outgoing=(H,m). The final incoming
    column is the hidden bias. Output bias is always trainable; omitting its
    teacher value sets that value to zero rather than freezing the parameter.

    For neuron j, choose a nonzero teacher incoming coordinate c_j=w_jk^*.
    Keep that pivot fixed at c_j in the chart and let every other incoming
    coordinate vary. For original weights the omitted scale is
    tau_j=w_jk/c_j>0 locally, the chart incoming row is w_j/tau_j, and its
    outgoing coordinate is alpha_j=tau_j*v_j. Thus this is an actual positive
    ReLU scaling even when c_j is negative. The exact path block
    alpha_j*b_j^T gives quadratic prediction residuals.

    A basis of the *rows* of L=[1 | diag(gamma_j) Xbar for all j] selects sample
    residuals generating exactly the full empirical ideal. This selection does
    not replace deficient regional covariances by artificial full-rank data.
    """
    if not isinstance(prefix, str) or not prefix.isidentifier():
        raise ValueError("prefix must be a nonempty Python-style identifier")
    samples = _rational_matrix(X, "X")
    weights = _rational_matrix(incoming, "incoming")
    vectors = _rational_matrix(outgoing, "outgoing")
    N, d = samples.shape
    H, p = weights.shape
    if N < 1 or d < 1 or H < 1:
        raise ValueError("sample count, input dimension, and hidden width must be positive")
    if p != d+1:
        raise ValueError("incoming must have d+1 columns, including the hidden bias")
    if vectors.rows != H or vectors.cols < 1:
        raise ValueError("outgoing must have H rows and at least one output column")
    m = vectors.cols
    bias = _output_bias(output_bias, m)
    augmented = samples.row_join(s.ones(N, 1))
    preactivation = augmented*weights.T
    gates = preactivation.applyfunc(lambda value: s.Integer(1) if value > 0 else s.Integer(0))
    activations = preactivation.applyfunc(lambda value: value if value > 0 else s.Integer(0))
    target = activations*vectors+s.ones(N, 1)*bias.T

    region_map: dict[tuple[int, ...], list[int]] = {}
    for row in range(N):
        signature = tuple(int(value) for value in gates[row, :])
        region_map.setdefault(signature, []).append(row)
    gamma = s.Matrix(list(region_map))
    region_indices = tuple(tuple(rows) for rows in region_map.values())
    regional_ranks = tuple(augmented[list(rows), :].rank() for rows in region_indices)
    feature_rank = int(s.ones(N, 1).row_join(activations).rank())
    boundary = tuple((i, j) for i in range(N) for j in range(H) if preactivation[i, j] == 0)
    pivots = tuple(next((k for k in range(p) if weights[j, k] != 0), None) for j in range(H))
    zero_blocks = tuple(j for j in range(H)
                        if pivots[j] is None or all(value == 0 for value in vectors[j, :]))
    notes = (
        "Regional unisolvency is reported but is not required by this sample-level compiler.",
        "Full rank of [1,Z] is necessary for dataset-minimality, not a certificate of it.",
        "RLCT statements assume squared loss, positive sample weights, and positive smooth local density.",
    )
    checks = AssumptionChecks(
        exact_rational=True, strict_cell=not boundary,
        nonzero_path_blocks=not zero_blocks,
        regional_unisolvency=all(rank == p for rank in regional_ranks),
        augmented_activation_full_column_rank=feature_rank == H+1,
        gate_margin=min(abs(value) for value in preactivation),
        input_pivot_indices=pivots, zero_block_indices=zero_blocks,
        boundary_entries=boundary, notes=notes,
    )
    if zero_blocks:
        minimality: MinimalityStatus = "not_minimal"
        minimality_reason = "A zero path block contributes zero and its neuron can be deleted."
    elif feature_rank < H+1:
        minimality = "not_minimal"
        minimality_reason = "An exact dependence among [1,Z] eliminates a neuron by changing outgoing weights and output bias."
    else:
        minimality = "not_certified"
        minimality_reason = "The necessary augmented activation-rank condition holds; no minimal-width proof was supplied."

    gated_inputs = [s.Matrix(N, p, lambda i, k: gates[i, j]*augmented[i, k]) for j in range(H)]
    observation = s.Matrix.hstack(s.ones(N, 1), *gated_inputs)
    selected = tuple(int(i) for i in observation.T.rref()[1])
    rank_L = len(selected)
    effective = H*(m+d)+m
    reasons = tuple(reason for failed, reason in (
        (bool(boundary), "gate_boundary"), (bool(zero_blocks), "zero_path_block"),
    ) if failed)
    common = dict(
        scope="Ordinary biased shallow ReLU; exact rational finite samples; all weights/biases trainable; no skip.",
        unsupported_reasons=reasons, assumptions=checks,
        input_dimension=d, output_dimension=m, hidden_width=H, sample_count=N,
        raw_dimension=H*(p+m)+m, effective_dimension=effective,
        teacher_output=target, gates=gates, gate_matrix=gamma,
        region_sample_indices=region_indices, regional_ranks=regional_ranks,
        augmented_activation_rank=feature_rank, minimality_status=minimality,
        minimality_reason=minimality_reason, linear_observation_matrix=observation,
        observation_rank=rank_L, selected_sample_indices=selected,
        residual_target_dimension=m*rank_L,
    )
    if reasons:
        return DecisionReport(
            status="unsupported", variables=(), residuals=(), jacobian=None,
            jacobian_rank=None, kernel_dimension=None, rlct=None, multiplicity=None,
            transfer=None, **common,
        )

    variables = tuple(s.symbols(f"{prefix}0:{effective}", real=True))
    cursor = 0
    blocks = []
    for j, pivot in enumerate(pivots):
        assert pivot is not None  # Already excluded above, before taking a chart.
        alpha = vectors[j, :].T+s.Matrix(variables[cursor:cursor+m])
        cursor += m
        normalized = weights[j, :].copy()
        for k in range(p):
            if k != pivot:
                normalized[k] += variables[cursor]
                cursor += 1
        blocks.append(alpha*normalized)
    chart_bias = bias+s.Matrix(variables[cursor:cursor+m])
    cursor += m
    assert cursor == effective
    residuals = []
    for i in selected:
        prediction = chart_bias.copy()
        for j, block in enumerate(blocks):
            prediction += gates[i, j]*block*augmented[i, :].T
        residuals.extend(s.expand(prediction[a]-target[i, a]) for a in range(m))
    centered = dict.fromkeys(variables, s.Integer(0))
    assert all(residual.subs(centered) == 0 for residual in residuals)
    jacobian = s.Matrix(residuals).jacobian(variables).subs(centered)
    rho = int(jacobian.rank())
    nullity = effective-rho
    regular = nullity == 0
    transfer = (_transfer_presentation(observation, weights, vectors, rank_L, prefix)
                if build_transfer else None)
    return DecisionReport(
        status="regular" if regular else "unresolved", variables=variables,
        residuals=tuple(residuals), jacobian=jacobian, jacobian_rank=rho,
        kernel_dimension=nullity, rlct=s.Rational(effective, 2) if regular else None,
        multiplicity=1 if regular else None, transfer=transfer, **common,
    )
