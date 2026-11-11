# ReLU RLCT

Compute exact local learning coefficients for shallow ReLU networks on finite data. Give the tool a dataset of inputs and a network at the point you want to study. It uses that network's predictions as targets and analyzes the squared loss as all weights and biases vary near the supplied point.

| Entry point | What it computes |
|---|---|
| `certify_scalar_relu` | The exact pair for scalar output, strict gates on the supplied inputs, and nonzero outgoing coefficients. Uses a matrix rank calculation. |
| `solve_shallow_relu` | The exact pair when one of the implemented symbolic certificates applies. Returns the residual equations and the attempted reductions when it cannot certify a pair. |
| `certify_shared_hub` | The exact pair and the graph of products responsible for it in a structured multioutput family. Takes a supplied grouping of the hidden neurons and checks the geometric hypotheses. |

## Install

Use Python 3.10 or newer. From a checkout of this repository, run

```sh
python -m pip install .
```

The only runtime dependency is SymPy.

## Quick start

```python
from relu_rlct import certify_scalar_relu

result = certify_scalar_relu(
    X=[[-2], [-1], [1], [2]],
    incoming=[[1, 0], [-1, 0]],
    outgoing=[[1], [-1]],
)
print(result.status, result.rlct, result.multiplicity)
# certified 2 1
```

For input dimension `d`, hidden width `H`, and output dimension `m`, the incoming matrix has shape `H × (d+1)` with the hidden bias last. The outgoing matrix has shape `H × m`. Use integers or exact rationals such as `sympy.Rational(1, 10)`. The tool rejects floating-point coefficients because equality and rank decisions are part of the certificates.

The model has one hidden layer, a trainable affine readout, and no skip connections. All three methods apply to the local finite-data squared loss with positive sample weights and a smooth parameter density positive at the marked point. In a small neighborhood, the exponent and pole multiplicity describe the leading power of the Laplace integral and its logarithmic factor, respectively. See [the mathematical scope and certificate methods](docs/methods.md).

The symbolic solver returns `certified`, `unresolved`, or `unsupported`. For a supported but unresolved input, `result.report.residuals` retains the exact equations and `result.attempts` records the certificate attempts. Gates at zero and zero path blocks are outside the symbolic solver's smooth chart. The shared-hub checker returns `unresolved` when the supplied grouping fails its geometric tests.

## Reproduce the examples

```sh
python examples/fold.py
python examples/benchmark.py --pairs 2 8 15 31
```

The fold example has 22 trainable parameters and two outputs. The residual Jacobian has rank 17. The remaining residual vanishes to order two along a single direction after eliminating the regular coordinates, giving an exact coefficient of $17/2 + 1/4 = 35/4$ and multiplicity 1.

The benchmark constructs structured teachers with a supplied anchor, hub, and neuron pairing. Each run measures the exact certificate call. It excludes fixture generation and measures scaling within this family. The timings below are from one run on arm64 macOS with Python 3.12 and SymPy 1.14.

| Hidden width | Inputs / outputs | Samples | Trainable parameters | RLCT | Multiplicity | Local run time |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 3 / 3 | 16 | 45 | 75/4 | 1 | 0.02 s |
| 18 | 9 / 9 | 100 | 351 | 657/4 | 1 | 0.04 s |
| 32 | 16 / 16 | 289 | 1,072 | 516 | 1 | 0.18 s |
| 64 | 32 / 32 | 1,089 | 4,192 | 2056 | 1 | 1.18 s |

The graph certificate also explains how the local geometry changes. In the width-6 example, the two neuron pairs and a bias direction give a triangle of products. Aligning one pair's outgoing vector with the hub removes one edge and changes the RLCT from $75/4$ to $37/2$ at the same Jacobian rank. The incoming weights and sampled gate patterns stay fixed. Each calculation uses the predictions of its own marked teacher as targets.

## Development

```sh
python -m unittest discover -s tests -v
python examples/fold.py
python examples/benchmark.py
```

The library extracts the computational parts of the shallow ReLU RLCT research by Arturs Berzins. See [provenance](docs/provenance.md) for the source revision and examples. Released under the [MIT license](LICENSE).
