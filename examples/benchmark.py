"""Measure exact certification on a structured family with increasing width."""

from argparse import ArgumentParser
from time import perf_counter

from sympy import Matrix, Rational

from relu_rlct import certify_shared_hub


def shared_hub_example(pair_count):
    """Build a strict teacher and enough samples to verify the gate rank tests."""
    r = pair_count
    dimension = r + 1
    incoming = [[-1] + [Rational(1, 2 * r)] * r + [-1], [1] + [0] * (r + 1)]
    hub = [0] * r + [1]
    outgoing = [hub.copy(), hub.copy()]
    for i in range(r):
        for sign in (1, -1):
            row = [1] + [0] * (r + 1)
            row[i + 1] = sign
            vector = hub.copy()
            vector[i] = -sign
            incoming.append(row)
            outgoing.append(vector)

    # The all-negative corner and its r single-coordinate flips suffice.
    base = [1] + [-3] * r
    centers = [[-2] + [0] * r, base]
    for i in range(1, dimension):
        corner = base.copy()
        corner[i] = 3
        centers.append(corner)
    points = []
    for center in centers:
        points.append(center)
        for i in range(dimension):
            offset = center.copy()
            offset[i] += Rational(1, 100)
            points.append(offset)
    return Matrix(points), Matrix(incoming), Matrix(outgoing)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", type=int, default=[2, 8, 15],
                        help="Numbers of neuron pairs to benchmark, each at least 2")
    args = parser.parse_args()
    if any(pairs < 2 for pairs in args.pairs):
        parser.error("each pair count must be at least 2")

    print("width,inputs,outputs,samples,parameters,rlct,multiplicity,seconds")
    for pairs in args.pairs:
        X, incoming, outgoing = shared_hub_example(pairs)
        start = perf_counter()
        result = certify_shared_hub(X, incoming, outgoing)
        seconds = perf_counter() - start
        width, dimension, outputs = incoming.rows, X.cols, outgoing.cols
        parameter_count = width * (dimension + outputs + 1) + outputs
        expected = 2 * (pairs + 1) ** 2 + Rational(pairs + 1, 4)
        assert (result.status, result.rlct, result.multiplicity) == ("certified", expected, 1)
        print(f"{width},{dimension},{outputs},{X.rows},{parameter_count},"
              f"{result.rlct},{result.multiplicity},{seconds:.6f}")


if __name__ == "__main__":
    main()
