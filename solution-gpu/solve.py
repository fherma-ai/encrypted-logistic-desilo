"""The logistic function of every element, over CKKS, with DESILO's engine.

1 / (1 + e^-x) over [-25, 25], held to 1e-3 everywhere.

The series is a Chebyshev projection of the logistic function itself, computed
in `init` from the function rather than carried as a table of numbers: degree
59, which is the degree the challenge-winning entry uses, and which reaches
3.4e-4 on this specification's domain — inside the bar with room for the
scheme's noise. This is the one place these three DESILO answers differ from
the OpenFHE ones beside them: the published component spells its series out by
hand, with the high-degree terms unrolled, and the polynomial here is the
plain projection at the same degree rather than that fit.

The Chebyshev basis is on [-1, 1] and the data is on [-25, 25], so the first
operation of the circuit is the scaling, which costs one level.
"""
import numpy as np

from fherma import Inputs, Outputs, Point

DEGREE = 59
BOUND = 25.0
#: Chebyshev nodes for the projection — far more than the degree needs.
NODES = 8192


def init(p: Point, cc):
    """The coefficients, from the function. Not measured, and sees no data."""
    k = np.arange(NODES)
    nodes = np.cos(np.pi * (k + 0.5) / NODES)
    values = 1.0 / (1.0 + np.exp(-nodes * BOUND))
    angles = np.pi * (k + 0.5) / NODES
    coefficients = np.array([
        (2.0 / NODES) * np.sum(values * np.cos(degree * angles))
        for degree in range(DEGREE + 1)
    ])
    coefficients[0] /= 2.0
    return coefficients.tolist()


def encoding(cc, inp: Inputs) -> list:
    # The whole vector in one packing, slot i holding element i.
    return [np.asarray(inp.xs.data, dtype=float).reshape(-1)]


def run(state, cc, cts: list) -> list:
    scaled = cc.engine.multiply(cts[0], 1.0 / BOUND)
    return [cc.engine.evaluate_chebyshev_polynomial(
        scaled, state, cc.keys.relinearization)]


def decoding(p: Point, cc, pts: list) -> Outputs:
    from fherma import Tensor

    values = np.real(np.asarray(pts[0])).tolist()
    return Outputs(y=Tensor((p.N,), values[:p.N], "f64"))
