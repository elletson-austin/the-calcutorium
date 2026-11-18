"""Level curves: where does f(u, v) = c?  All pure functions.

    marching_squares   grid of values → line segments where the grid crosses one level
    auto_levels        values → a handful of round-numbered levels
    sample_contours    Equation → 3D line segments for each level (the output axis = c)
"""

import numpy as np

from .equation import AXES, Equation
from .ticks import nice_spacing

Range = tuple[float, float]

# Each grid cell has corners a (u0,v0), b (u1,v0), c (u1,v1), d (u0,v1) and edges
#   0: a–b (bottom)   1: b–c (right)   2: d–c (top)   3: a–d (left)
# The case number has one bit per corner above the level: a=1, b=2, c=4, d=8.
# For each case: which pairs of edges to join with a segment.
_CASES: dict[int, tuple[tuple[int, int], ...]] = {
    1: ((3, 0),), 2: ((0, 1),), 3: ((3, 1),), 4: ((1, 2),),
    6: ((0, 2),), 7: ((3, 2),), 8: ((2, 3),), 9: ((0, 2),),
    11: ((1, 2),), 12: ((1, 3),), 13: ((0, 1),), 14: ((0, 3),),
}
# Saddles (two opposite corners above): ambiguous, so look at the cell's center value.
# If the center is above, the two "above" corners connect through the middle and the
# lines cut off the other two corners; otherwise they cut off the "above" corners.
_CUT_A_AND_C = ((3, 0), (1, 2))
_CUT_B_AND_D = ((0, 1), (2, 3))
# case → (edge pairs if the center is above, edge pairs if it's below)
_SADDLES = {5: (_CUT_B_AND_D, _CUT_A_AND_C), 10: (_CUT_A_AND_C, _CUT_B_AND_D)}


def marching_squares(values: np.ndarray, u: np.ndarray, v: np.ndarray, level: float) -> np.ndarray:
    """Segments where `values` crosses `level`, shape (k, 2 endpoints, 2 coords (u, v)).

    values[i, j] is f(u[j], v[i]): rows follow v, columns follow u (as np.meshgrid gives).
    Cells with an undefined corner (nan) are skipped.
    """
    fa, fb = values[:-1, :-1], values[:-1, 1:]
    fc, fd = values[1:, 1:], values[1:, :-1]
    u0, u1 = u[None, :-1], u[None, 1:]
    v0, v1 = v[:-1, None], v[1:, None]

    with np.errstate(invalid="ignore", divide="ignore"):
        def cross(f_from, f_to):  # how far along an edge the level is crossed (0..1)
            return (level - f_from) / (f_to - f_from)

        edges = [
            (u0 + cross(fa, fb) * (u1 - u0), np.broadcast_to(v0, fa.shape)),  # 0 bottom
            (np.broadcast_to(u1, fa.shape), v0 + cross(fb, fc) * (v1 - v0)),  # 1 right
            (u0 + cross(fd, fc) * (u1 - u0), np.broadcast_to(v1, fa.shape)),  # 2 top
            (np.broadcast_to(u0, fa.shape), v0 + cross(fa, fd) * (v1 - v0)),  # 3 left
        ]
        points = [np.stack([eu, ev], axis=-1) for eu, ev in edges]  # each (rows, cols, 2)
        center_above = (fa + fb + fc + fd) / 4 > level

    defined = np.isfinite(fa) & np.isfinite(fb) & np.isfinite(fc) & np.isfinite(fd)
    case = (fa > level) * 1 + (fb > level) * 2 + (fc > level) * 4 + (fd > level) * 8
    case = np.where(defined, case, 0)

    def join(cells: np.ndarray, pairs) -> list[np.ndarray]:
        return [np.stack([points[e1][cells], points[e2][cells]], axis=1) for e1, e2 in pairs]

    segments = []
    for number, pairs in _CASES.items():
        segments += join(case == number, pairs)
    for number, (if_above, if_below) in _SADDLES.items():
        segments += join((case == number) & center_above, if_above)
        segments += join((case == number) & ~center_above, if_below)

    return np.concatenate(segments) if segments else np.zeros((0, 2, 2))


def auto_levels(values: np.ndarray, count: int = 10) -> tuple[float, ...]:
    """About `count` round-numbered levels strictly inside the range of the (finite) values."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return ()
    lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-12:
        return (lo,)
    step = nice_spacing(hi - lo, count)
    k = np.arange(np.ceil(lo / step), np.floor(hi / step) + 1)
    levels = [round(float(x), 12) + 0.0 for x in k * step]
    return tuple(c for c in levels if lo < c < hi)


def sample_contours(
    eq: Equation, range1: Range, range2: Range,
    levels: tuple[float, ...] | None = None, n: int = 120,
) -> list[tuple[float, np.ndarray]]:
    """Level curves of a 2-input equation, as 3D segments with the output axis set to the level.

    Returns [(level, segments of shape (k, 2, 3)), ...]. With levels=None, picks round
    levels from the values seen in this range. range1 is for eq.inputs[0], range2 for inputs[1].
    """
    if len(eq.inputs) != 2:
        raise ValueError(f"'{eq.text}' needs 2 inputs to have level curves")
    u, v = np.linspace(*range1, n), np.linspace(*range2, n)
    values = eq(*np.meshgrid(u, v))
    if levels is None:
        levels = auto_levels(values)

    index = {axis: i for i, axis in enumerate(AXES)}
    a, b = eq.inputs
    result = []
    for level in levels:
        flat = marching_squares(values, u, v, level)  # (k, 2, 2) in (inputs[0], inputs[1])
        segments = np.zeros(flat.shape[:2] + (3,))
        segments[..., index[a]] = flat[..., 0]
        segments[..., index[b]] = flat[..., 1]
        segments[..., index[eq.output]] = level
        result.append((level, segments))
    return result
