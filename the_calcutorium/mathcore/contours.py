"""Level curves: where does f(u, v) = c?  All pure functions.

    marching_squares   grid of values → line segments where the grid crosses one level
    contour_levels     values + interval → the levels c = k·interval inside their range
    sample_contours    Equation → 3D line segments for each level (the output axis = c)
"""

import numpy as np

from .equation import AXES, Equation
from .ticks import nice_spacing

Range = tuple[float, float]

# Each grid cell has corners a (u0,v0), b (u1,v0), c (u1,v1), d (u0,v1) and edges
#   0: a–b (bottom)   1: b–c (right)   2: d–c (top)   3: a–d (left)
# The case number has one bit per corner above the level: a=1, b=2, c=4, d=8.
# _EDGES[case] = the edges joined by the cell's first segment and (saddles only) second.
# -1 means "no segment". Cases 0 and 15 (all below / all above) have no crossing.
_NONE = (-1, -1)
_EDGES = np.array([
    #  first     second
    (_NONE,   _NONE),    # 0
    ((3, 0),  _NONE),    # 1   a
    ((0, 1),  _NONE),    # 2   b
    ((3, 1),  _NONE),    # 3   a b
    ((1, 2),  _NONE),    # 4   c
    ((3, 0),  (1, 2)),   # 5   a c    saddle (see below)
    ((0, 2),  _NONE),    # 6   b c
    ((3, 2),  _NONE),    # 7   a b c
    ((2, 3),  _NONE),    # 8   d
    ((0, 2),  _NONE),    # 9   a d
    ((0, 1),  (2, 3)),   # 10  b d    saddle (see below)
    ((1, 2),  _NONE),    # 11  a b d
    ((1, 3),  _NONE),    # 12  c d
    ((0, 1),  _NONE),    # 13  a c d
    ((0, 3),  _NONE),    # 14  b c d
    (_NONE,   _NONE),    # 15
])
# Saddles (two opposite corners above) are ambiguous, so look at the cell's center value.
# If the center is above, the two "above" corners connect through the middle, and the
# lines must cut off the other two corners instead: swap case 5's and case 10's edge pairs.
_SADDLE_SWAP = {5: 10, 10: 5}


def marching_squares(values: np.ndarray, u: np.ndarray, v: np.ndarray, level: float) -> np.ndarray:
    """Segments where `values` crosses `level`, shape (k, 2 endpoints, 2 coords (u, v)).

    values[i, j] is f(u[j], v[i]): rows follow v, columns follow u (as np.meshgrid gives).
    Cells with an undefined corner (nan) are skipped.
    """
    fa, fb = values[:-1, :-1], values[:-1, 1:]
    fc, fd = values[1:, 1:], values[1:, :-1]
    with np.errstate(invalid="ignore"):
        case = (fa > level) * 1 + (fb > level) * 2 + (fc > level) * 4 + (fd > level) * 8
    defined = np.isfinite(fa) & np.isfinite(fb) & np.isfinite(fc) & np.isfinite(fd)

    # Only the cells the level actually passes through. Usually a small fraction of the
    # grid, and doing the rest of the work for just these is what keeps this fast.
    rows, cols = np.nonzero(defined & (case != 0) & (case != 15))
    if len(rows) == 0:
        return np.zeros((0, 2, 2))
    a, b, c, d = fa[rows, cols], fb[rows, cols], fc[rows, cols], fd[rows, cols]
    case = case[rows, cols]
    center_above = (a + b + c + d) / 4 > level
    for saddle, other in _SADDLE_SWAP.items():
        case = np.where((case == saddle) & center_above, other, case)

    u0, u1, v0, v1 = u[cols], u[cols + 1], v[rows], v[rows + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        def cross(f_from, f_to):  # how far along an edge the level is crossed (0..1)
            return (level - f_from) / (f_to - f_from)

        points = np.stack([                                           # (cells, 4 edges, 2)
            np.stack([u0 + cross(a, b) * (u1 - u0), v0], axis=-1),   # 0 bottom
            np.stack([u1, v0 + cross(b, c) * (v1 - v0)], axis=-1),   # 1 right
            np.stack([u0 + cross(d, c) * (u1 - u0), v1], axis=-1),   # 2 top
            np.stack([u0, v0 + cross(a, d) * (v1 - v0)], axis=-1),   # 3 left
        ], axis=1)

    segments = []
    cell = np.arange(len(case))
    for which in (0, 1):  # first segment of every cell, then saddles' second segment
        e1, e2 = _EDGES[case, which, 0], _EDGES[case, which, 1]
        has = e1 >= 0
        segments.append(np.stack([points[cell[has], e1[has]], points[cell[has], e2[has]]], axis=1))
    return np.concatenate(segments)


MAX_LEVELS = 50  # more lines than this is unreadable, and slow to redraw while panning (~0.4 ms each)
AUTO_LEVEL_COUNT = 10


def contour_levels(values: np.ndarray, interval: float | None = None,
                   max_levels: int = MAX_LEVELS) -> tuple[float, ...]:
    """Every multiple of `interval` strictly inside the range of the (finite) values.

    interval=None picks a round interval giving about 10 levels. If the interval would
    give more than max_levels, it's multiplied up (×2, ×3, …) until it fits, so the
    levels shown are always multiples of the interval asked for.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return ()
    lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-12:
        return ()  # a constant: its "level set" is the whole plane, nothing to draw
    if interval is None:
        interval = nice_spacing(hi - lo, AUTO_LEVEL_COUNT)
    interval *= effective_multiplier(lo, hi, interval, max_levels)

    k = np.arange(np.ceil(lo / interval), np.floor(hi / interval) + 1)
    levels = [round(float(x), 12) + 0.0 for x in k * interval]
    return tuple(c for c in levels if lo < c < hi)


def effective_multiplier(lo: float, hi: float, interval: float, max_levels: int = MAX_LEVELS) -> int:
    """How many intervals to skip per drawn level so at most max_levels fit in [lo, hi]."""
    return max(1, int(np.ceil((hi - lo) / interval / max_levels)))


def sample_contours(
    eq: Equation, range1: Range, range2: Range,
    interval: float | None = None, n: int = 120,
) -> list[tuple[float, np.ndarray]]:
    """Level curves of a 2-input equation, as 3D segments with the output axis set to the level.

    Returns [(level, segments of shape (k, 2, 3)), ...], one entry per multiple of
    `interval` within the values seen in this range (see contour_levels).
    range1 is for eq.inputs[0], range2 for inputs[1].
    """
    if len(eq.inputs) != 2:
        raise ValueError(f"'{eq.text}' needs 2 inputs to have level curves")
    u, v = np.linspace(*range1, n), np.linspace(*range2, n)
    values = eq(*np.meshgrid(u, v))
    levels = contour_levels(values, interval)

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
