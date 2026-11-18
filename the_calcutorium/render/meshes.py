"""Scene data → vertex arrays, ready to upload to the GPU.

Everything here is pure numpy (no OpenGL), so it can be unit-tested. The GPU
side (renderer.py) only uploads `Mesh.data` and draws it with `Mesh.program`.

Vertex layouts per program:
    "basic"    position(3) color(3)
    "surface"  position(3) normal(3) color(3)
    "grid"     position(3) color(3) is_major(1)
"""

from dataclasses import dataclass

import numpy as np

from ..mathcore import sample_curve, sample_surface
from ..planes import AXES, AXIS_INDEX, PLANE_AXES
from ..scene import Plot

Range = tuple[float, float]

DEFAULT_DOMAIN: Range = (-10.0, 10.0)  # what curves/surfaces span in 3D
CURVE_SAMPLES = 500
SURFACE_SAMPLES = 100


@dataclass(frozen=True)
class Mesh:
    program: str        # which shader program draws it: "basic" | "surface" | "grid"
    mode: str           # "lines" | "triangles"
    data: np.ndarray    # float32, shape (vertex_count, floats_per_vertex)


# ---- axes & grid --------------------------------------------------------------

def axes_mesh(length: float = 10.0) -> Mesh:
    """Three colored axis lines: x red, y green, z blue."""
    rows = []
    for i in range(3):
        color = np.eye(3)[i]
        for sign in (-1, 1):
            pos = np.zeros(3)
            pos[i] = sign * length
            rows.append(np.concatenate([pos, color]))
    return Mesh("basic", "lines", np.array(rows, dtype=np.float32))


def nice_spacing(span: float, target_lines: int = 10) -> float:
    """A grid spacing of 1, 2 or 5 × 10^k giving about `target_lines` lines across `span`."""
    raw = span / target_lines
    power = 10 ** np.floor(np.log10(raw))
    for step, below in ((1, 1.5), (2, 3.5), (5, 7.5)):
        if raw / power < below:
            return float(step * power)
    return float(10 * power)


@dataclass(frozen=True)
class GridLabels:
    h: tuple[float, ...]  # values of major vertical lines (labelled along the bottom)
    v: tuple[float, ...]  # values of major horizontal lines (labelled along the left)


def grid_mesh(plane: str, h_range: Range, v_range: Range, major_every: int = 5) -> tuple[Mesh, GridLabels]:
    h_axis, v_axis, _ = PLANE_AXES[plane]
    hi, vi = AXIS_INDEX[h_axis], AXIS_INDEX[v_axis]
    spacing = nice_spacing(min(h_range[1] - h_range[0], v_range[1] - v_range[0]))

    def line_values(r: Range):
        k = np.arange(np.floor(r[0] / spacing), np.ceil(r[1] / spacing) + 1)
        return k * spacing, (k % major_every == 0)

    rows = []

    def add_lines(values, majors, along: int, across: int, extent: Range):
        for value, major in zip(values, majors):
            color = (0.7, 0.7, 0.7) if major else (0.4, 0.4, 0.4)
            for end in extent:
                pos = [0.0, 0.0, 0.0]
                pos[along], pos[across] = end, value
                rows.append([*pos, *color, float(major)])

    v_values, v_major = line_values(v_range)
    h_values, h_major = line_values(h_range)
    add_lines(v_values, v_major, along=hi, across=vi, extent=h_range)  # horizontal lines
    add_lines(h_values, h_major, along=vi, across=hi, extent=v_range)  # vertical lines

    labels = GridLabels(h=tuple(h_values[h_major]), v=tuple(v_values[v_major]))
    return Mesh("grid", "lines", np.array(rows, dtype=np.float32)), labels


# ---- plots ----------------------------------------------------------------------

def plot_mesh(plot: Plot, plane: str | None = None,
              h_range: Range = DEFAULT_DOMAIN, v_range: Range = DEFAULT_DOMAIN) -> Mesh | None:
    """The mesh for a plot in 3D (plane=None) or in a 2D plane's visible ranges.

    Returns None when there's nothing to draw (e.g. a surface in 2D, or a curve that
    doesn't lie in the plane).
    """
    if plot.is_surface:
        return surface_mesh(plot) if plane is None else None
    return curve_mesh(plot, plane, h_range, v_range)


def curve_mesh(plot: Plot, plane: str | None, h_range: Range, v_range: Range) -> Mesh | None:
    eq = plot.equation
    if plane is None:
        input_axis = eq.inputs[0] if eq.inputs else _first_axis_except(eq.output)
        domain, out_span = DEFAULT_DOMAIN, DEFAULT_DOMAIN[1] - DEFAULT_DOMAIN[0]
    else:
        h_axis, v_axis, _ = PLANE_AXES[plane]
        if eq.output not in (h_axis, v_axis):
            return None
        input_axis = v_axis if eq.output == h_axis else h_axis
        if eq.inputs and eq.inputs[0] != input_axis:
            return None
        domain = h_range if input_axis == h_axis else v_range
        out_range = v_range if input_axis == h_axis else h_range
        out_span = out_range[1] - out_range[0]

    # Sample twice as finely: even samples are segment endpoints, odd ones their midpoints.
    fine = sample_curve(eq, domain, 2 * CURVE_SAMPLES - 1, along=input_axis)
    segments = _curve_segments(fine[::2], fine[1::2], AXIS_INDEX[eq.output], out_span)
    if len(segments) == 0:
        return None
    colors = np.broadcast_to(plot.color, (len(segments), 3))
    return Mesh("basic", "lines", np.hstack([segments, colors]).astype(np.float32))


def _first_axis_except(axis: str) -> str:
    return next(a for a in AXES if a != axis)


def _curve_segments(points: np.ndarray, midpoints: np.ndarray, out_index: int, out_span: float) -> np.ndarray:
    """Consecutive point pairs as line segments, skipping undefined points and asymptotes.

    Asymptote test, applied only to segments with a big jump: a steep but continuous
    piece goes steadily one way, so the value halfway along lies between the endpoint
    values. Across an asymptote like tan(x) at π/2 the midpoint either shoots outside
    them or the curve reverses direction (up to +∞, then back from −∞).
    """
    a, b = points[:-1], points[1:]
    keep = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)

    fa, fb, fm = a[:, out_index], b[:, out_index], midpoints[:, out_index]
    with np.errstate(invalid="ignore"):
        big_jump = np.abs(fb - fa) > 0.05 * out_span
        monotone = (fm - fa) * (fb - fm) >= 0  # also False when fm is nan
        keep &= ~(big_jump & ~monotone)
    return np.stack([a[keep], b[keep]], axis=1).reshape(-1, 3)


def surface_mesh(plot: Plot, domain: Range = DEFAULT_DOMAIN, n: int = SURFACE_SAMPLES) -> Mesh | None:
    grid = sample_surface(plot.equation, domain, domain, n)  # (n, n, 3)

    # Each grid cell becomes two triangles: (p1, p3, p2) and (p2, p3, p4)
    p1, p2 = grid[:-1, :-1], grid[:-1, 1:]
    p3, p4 = grid[1:, :-1], grid[1:, 1:]
    tri_a = np.stack([p1, p3, p2], axis=2).reshape(-1, 3, 3)
    tri_b = np.stack([p2, p3, p4], axis=2).reshape(-1, 3, 3)
    triangles = np.concatenate([tri_a, tri_b])           # (count, 3 corners, xyz)
    triangles = triangles[np.isfinite(triangles).all(axis=(1, 2))]
    if len(triangles) == 0:
        return None

    normals = _face_normals(triangles)                    # (count, 3)
    normals = np.repeat(normals[:, None, :], 3, axis=1)   # one per corner
    colors = np.broadcast_to(plot.color, triangles.shape)
    data = np.concatenate([triangles, normals, colors], axis=2).reshape(-1, 9)
    return Mesh("surface", "triangles", data.astype(np.float32))


def _face_normals(triangles: np.ndarray) -> np.ndarray:
    n = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(length > 1e-12, length, 1.0)
