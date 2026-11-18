"""What's under the cursor in a 2D view, as lines of text. Pure: same arguments, same lines.

meshes.py turns the scene into vertex arrays; this turns the scene plus a cursor
point into the hover readout. The overlay only draws what this returns.

    x = 1.50,  y = 0.75              the cursor, rounded to what one pixel can show
    y = x^2:  y = 2.25               a curve, at the cursor's x (and a dot on the curve there)
    z = x^2 + y^2:  z = 2.8125       a surface or contour, at the cursor's (x, y)
"""

import math
from dataclasses import dataclass

import numpy as np

from ..planes import PLANE_AXES
from ..scene import Color, Contour, Curve, Plot, Surface
from .meshes import curve_axis_in_plane

Range = tuple[float, float]
Point = tuple[float, float]  # (horizontal, vertical) in plane coordinates


@dataclass(frozen=True)
class ReadoutLine:
    text: str
    color: Color | None = None   # None: the overlay's normal text color
    marker: Point | None = None  # a point on a curve to mark with a dot


def plane_point(pixel: Point, size: tuple[int, int], h_range: Range, v_range: Range) -> Point:
    """Widget pixel (x right, y down) → plane coordinates (h right, v up)."""
    (px, py), (width, height) = pixel, size
    h = h_range[0] + px / width * (h_range[1] - h_range[0])
    v = v_range[1] - py / height * (v_range[1] - v_range[0])
    return h, v


def readout(plots: list[Plot], plane: str, point: Point, step: float) -> tuple[ReadoutLine, ...]:
    """The readout at `point` in `plane`. `step` is the size of one pixel in world units.

    The point is first rounded to the digits that `step` can resolve, and every value
    is computed at that rounded point, so the numbers shown always agree with each other.
    """
    h_axis, v_axis, _ = PLANE_AXES[plane]
    decimals = max(0, math.ceil(-math.log10(step) - 1e-9))
    h, v = round(point[0], decimals), round(point[1], decimals)
    at = {h_axis: h, v_axis: v}

    lines = [ReadoutLine(f"{h_axis} = {_fixed(h, decimals)},  {v_axis} = {_fixed(v, decimals)}")]
    shown: set[str] = set()
    for plot in plots:
        line = _plot_line(plot, plane, at)
        if line is not None and plot.text not in shown:  # a Surface and its Contour: once
            shown.add(plot.text)
            lines.append(line)
    return tuple(lines)


def _plot_line(plot: Plot, plane: str, at: dict[str, float]) -> ReadoutLine | None:
    eq = plot.equation
    h_axis, v_axis, _ = PLANE_AXES[plane]

    if isinstance(plot, Curve):
        input_axis = curve_axis_in_plane(eq, plane)
        if input_axis is None:
            return None
        value = _evaluate(plot, at)
        marker = None
        if math.isfinite(value):
            point = {input_axis: at[input_axis], eq.output: value}
            marker = (point[h_axis], point[v_axis])
        return ReadoutLine(_describe(plot, value), plot.color, marker)

    if isinstance(plot, (Surface, Contour)):
        if set(eq.inputs) != {h_axis, v_axis}:
            return None
        return ReadoutLine(_describe(plot, _evaluate(plot, at)), plot.color)

    return None


def _evaluate(plot: Plot, at: dict[str, float]) -> float:
    eq = plot.equation
    return float(eq(*(np.float64(at[axis]) for axis in eq.inputs)))


def _describe(plot: Plot, value: float) -> str:
    if not math.isfinite(value):
        return f"{plot.text}:  undefined"
    return f"{plot.text}:  {plot.equation.output} = {_significant(value)}"


def _fixed(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}"
    return text.lstrip("-") if float(text) == 0 else text  # "-0.00" → "0.00"


def _significant(value: float) -> str:
    return f"{round(value, 10) + 0.0:.5g}"  # drop float noise and -0
