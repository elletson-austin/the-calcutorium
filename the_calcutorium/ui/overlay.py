"""Text drawn over the 2D view: grid numbers, axis names, and plot labels.

Drawn with QPainter after OpenGL finishes each frame.
"""

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QWidget

from ..planes import AXIS_INDEX, PLANE_AXES
from ..render.meshes import GridLabels

Range = tuple[float, float]


def _number(value: float) -> str:
    value = round(value, 10) + 0.0  # drop float noise and -0
    return f"{value:g}"


def draw_overlay(
    widget: QWidget,
    plane: str,
    h_range: Range,
    v_range: Range,
    grid: GridLabels,
    plot_labels: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]],
) -> None:
    """plot_labels: (text, rgb color 0..1, world-space anchor point) per plot."""
    width, height = widget.width(), widget.height()
    if width == 0 or height == 0:
        return
    h_axis, v_axis, _ = PLANE_AXES[plane]
    px_per_h = width / (h_range[1] - h_range[0])
    px_per_v = height / (v_range[1] - v_range[0])

    def to_screen(h: float, v: float) -> tuple[int, int]:
        return int((h - h_range[0]) * px_per_h), int(height - (v - v_range[0]) * px_per_v)

    painter = QPainter(widget)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    painter.setFont(QFont("Segoe UI", 9))
    painter.setPen(QColor(200, 200, 200))

    for h in grid.h:
        x, _ = to_screen(h, 0)
        if 0 <= x < width:
            painter.drawText(QRect(x - 30, height - 20, 60, 18), Qt.AlignmentFlag.AlignCenter, _number(h))
    for v in grid.v:
        _, y = to_screen(0, v)
        if 0 <= y < height:
            painter.drawText(QRect(4, y - 9, 60, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, _number(v))

    painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
    painter.drawText(QRect(width - 30, height - 40, 24, 20), Qt.AlignmentFlag.AlignCenter, h_axis)
    painter.drawText(QRect(36, 4, 24, 20), Qt.AlignmentFlag.AlignCenter, v_axis)

    painter.setFont(QFont("Segoe UI", 10))
    for text, color, anchor in plot_labels:
        h, v = anchor[AXIS_INDEX[h_axis]], anchor[AXIS_INDEX[v_axis]]
        if not (h_range[0] <= h <= h_range[1] and v_range[0] <= v <= v_range[1]):
            continue
        x, y = to_screen(h, v)
        painter.setPen(QColor.fromRgbF(*color))
        painter.drawText(QRect(x - 150, y - 24, 150, 20), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, text)

    painter.end()
