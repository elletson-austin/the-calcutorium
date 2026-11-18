"""Text drawn over the 2D view: grid numbers, axis names, plot labels, and the hover readout.

Drawn with QPainter after OpenGL finishes each frame.
"""

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..planes import AXIS_INDEX, PLANE_AXES
from ..render.meshes import GridLabels
from ..render.readout import ReadoutLine

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
    readout_lines: tuple[ReadoutLine, ...] = (),
) -> None:
    """plot_labels: (text, rgb color 0..1, world-space anchor point) per plot.
    readout_lines: the hover readout (render/readout.py); empty when the mouse is elsewhere."""
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

    if readout_lines:
        _draw_readout(painter, width, readout_lines, to_screen)
    painter.end()


def _draw_readout(painter: QPainter, width: int, lines: tuple[ReadoutLine, ...], to_screen) -> None:
    """A box of lines in the top-right corner, plus a dot on each curve at the cursor."""
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    for line in lines:
        if line.marker is not None:
            x, y = to_screen(*line.marker)
            painter.setPen(QPen(QColor(20, 20, 20), 1.5))
            painter.setBrush(QColor.fromRgbF(*line.color))
            painter.drawEllipse(QPoint(x, y), 4, 4)

    painter.setFont(QFont("Consolas", 10))
    metrics = QFontMetrics(painter.font())
    pad, row = 8, metrics.height()
    box_w = max(metrics.horizontalAdvance(line.text) for line in lines) + 2 * pad
    box = QRect(width - box_w - 10, 10, box_w, row * len(lines) + 2 * pad)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(0, 0, 0, 170))
    painter.drawRoundedRect(box, 4, 4)

    for i, line in enumerate(lines):
        painter.setPen(QColor.fromRgbF(*line.color) if line.color else QColor(220, 220, 220))
        painter.drawText(QRect(box.left() + pad, box.top() + pad + i * row, box_w, row),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, line.text)
