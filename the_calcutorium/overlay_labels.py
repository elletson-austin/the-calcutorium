from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QFont, QColor

from .scene import SceneObject, MathFunction, Grid
from .render_types import SnapMode

if TYPE_CHECKING:
    from .render_window import RenderWindow

class OverlayLabelRenderer:
    """Responsible for 2D overlay labels (grid, functions, arbitrary objects)."""
    def __init__(self, render_window: 'RenderWindow'):
        self._rw = render_window


    def _begin_label_painter(self, font_size: int = 10, bold: bool = False, color: QColor | None = None) -> QPainter:
        painter = QPainter(self._rw)
        weight = QFont.Weight.Bold if bold else QFont.Weight.Normal
        painter.setFont(QFont("Arial", font_size, weight))
        painter.setPen(color or QColor(200, 200, 200))
        return painter

    def _get_2d_plane_components(self):
        cam2d = self._rw.camera
        axis_components = {
            SnapMode.XY: ("x", "y"),
            SnapMode.XZ: ("x", "z"),
            SnapMode.YZ: ("z", "y"),
        }
        return axis_components.get(cam2d.snap_mode, ("x", "y"))

    def _compute_view_scaling(self, h_range, v_range, width: int, height: int):
        h_diff = h_range[1] - h_range[0]
        v_diff = v_range[1] - v_range[0]

        h_pixels_per_unit = width / h_diff if h_diff != 0 else 0
        v_pixels_per_unit = height / v_diff if v_diff != 0 else 0
        return h_pixels_per_unit, v_pixels_per_unit

    def _world_to_screen(self, world_pos, h_range, v_range, width: int, height: int):
        h_comp, v_comp = self._get_2d_plane_components()
        comp_index = {"x": 0, "y": 1, "z": 2}
        x, y, z = world_pos
        world = [x, y, z]

        h_val = world[comp_index[h_comp]]
        v_val = world[comp_index[v_comp]]

        h_pixels_per_unit, v_pixels_per_unit = self._compute_view_scaling(h_range, v_range, width, height)

        screen_x = int((h_val - h_range[0]) * h_pixels_per_unit)
        screen_y = int(height - (v_val - v_range[0]) * v_pixels_per_unit)
        return screen_x, screen_y, h_val, v_val

    def _get_object_label_text(self, obj: SceneObject) -> str:
        if hasattr(obj, "label") and getattr(obj, "label"):
            return str(obj.label)
        if getattr(obj, "name", None):
            return str(obj.name)
        if getattr(obj, "equation_str", None):
            return str(obj.equation_str)
        return type(obj).__name__

    # Public API

    def render_grid_labels(self, h_range, v_range):
        if h_range is None or v_range is None:
            return
        if self._rw.scene is None:
            return

        grid_obj = None
        for obj in self._rw.scene.objects:
            if isinstance(obj, Grid):
                grid_obj = obj
                break

        if grid_obj is None:
            return

        width, height = self._rw.width(), self._rw.height()
        painter = self._begin_label_painter(font_size=10, bold=False)

        h_axis_name, v_axis_name = self._get_2d_plane_components()

        h_pixels_per_unit, v_pixels_per_unit = self._compute_view_scaling(h_range, v_range, width, height)

        if not hasattr(grid_obj, "labels"):
            return

        labels = grid_obj.labels

        if "h_labels" in labels:
            for h_val, h_min, h_max, h_idx in labels["h_labels"]:
                screen_x = int((h_val - h_range[0]) * h_pixels_per_unit)
                screen_y = height - 20

                if 0 <= screen_x < width:
                    label_text = f"{h_val:.1f}"
                    painter.drawText(screen_x - 15, screen_y, 30, 20, Qt.AlignCenter, label_text)

        # Draw vertical axis numeric labels (left)
        if "v_labels" in labels:
            for v_val, v_min, v_max, v_idx in labels["v_labels"]:
                screen_y = int(height - (v_val - v_range[0]) * v_pixels_per_unit)
                screen_x = 5  # Near left edge

                if 0 <= screen_y < height:
                    label_text = f"{v_val:.1f}"
                    painter.drawText(screen_x, screen_y - 10, 35, 20, Qt.AlignCenter, label_text)

        painter.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        painter.drawText(width - 40, height - 25, 30, 20, Qt.AlignCenter, h_axis_name)
        painter.drawText(5, 5, 30, 20, Qt.AlignCenter, v_axis_name)

        painter.end()

    def render_function_labels(self, h_range, v_range):
        if h_range is None or v_range is None:
            return
        if self._rw.scene is None:
            return

        width, height = self._rw.width(), self._rw.height()
        if width == 0 or height == 0:
            return

        painter = self._begin_label_painter(font_size=10, bold=False, color=QColor(230, 230, 230))

        for obj in self._rw.scene.objects:
            if not isinstance(obj, MathFunction):
                continue
            if obj.vertices.size == 0:
                continue

            # Take the last vertex as a simple label anchor
            data = obj.vertices.reshape(-1, 6)
            x, y, z = data[-1, 0:3]

            screen_x, screen_y, h_val, v_val = self._world_to_screen(
                (x, y, z), h_range, v_range, width, height
            )

            # Skip labels that are far outside the current view (world space)
            if not (h_range[0] <= h_val <= h_range[1] and v_range[0] <= v_val <= v_range[1]):
                continue

            # Convert to screen coords (offset a bit so text doesn't overlap the curve endpoint)
            screen_x += 8
            screen_y -= 8

            label_text = self._get_object_label_text(obj)
            if not label_text:
                continue

            painter.drawText(screen_x, screen_y - 10, 120, 20, Qt.AlignLeft | Qt.AlignVCenter, label_text)

        painter.end()

    def render_object_labels(self, h_range, v_range):
        """Render labels for arbitrary objects that expose label & label_position."""
        if h_range is None or v_range is None:
            return
        if self._rw.scene is None:
            return

        width, height = self._rw.width(), self._rw.height()
        if width == 0 or height == 0:
            return

        painter = self._begin_label_painter(font_size=10, bold=False, color=QColor(240, 240, 240))

        for obj in self._rw.scene.objects:
            if not hasattr(obj, "label") or not hasattr(obj, "label_position"):
                continue

            label_text = self._get_object_label_text(obj)
            if not label_text:
                continue

            world_pos = obj.label_position
            if world_pos is None:
                continue

            x, y, z = world_pos
            screen_x, screen_y, h_val, v_val = self._world_to_screen(
                (x, y, z), h_range, v_range, width, height
            )

            if not (h_range[0] <= h_val <= h_range[1] and v_range[0] <= v_val <= v_range[1]):
                continue

            screen_x += 6
            screen_y -= 6

            painter.drawText(screen_x, screen_y - 10, 140, 20, Qt.AlignLeft | Qt.AlignVCenter, label_text)

        painter.end()
