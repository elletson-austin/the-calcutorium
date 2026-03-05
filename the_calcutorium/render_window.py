from typing import Any

import moderngl
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from .camera import Camera2D, Camera3D, InputState
from .overlay_labels import OverlayLabelRenderer
from .render_types import Projection, SnapMode
from .renderer import Renderer
from .scene import Grid, MathFunction, Scene, SceneObject


class RenderWindow(QOpenGLWidget):
    manual_range_cleared = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera: Camera3D | Camera2D | None = Camera3D()
        self.input_state: InputState = InputState()
        self.ctx: moderngl.Context
        self.scene: Scene
        self.framebuffer: moderngl.Framebuffer
        self.renderer: Renderer
        self.mouse_hovering: bool = False
        self._manual_ranges: dict[str, tuple[float, float]] = {}

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)  # ~60 FPS
        self.render_objects: dict = {}
        self.overlay_labels: OverlayLabelRenderer | None = OverlayLabelRenderer(self)

    def get_manual_ranges(self) -> dict[str, tuple[float, float]]:
        return self._manual_ranges.copy()

    def set_manual_range(self, axis: str, min_val: float, max_val: float) -> None:
        self._manual_ranges[axis] = (min_val, max_val)
        self.update()

    def clear_manual_ranges(self) -> None:
        if self._manual_ranges:
            self._manual_ranges.clear()
            self.manual_range_cleared.emit()
            self.update()

    def has_manual_range(self, axis: str) -> bool:
        return axis in self._manual_ranges

    def set_camera(self, camera: Camera3D | Camera2D) -> None:
        self.camera = camera
        self.update()

    def set_scene(self, scene: Scene) -> None:
        self.scene = scene

    def initializeGL(self) -> None:
        self.makeCurrent()
        self.ctx = moderngl.create_context()
        self.ctx.enable(
            moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND
        )
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.renderer = Renderer(self.ctx)
        self.framebuffer = self.ctx.detect_framebuffer()

    def resizeGL(self, w: int, h: int) -> None:
        if self.ctx:
            self.ctx.viewport = (0, 0, w, h)
            self.framebuffer = self.ctx.detect_framebuffer()

    def _update_2d_rendering_params(
        self, width: int, height: int
    ) -> tuple[float, float]:
        h_proj_range, v_proj_range = None, None
        cam2d = self.camera
        snap_map = {
            SnapMode.XY: ("x", "y", "xy"),
            SnapMode.XZ: ("x", "z", "xz"),
            SnapMode.YZ: ("z", "y", "yz"),
        }
        h_axis, v_axis, plane_str = snap_map.get(cam2d.snap_mode, (None, None, None))

        if h_axis and v_axis:
            dynamic_h_range, dynamic_v_range = self._compute_2d_dynamic_ranges(
                h_axis, v_axis, width, height
            )

            if dynamic_h_range and dynamic_v_range:
                self._apply_dynamic_ranges_to_scene(
                    plane_str, dynamic_h_range, dynamic_v_range
                )

            # If manual ranges are present, return them for use by the renderer
            if self.has_manual_range(h_axis) and self.has_manual_range(v_axis):
                h_proj_range = self._manual_ranges[h_axis]
                v_proj_range = self._manual_ranges[v_axis]

        return h_proj_range, v_proj_range

    def _compute_2d_dynamic_ranges(
        self, h_axis: str, v_axis: str, width: int, height: int
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Compute dynamic horizontal and vertical ranges for 2D camera view.
        Returns a tuple (h_range, v_range) or (None, None) if not computable.
        """
        if self.has_manual_range(h_axis) and self.has_manual_range(v_axis):
            return self._manual_ranges[h_axis], self._manual_ranges[v_axis]

        if height <= 0:
            return ((0, 0), (0, 0))

        cam2d = self.camera
        aspect = width / height
        view_height = cam2d.distance
        view_width = view_height * aspect
        axis_map = {"x": 0, "y": 1, "z": 2}
        h_center = cam2d.position_center[axis_map[h_axis]]
        v_center = cam2d.position_center[axis_map[v_axis]]
        h_min, h_max = h_center - (view_width / 2), h_center + (view_width / 2)
        v_min, v_max = v_center - (view_height / 2), v_center + (view_height / 2)
        return (h_min, h_max), (v_min, v_max)

    def _apply_dynamic_ranges_to_scene(
        self, plane_str: str, h_range: tuple[float, float], v_range: tuple[float, float]
    ) -> None:
        for obj in self.scene.objects:
            if isinstance(obj, Grid):
                obj.update(h_range=h_range, v_range=v_range, plane=plane_str)
            elif isinstance(obj, MathFunction):
                obj.update(plane=plane_str, h_range=h_range, v_range=v_range)

    def paintGL(self) -> None:  # This is the corrected and single definition of paintGL
        self.makeCurrent()
        if not all([self.ctx, self.scene, self.renderer, self.framebuffer]):
            return

        self.framebuffer.use()
        self.update_camera(0.016)  # Assuming 60fps for now

        self.framebuffer.clear(0.1, 0.1, 0.1, 1.0)

        width, height = self.width(), self.height()
        if width == 0 or height == 0:
            return

        h_proj_range, v_proj_range = None, None

        if isinstance(self.camera, Camera2D):
            h_proj_range, v_proj_range = self._update_2d_rendering_params(width, height)
        else:  # 3D Camera
            for obj in self.scene.objects:
                if isinstance(obj, Grid):
                    obj.update()

        if isinstance(self.camera, Camera2D):
            filtered_scene_objs = {
                obj
                for obj in self.scene.objects
                if obj.is_2d and hasattr(obj, "vertices") and obj.vertices.size > 0
            }
        else:  # Camera3D
            filtered_scene_objs = {
                obj
                for obj in self.scene.objects
                if hasattr(obj, "vertices") and obj.vertices.size > 0
            }

        self._manage_render_objects(filtered_scene_objs)

        self.renderer.render(
            list[Any](self.render_objects.values()),
            self.camera,
            width,
            height,
            h_range=h_proj_range,
            v_range=v_proj_range,
        )

        if isinstance(self.camera, Camera2D):
            self.overlay_labels.render_grid_labels(h_proj_range, v_proj_range)
            self.overlay_labels.render_function_labels(h_proj_range, v_proj_range)
            self.overlay_labels.render_object_labels(h_proj_range, v_proj_range)

    def _manage_render_objects(self, filtered_scene_objs: set[SceneObject]) -> None:
        current_render_obj_keys = set[Any](self.render_objects.keys())

        # Add or update render objects
        for obj in filtered_scene_objs:
            self._add_or_update_render_object(obj)

        # Remove render objects that are no longer in the filtered set
        for obj in current_render_obj_keys - filtered_scene_objs:
            self._remove_render_object_if_present(obj)

    def _add_or_update_render_object(self, obj: SceneObject) -> None:
        if getattr(obj, "is_dirty", False) and obj in self.render_objects:
            self.render_objects.pop(obj).release()
            obj.is_dirty = False
        if obj not in self.render_objects:
            self.render_objects[obj] = self.renderer.create_render_object(obj)

    def _remove_render_object_if_present(self, obj: SceneObject) -> None:
        if obj in self.render_objects:
            self.render_objects.pop(obj).release()

    def closeEvent(self, event) -> None:
        for ro in self.render_objects.values():
            ro.release()
        super().closeEvent(event)

    def update_camera(self, dt: float = 0.016) -> None:
        if (
            self.input_state.left_mouse_pressed
            or abs(self.input_state.scroll_delta) > 0
        ):
            self.clear_manual_ranges()

        self.camera.update(self.input_state, dt, self.width(), self.height())

        self.input_state.mouse_delta[:] = 0
        self.input_state.scroll_delta = 0

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse button press events."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.input_state.left_mouse_pressed = True
        elif event.button() == Qt.MouseButton.RightButton:
            self.input_state.right_mouse_pressed = True
        self.input_state.mouse_pos[:] = (event.position().x(), event.position().y())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse button release events."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.input_state.left_mouse_pressed = False
        elif event.button() == Qt.MouseButton.RightButton:
            self.input_state.right_mouse_pressed = False

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse movement events."""
        new_pos = np.array(
            [event.position().x(), event.position().y()], dtype=np.float32
        )
        self.input_state.mouse_delta += new_pos - self.input_state.mouse_pos
        self.input_state.mouse_pos = new_pos

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse wheel scroll events for zooming."""
        self.input_state.scroll_delta = event.angleDelta().y() / 120.0

    def enterEvent(self, event) -> None:
        """Handle mouse entering the widget."""
        self.mouse_hovering = True
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        """Handle mouse leaving the widget."""
        self.mouse_hovering = False
        super().leaveEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle key press events."""
        self.input_state.keys_held.add(event.key())

        if (
            event.key() == Qt.Key.Key_Tab
            and self.mouse_hovering
            and isinstance(self.camera, Camera3D)
        ):
            self.camera.projection = (
                Projection.Orthographic
                if self.camera.projection == Projection.Perspective
                else Projection.Perspective
            )
            event.accept()
            return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        """Handle key release events."""
        self.input_state.keys_held.discard(event.key())
        super().keyReleaseEvent(event)
