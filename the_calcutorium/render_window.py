import moderngl
import numpy as np
from PySide6.QtOpenGLWidgets import QOpenGLWidget   
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QMouseEvent, QWheelEvent, QKeyEvent

from .scene import SceneObject, MathFunction, Grid
from .render_types import SnapMode, Projection
from .camera import Camera3D, Camera2D, InputState
from .renderer import Renderer

class RenderWindow(QOpenGLWidget):
    manual_range_cleared = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera: Camera3D | Camera2D = Camera3D()
        self.input_state: InputState = InputState()
        self.ctx: moderngl.Context = None
        self.scene = None
        self.screen = None
        self.renderer: Renderer = None
        self.mouse_hovering: bool = False
        self.manual_ranges = {}

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16) # ~60 FPS
        self.render_objects: dict = {}

    def set_camera(self, camera):
        self.camera = camera
        self.update()

    def set_scene(self, scene):
        self.scene = scene

    def initializeGL(self):
        self.makeCurrent()
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.renderer = Renderer(self.ctx)
        self.screen = self.ctx.detect_framebuffer()

    def resizeGL(self, w: int, h: int):
        if self.ctx:
            self.ctx.viewport = (0, 0, w, h)
            self.screen = self.ctx.detect_framebuffer()

    def _update_2d_rendering_params(self, width: int, height: int):
        h_proj_range, v_proj_range = None, None
        cam2d = self.camera
        snap_map = {SnapMode.XY: ('x', 'y', 'xy'), SnapMode.XZ: ('x', 'z', 'xz'), SnapMode.YZ: ('z', 'y', 'yz')}
        h_axis, v_axis, plane_str = snap_map.get(cam2d.snap_mode, (None, None, None))

        if h_axis and v_axis:
            dynamic_h_range, dynamic_v_range = None, None
            if h_axis in self.manual_ranges and v_axis in self.manual_ranges:
                h_proj_range = self.manual_ranges[h_axis]
                v_proj_range = self.manual_ranges[v_axis]
                dynamic_h_range, dynamic_v_range = h_proj_range, v_proj_range
            elif height > 0:
                aspect = width / height
                view_height = cam2d.distance
                view_width = view_height * aspect
                buffer = 1.05
                axis_map = {'x': 0, 'y': 1, 'z': 2}
                h_center, v_center = cam2d.position_center[axis_map[h_axis]], cam2d.position_center[axis_map[v_axis]]
                h_min, h_max = h_center - (view_width / 2), h_center + (view_width / 2)
                v_min, v_max = v_center - (view_height / 2), v_center + (view_height / 2)
                dynamic_h_range, dynamic_v_range = (h_min, h_max), (v_min, v_max)

            if dynamic_h_range and dynamic_v_range:
                for obj in self.scene.objects:
                    if isinstance(obj, Grid): obj.set_ranges(dynamic_h_range, dynamic_v_range, plane_str)
                    elif isinstance(obj, MathFunction): obj.update_for_plane(plane_str, dynamic_h_range, dynamic_v_range)
        return h_proj_range, v_proj_range

    def paintGL(self): # This is the corrected and single definition of paintGL
        self.makeCurrent()
        if not all([self.ctx, self.scene, self.renderer, self.screen]):
            return

        self.screen.use()
        self.update_camera(0.016) # Assuming 60fps for now

        self.screen.clear(0.0, 0.2, 0.2, 1.0)
        
        width, height = self.width(), self.height()
        if width == 0 or height == 0:
            return

        h_proj_range, v_proj_range = None, None

        if isinstance(self.camera, Camera2D):
            h_proj_range, v_proj_range = self._update_2d_rendering_params(width, height)
        else: # 3D Camera
            for obj in self.scene.objects:
                if isinstance(obj, Grid): obj.set_to_default()

        # Determine the set of scene objects to be rendered based on camera type
        if isinstance(self.camera, Camera2D):
            # In 2D camera mode, only render objects explicitly marked as 2D
            filtered_scene_objs = {obj for obj in self.scene.objects if obj.Is2d and hasattr(obj, 'vertices') and obj.vertices.size > 0}
        else: # Camera3D
            # In 3D camera mode, render all objects that have vertices (both 2D and 3D)
            filtered_scene_objs = {obj for obj in self.scene.objects if hasattr(obj, 'vertices') and obj.vertices.size > 0}
        
        self._manage_render_objects(filtered_scene_objs)

        self.renderer.render(list(self.render_objects.values()), self.camera, width, height, h_range=h_proj_range, v_range=v_proj_range)


    def _manage_render_objects(self, filtered_scene_objs: set):
        current_render_obj_keys = set(self.render_objects.keys())

        # Add or update render objects
        for obj in filtered_scene_objs:
            if getattr(obj, 'is_dirty', False) and obj in self.render_objects:
                self.render_objects.pop(obj).release()
                obj.is_dirty = False
            if obj not in self.render_objects:
                self.render_objects[obj] = self.renderer.create_render_object(obj)

        # Remove render objects that are no longer in the filtered set
        for obj in current_render_obj_keys - filtered_scene_objs:
            if obj in self.render_objects:
                self.render_objects.pop(obj).release()


    def closeEvent(self, event):
        for ro in self.render_objects.values():
            ro.release()
        super().closeEvent(event)


    def update_camera(self, dt: float = 0.016):
        if self.input_state.left_mouse_pressed or abs(self.input_state.scroll_delta) > 0:
            if self.manual_ranges:
                self.manual_ranges.clear()
                self.manual_range_cleared.emit()

        self.camera.update(self.input_state, dt, self.width(), self.height())

        self.input_state.mouse_delta[:] = 0
        self.input_state.scroll_delta = 0


    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse button press events."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.input_state.left_mouse_pressed = True
        elif event.button() == Qt.MouseButton.RightButton:
            self.input_state.right_mouse_pressed = True
        self.input_state.mouse_pos[:] = (event.position().x(), event.position().y())


    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse button release events."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.input_state.left_mouse_pressed = False
        elif event.button() == Qt.MouseButton.RightButton:
            self.input_state.right_mouse_pressed = False


    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse movement events."""
        new_pos = np.array([event.position().x(), event.position().y()], dtype=np.float32)
        self.input_state.mouse_delta += new_pos - self.input_state.mouse_pos
        self.input_state.mouse_pos = new_pos


    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel scroll events for zooming."""
        self.input_state.scroll_delta = event.angleDelta().y() / 120.0


    def enterEvent(self, event):
        """Handle mouse entering the widget."""
        self.mouse_hovering = True
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Handle mouse leaving the widget."""
        self.mouse_hovering = False
        super().leaveEvent(event)


    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events."""
        self.input_state.keys_held.add(event.key())
        
        if event.key() == Qt.Key.Key_Tab and self.mouse_hovering and isinstance(self.camera, Camera3D):
            self.camera.projection = Projection.Orthographic if self.camera.projection == Projection.Perspective else Projection.Perspective
            event.accept()
            return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        """Handle key release events."""
        self.input_state.keys_held.discard(event.key())
        super().keyReleaseEvent(event)