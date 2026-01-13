import moderngl
import numpy as np
from PySide6.QtOpenGLWidgets import QOpenGLWidget   
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QMouseEvent, QWheelEvent, QKeyEvent

from renderer import Renderer
from camera import Camera, InputState, Projection, CameraMode


class PySideRenderSpace(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cam: Camera = Camera()
        self.input_state: InputState = InputState()
        self.ctx: moderngl.Context = None
        self.scene = None
        self.screen = None
        self.renderer: Renderer = None
        self.mouse_hovering: bool = False # Track mouse hover state

        # Set a strong focus policy to receive keyboard events.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Enable mouse tracking to receive enter/leave events
        self.setMouseTracking(True)
        # Timer to trigger continuous updates for a smooth render loop
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16) # Approximately 60 FPS
        self.render_objects: dict = {}  # Map SceneObjects to their RenderObjects


    def set_scene(self, scene):
        self.scene = scene


    def initializeGL(self):
        self.makeCurrent()
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.renderer = Renderer(self.ctx)
        self.screen = self.ctx.detect_framebuffer()


    def resizeGL(self, w: int, h: int):
        if self.ctx:
            self.ctx.viewport = (0, 0, w, h)
            self.screen = self.ctx.detect_framebuffer()


    def paintGL(self):
        self.makeCurrent()
        if not self.ctx or not self.scene or not self.renderer or not self.screen:
            return

        self.screen.use()
        self.update_camera()

        # Clear the screen with a dark color
        self.screen.clear(0.0, 0.2, 0.2, 1.0)
        
        width, height = self.width(), self.height()
        if width == 0 or height == 0:
            return

        # Create render objects for any new scene objects and update existing ones
        for obj in self.scene.objects:
            if hasattr(obj, 'vertices') and obj.vertices.size > 0:
                if obj not in self.render_objects:
                    self.render_objects[obj] = self.renderer.create_render_object(obj)
                elif getattr(obj, 'is_dirty', False):
                    self.renderer.update_render_object(self.render_objects[obj], obj)
                    obj.is_dirty = False
            else:
                if obj in self.render_objects:
                    ro = self.render_objects.pop(obj)
                    ro.release()

        # Handle object removal from the scene
        removed_objects = [obj for obj in self.render_objects if obj not in self.scene.objects]
        for obj in removed_objects:
            ro = self.render_objects.pop(obj)
            ro.release()

        # Tell the renderer to draw the objects
        self.renderer.render(list(self.render_objects.values()), self.cam, width, height)


    def closeEvent(self, event):
        for ro in self.render_objects.values():
            ro.release()
        super().closeEvent(event)


    def update_camera(self, dt: float = 0.016): # dt is roughly 1/60th of a second
        """Updates the camera's position and orientation based on user input."""
        keys = self.input_state.keys_held

        if self.cam.mode == CameraMode.TwoD:
            # --- Mouse Panning (2D Mode) ---
            if self.input_state.left_mouse_pressed:
                # Adjust sensitivity based on distance (zoom level)
                pan_speed = self.cam.distance / self.height() * 2
                self.cam.position_center[0] -= self.input_state.mouse_delta[0] * pan_speed
                self.cam.position_center[1] += self.input_state.mouse_delta[1] * pan_speed
        else:
            # --- Mouse Rotation (3D Mode) ---
            if self.input_state.left_mouse_pressed:
                self.cam.rotation[0] += self.input_state.mouse_delta[1] * 2.0 # Pitch
                self.cam.rotation[1] += self.input_state.mouse_delta[0] * 2.0 # Yaw
                self.cam.rotation[0] = np.clip(self.cam.rotation[0], -89.0, 89.0) # Clamp pitch
                self.cam.rotation[1] = self.cam.rotation[1] % 360.0 # Wrap yaw

        # --- Mouse Zoom (Shared) ---
        if abs(self.input_state.scroll_delta) > 0:
            self.cam.distance -= self.input_state.scroll_delta * 1.0
            self.cam.distance = np.clip(self.cam.distance, 1.0, 300.0)

        # --- Keyboard Movement (WASD, 3D Mode Only) ---
        if self.cam.mode == CameraMode.ThreeD:
            if Qt.Key.Key_W in keys:
                # Move camera center forward
                forward = self.cam.position_center - self.cam.get_position()
                forward[1] = 0  # Project onto XZ plane
                if np.linalg.norm(forward) > 0.1:
                    forward = forward / np.linalg.norm(forward)
                    self.cam.position_center += forward * dt * 20.0

            if Qt.Key.Key_S in keys:
                # Move camera center backward
                forward = self.cam.position_center - self.cam.get_position()
                forward[1] = 0 # Project onto XZ plane
                if np.linalg.norm(forward) > 0.1:
                    forward = forward / np.linalg.norm(forward)
                    self.cam.position_center -= forward * dt * 20.0

            if Qt.Key.Key_A in keys:
                # Move camera center left
                forward = self.cam.position_center - self.cam.get_position()
                right = np.cross(forward, np.array([0, 1, 0], dtype=np.float32))
                right[1] = 0 # Project onto XZ plane
                if np.linalg.norm(right) > 0.1:
                    right = right / np.linalg.norm(right)
                    self.cam.position_center -= right * dt * 20.0

            if Qt.Key.Key_D in keys:
                # Move camera center right
                forward = self.cam.position_center - self.cam.get_position()
                right = np.cross(forward, np.array([0, 1, 0], dtype=np.float32))
                right[1] = 0 # Project onto XZ plane
                if np.linalg.norm(right) > 0.1:
                    right = right / np.linalg.norm(right)
                    self.cam.position_center += right * dt * 20.0

        # Reset mouse and scroll deltas after they've been processed
        self.input_state.mouse_delta[:] = 0
        self.input_state.scroll_delta = 0


    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse button press events."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.input_state.left_mouse_pressed = True
        elif event.button() == Qt.MouseButton.RightButton:
            self.input_state.right_mouse_pressed = True
        # Record the current mouse position
        self.input_state.mouse_pos[:] = (event.position().x(), event.position().y())


    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse button release events."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.input_state.left_mouse_pressed = False
        elif event.button() == Qt.MouseButton.RightButton:
            self.input_state.right_mouse_pressed = False
        # No need to update mouse_pos here, it will be handled by mouseMoveEvent or next mousePressEvent


    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse movement events."""
        new_pos = np.array([event.position().x(), event.position().y()], dtype=np.float32)
        self.input_state.mouse_delta = new_pos - self.input_state.mouse_pos
        self.input_state.mouse_pos = new_pos


    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel scroll events for zooming."""
        # PySide6's angleDelta().y() is usually in 1/8ths of a degree, so 120 is a standard wheel click
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
        
        # Toggle projection mode with Tab key only if mouse is hovering
        if event.key() == Qt.Key.Key_Tab and self.mouse_hovering:
            if self.cam.projection == Projection.Orthographic:
                self.cam.projection = Projection.Perspective
            else:
                self.cam.projection = Projection.Orthographic
        
        # Toggle 2D/3D mode with M key
        if event.key() == Qt.Key.Key_M:
            if self.cam.mode == CameraMode.ThreeD:
                self.cam.mode = CameraMode.TwoD
                self.cam.projection = Projection.Orthographic
                print("Switched to 2D Mode")
            else:
                self.cam.mode = CameraMode.ThreeD
                print("Switched to 3D Mode")

        super().keyPressEvent(event)


    def keyReleaseEvent(self, event: QKeyEvent):
        """Handle key release events."""
        self.input_state.keys_held.discard(event.key())