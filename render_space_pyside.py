import moderngl
import numpy as np
from PySide6.QtOpenGLWidgets import QOpenGLWidget   
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QMouseEvent, QWheelEvent, QKeyEvent

from renderer import Renderer
from camera import Camera, InputState, Projection


class PySideRenderSpace(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cam: Camera = Camera()
        self.input_state: InputState = InputState()
        self.ctx: moderngl.Context = None
        self.scene = None
        self.screen = None # Add screen attribute

        # Set a strong focus policy to receive keyboard events.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Timer to trigger continuous updates for a smooth render loop
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)


    def set_scene(self, scene):
        self.scene = scene


    def initializeGL(self):
        self.ctx = None
        self.update()


    def resizeGL(self, w: int, h: int):
        if self.ctx:
            self.ctx.viewport = (0, 0, w, h)


    def paintGL(self):
        self.makeCurrent()
        if self.ctx is None:
            self.ctx = moderngl.create_context()
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
            self.renderer = Renderer(self.ctx)
            self.screen = self.ctx.detect_framebuffer() # Explicitly detect framebuffer

        if not self.ctx or not self.scene or not self.renderer or not self.screen:
            return

        self.screen.use() # Ensure we are drawing to the widget's framebuffer
        self.update_camera()

        # Clear the screen with a dark color
        self.screen.clear(0.0, 0.2, 0.2, 1.0) # Use self.screen.clear
        
        width, height = self.width(), self.height()
        if width == 0 or height == 0:
            return

        # NOTE: This part is simple but inefficient. We are creating new GPU buffers
        # every single frame. This is fine for a simple scene, but we will
        # optimize this later by caching the render objects.
        render_objects = []
        for obj in self.scene.objects:
            ro = self.renderer.create_render_object(obj)
            render_objects.append(ro)

        # Tell the renderer to draw the objects
        self.renderer.render(render_objects, self.cam, width, height)

        # Clean up the temporary render objects
        for ro in render_objects:
            ro.release()


    def update_camera(self, dt: float = 0.016): # dt is roughly 1/60th of a second
        """Updates the camera's position and orientation based on user input."""
        keys = self.input_state.keys_held

        # --- Mouse Rotation ---
        if self.input_state.left_mouse_pressed:
            self.cam.rotation[0] += self.input_state.mouse_delta[1] * 2.0 # Pitch
            self.cam.rotation[1] += self.input_state.mouse_delta[0] * 2.0 # Yaw
            self.cam.rotation[0] = np.clip(self.cam.rotation[0], -90.0, 90.0) # Clamp pitch
            self.cam.rotation[1] = self.cam.rotation[1] % 360.0 # Wrap yaw

        # --- Mouse Zoom ---
        if abs(self.input_state.scroll_delta) > 0:
            self.cam.distance -= self.input_state.scroll_delta * 1.0
            self.cam.distance = np.clip(self.cam.distance, 1.0, 300.0)

        # --- Keyboard Movement (WASD) ---
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
        print('wheel scrolled')


    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events."""
        self.input_state.keys_held.add(event.key())
        
        # Toggle projection mode with Tab key
        if event.key() == Qt.Key.Key_Tab:
            if self.cam.projection == Projection.Orthographic:
                self.cam.projection = Projection.Perspective
            else:
                self.cam.projection = Projection.Orthographic


    def keyReleaseEvent(self, event: QKeyEvent):
        """Handle key release events."""
        self.input_state.keys_held.discard(event.key())