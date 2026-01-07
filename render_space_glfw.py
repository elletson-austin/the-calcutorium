import numpy as np
import glfw
import moderngl
from dataclasses import dataclass, field
from enum import Enum, auto

class Projection(Enum):
    Perspective = auto()
    Orthographic = auto()

class SnapMode(Enum):
    NONE = auto()
    XY = auto()
    XZ = auto()
    YZ = auto()

class Camera: # TODO It would make sense to have two different camera objects between 2d and 3d visualization.
    
    def __init__(self, 
                 position_center: np.ndarray = None,
                 rotation: np.ndarray = None,
                 fov: float = 60.0, distance: float = 50.0):
        
        if position_center is not None:
            self.position_center = position_center
        else:
            self.position_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        if rotation is not None:
            self.rotation = rotation
        else:
            self.rotation = np.array([90.0, 0.0, 0.0], dtype=np.float32)

        self.fov = fov
        self.distance = distance
        self.projection = Projection.Orthographic
        self.snap_mode = SnapMode.XY
    # Returns the current position of the camera [x, y, z]
    def get_position(self) -> np.ndarray: # Calculate the camera position based on the orientation, distance and center
        
        pitch = np.radians(self.rotation[0])
        yaw = np.radians(self.rotation[1])   

        x = self.distance * np.cos(pitch) * np.sin(yaw)
        y = self.distance * np.sin(pitch)
        z = self.distance * np.cos(pitch) * np.cos(yaw)

        return self.position_center + np.array([x, y, z], dtype=np.float32)

    def get_view_matrix(self) -> np.ndarray:
        cam_pos = self.get_position()
        target = self.position_center
        world_up = np.array([0, 1, 0], dtype=np.float32)
        
        forward = target - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        view = np.eye(4, dtype=np.float32)
        
        # Rotation part
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        
        # Translation part - ALL NEGATIVE
        view[0, 3] = -np.dot(right, cam_pos)
        view[1, 3] = -np.dot(up, cam_pos)
        view[2, 3] = np.dot(forward, cam_pos)  
        
        # Bottom row should be [0, 0, 0, 1] - already set by np.eye()
        
        return view.T.flatten()
    def _perspective_matrix(self, width, height) -> np.ndarray: 
        aspect_ratio = width / height
        fov_rad = np.radians(self.fov)  # Field of view in radians
        near = 1.0     # Near clipping plane (minimum render distance)
        far = 500.0    # Far clipping plane (maximum render distance)
        focal_len = 1.0 / np.tan(fov_rad / 2.0)  # Calculate focal length (controls zoom)

        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = focal_len / aspect_ratio
        proj[1, 1] = focal_len
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2.0 * far * near) / (near - far)
        proj[3, 2] = -1.0
        
        return proj.T.flatten()
        
    def _orthographic_matrix(self, width, height):

        scale = self.distance * 0.1
        aspect = width / height
        left   = -scale * aspect
        right  =  scale * aspect
        bottom = -scale
        top    =  scale
        near   = -1000.0
        far    =  1000.0

        proj = np.eye(4, dtype=np.float32)
        proj[0, 0] = 2 / (right - left)
        proj[1, 1] = 2 / (top - bottom)
        proj[2, 2] = -2 / (far - near)
        proj[0, 3] = -(right + left) / (right - left)
        proj[1, 3] = -(top + bottom) / (top - bottom)
        proj[2, 3] = -(far + near) / (far - near)

        return proj.T.flatten()
    
    def get_projection_matrix(self, width, height):
        
        if self.projection == Projection.Perspective:
            return self._perspective_matrix(width, height)
        else:
            return self._orthographic_matrix(width, height)
    

@dataclass
class WindowState: # Tracks the window state for fullscreen toggling
    is_fullscreen: bool = True
    windowed_pos: tuple = (100, 100)
    windowed_size: tuple = (1280, 800)

@dataclass
class InputState: # Tracks the state of the input
    mouse_pos: np.ndarray = field(
        default_factory=lambda: np.array([0, 0], dtype=np.float32)
    )
    mouse_delta: np.ndarray = field(
        default_factory=lambda: np.array([0, 0], dtype=np.float32)
    )
    keys_held: set = field(default_factory=set)
    mouse_pressed: bool = False
    scroll_delta: float = 0.0


class RenderSpace: # represents the 3d space in which everything is rendered.
    """
    Runtime rendering environment.
    Owns camera, window state, input state, and GPU context.
    Does NOT own scene geometry or simulation data.
    """
    def __init__(self):
        self.cam: Camera = Camera()
        self.window_state: WindowState = WindowState() # Holds the current dimensions of the window
        self.input_state: InputState = InputState() # Holds the current input events
        self.ctx: moderngl.Context = None # Moderngl Context made in glfw_init()
        self.window = None

    def update(self, dt: float = 0.01) -> None:

        keys = self.input_state.keys_held

        if abs(self.input_state.scroll_delta) > 0:
            self.cam.distance -= self.input_state.scroll_delta * 1.0
            self.cam.distance = np.clip(self.cam.distance, 1.0, 300.0)

        if self.input_state.mouse_pressed:
            self.cam.rotation[0] += self.input_state.mouse_delta[1] * 0.2 # Pitch
            self.cam.rotation[1] += self.input_state.mouse_delta[0] * 0.2 # Yaw
            self.cam.rotation[0] = np.clip(self.cam.rotation[0], -90.0, 90.0) # Prevent flipping
            self.cam.rotation[1] = self.cam.rotation[1] % 360.0 # Wrap around

        if glfw.KEY_W in keys:
            forward = self.cam.position_center - self.cam.get_position()
            forward /= np.linalg.norm(forward)
            self.cam.position_center += forward * dt * 20.0

        if glfw.KEY_S in keys:
            forward = self.cam.position_center - self.cam.get_position()
            forward /= np.linalg.norm(forward)
            self.cam.position_center -= forward * dt * 20.0

        if glfw.KEY_A in keys:
            forward = self.cam.position_center - self.cam.get_position()
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, np.array([0, 1, 0], dtype=np.float32))
            right /= np.linalg.norm(right)
            self.cam.position_center -= right * dt * 20.0

        if glfw.KEY_D in keys:
            forward = self.cam.position_center - self.cam.get_position()
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, np.array([0, 1, 0], dtype=np.float32))
            right /= np.linalg.norm(right)
            self.cam.position_center += right * dt * 20.0

        # Reset deltas
        self.input_state.mouse_delta[:] = 0
        self.input_state.scroll_delta = 0

    
def glfw_init(render_space_global: RenderSpace) -> None:  # Initializes GLFW and sets callbacks
    render_space = render_space_global
    title = 'start'
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    glfw.window_hint(glfw.RESIZABLE, glfw.TRUE) # Allow resizing

    
    
    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)
    render_space.window = glfw.create_window(mode.size.width, mode.size.height, title, monitor, None)
    if not render_space.window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(render_space.window)
    glfw.swap_interval(1)

    render_space.ctx = moderngl.create_context()         # Create ModernGL context
    render_space.ctx.enable(moderngl.DEPTH_TEST)
    render_space.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

    # Set GLFW callbacks
    def framebuffer_size_callback(window, width, height) -> None: # Sets the viewport to the current framebuffer size
        if width == 0 or height == 0:
            return
        render_space.ctx.viewport = (0, 0, width, height)

    def key_callback(window, key, scancode, action, mods) -> None:
        """
        Handles key presses:
        - F11: toggle fullscreen
        - ESC: switch to windowed mode with a smaller default size
        """

        # Track all key presses/releases
        if action == glfw.PRESS:
            render_space.input_state.keys_held.add(key)
        elif action == glfw.RELEASE:
            render_space.input_state.keys_held.discard(key)

        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            if render_space.window_state.is_fullscreen:
                # Set default windowed position and size
                render_space.window_state.windowed_pos = (100, 100)
                render_space.window_state.windowed_size = (1280, 800)
                glfw.set_window_monitor(
                    window,
                    None,             # windowed mode
                    render_space.window_state.windowed_pos[0],
                    render_space.window_state.windowed_pos[1],
                    render_space.window_state.windowed_size[0],
                    render_space.window_state.windowed_size[1],
                    0
                )
                render_space.window_state.is_fullscreen = False

        if key == glfw.KEY_TAB and action == glfw.PRESS:
            if render_space.cam.projection == Projection.Orthographic:
                render_space.cam.projection = Projection.Perspective
            else:
                render_space.cam.projection = Projection.Orthographic

    def make_mouse_callback(cam) -> None:
        def mouse_callback(window, button, action, mods):
            if button == glfw.MOUSE_BUTTON_1 and action == glfw.PRESS:
                render_space.input_state.mouse_pressed = True
            elif button == glfw.MOUSE_BUTTON_1 and action == glfw.RELEASE:
                render_space.input_state.mouse_pressed = False
        return mouse_callback
    
    def make_scroll_callback(cam) -> None:
        def scroll_callback(window, xoffset, yoffset):
            render_space.input_state.scroll_delta += yoffset
        return scroll_callback

    def make_cursor_pos_callback(cam) -> None:
        def cursor_pos_callback(window, xpos, ypos):
            render_space.input_state.mouse_delta[0] += xpos - render_space.input_state.mouse_pos[0] # Update mouse position and delta
            render_space.input_state.mouse_delta[1] += ypos - render_space.input_state.mouse_pos[1] # Update mouse position and delta
            render_space.input_state.mouse_pos[:] = (xpos, ypos)
        return cursor_pos_callback
    
    glfw.set_framebuffer_size_callback(render_space.window, framebuffer_size_callback)
    glfw.set_key_callback(render_space.window, key_callback)
    glfw.set_mouse_button_callback(render_space.window, make_mouse_callback(render_space.cam))    # I use this format to keep cam within the scope of the callback
    glfw.set_scroll_callback(render_space.window, make_scroll_callback(render_space.cam))
    glfw.set_cursor_pos_callback(render_space.window, make_cursor_pos_callback(render_space.cam)) 

    # Set initial viewport
    width, height = glfw.get_framebuffer_size(render_space.window)
    render_space.ctx.viewport = (0, 0, width, height)
