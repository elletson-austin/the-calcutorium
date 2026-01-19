import moderngl
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from PySide6.QtOpenGLWidgets import QOpenGLWidget   
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QMouseEvent, QWheelEvent, QKeyEvent
from scene import SceneObject, LorenzAttractor, ProgramID, Mode, MathFunction, Grid


from render_types import Projection, SnapMode, CameraMode

@dataclass
class InputState: # Tracks the state of the input
    mouse_pos: np.ndarray = field(
        default_factory=lambda: np.array([0, 0], dtype=np.float32)
    )
    mouse_delta: np.ndarray = field(
        default_factory=lambda: np.array([0, 0], dtype=np.float32)
    )
    keys_held: set = field(default_factory=set)
    left_mouse_pressed: bool = False
    right_mouse_pressed: bool = False
    scroll_delta: float = 0.0


class Camera: 
    
    def __init__(self, 
                 position_center: np.ndarray = None,
                 rotation: np.ndarray = None,
                 fov: float = 60.0, distance: float = 50.0,
                 mode: CameraMode = CameraMode.ThreeD):
        
        if position_center is not None:
            self.position_center = position_center
        else:
            self.position_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        if rotation is not None:
            self.rotation = rotation
        else:
            self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.fov = fov
        self.distance = distance
        self.projection = Projection.Orthographic
        self.snap_mode = SnapMode.XY
        self.mode = mode

    # Calculate the camera position based on the orientation, distance and center
    def get_position(self) -> np.ndarray: 
        if self.mode == CameraMode.TwoD:
            pos = self.position_center.copy()
            if self.snap_mode == SnapMode.XY:
                pos[2] += self.distance
            elif self.snap_mode == SnapMode.XZ:
                pos[1] += self.distance
            elif self.snap_mode == SnapMode.YZ:
                pos[0] += self.distance
            return pos
        
        pitch = np.radians(self.rotation[0])
        yaw = np.radians(self.rotation[1])

        x = self.distance * np.cos(pitch) * np.sin(yaw)
        y = self.distance * np.sin(pitch)
        z = self.distance * np.cos(pitch) * np.cos(yaw)

        return self.position_center + np.array([x, y, z], dtype=np.float32)


    def get_view_matrix(self) -> np.ndarray:
        cam_pos = self.get_position()
        target = self.position_center
        
        if self.mode == CameraMode.TwoD:
            if self.snap_mode == SnapMode.XY:
                world_up = np.array([0, 1, 0], dtype=np.float32)
            elif self.snap_mode == SnapMode.XZ:
                world_up = np.array([0, 0, 1], dtype=np.float32)
            elif self.snap_mode == SnapMode.YZ:
                world_up = np.array([0, 1, 0], dtype=np.float32)
        else:
            world_up = np.array([0, 1, 0], dtype=np.float32)
        
        forward = target - cam_pos
        # Normalize forward vector, handling the case where it's zero.
        norm_forward = np.linalg.norm(forward)
        if norm_forward > 1e-6:
            forward /= norm_forward
        
        right = np.cross(forward, world_up)
        # Normalize right vector
        norm_right = np.linalg.norm(right)
        if norm_right > 1e-6:
            right /= norm_right
        
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
        if height == 0:
            aspect_ratio = 1.0
        else:
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
        

    def _orthographic_matrix(self, width, height, h_range=None, v_range=None):
        if self.mode == CameraMode.TwoD and h_range is not None and v_range is not None:
            left, right = h_range
            bottom, top = v_range
        else:
            if height == 0:
                aspect = 1.0
            else:
                aspect = width / height
            
            view_height = self.distance
            top = view_height / 2.0
            bottom = -view_height / 2.0
            right = top * aspect
            left = -top * aspect

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
    

    def get_projection_matrix(self, width, height, h_range=None, v_range=None):
        
        if self.projection == Projection.Perspective:
            return self._perspective_matrix(width, height)
        else:
            return self._orthographic_matrix(width, height, h_range=h_range, v_range=v_range)


class ProgramManager: # holds and stores programs that draw points, lines, etc.

    def __init__(self, ctx: moderngl.Context):
        self.programs: dict[ProgramID, moderngl.Program] = {}
        self.compute_shaders: dict[ProgramID, moderngl.ComputeShader] = {}
        self.ctx = ctx

    
    def basic_3d_src(self):
        VERTEX_SOURCE = """
        #version 330

        layout (location = 0) in vec3 in_position;
        layout (location = 1) in vec3 in_color;

        uniform mat4 u_view;
        uniform mat4 u_proj;

        out vec3 v_color;

        void main() {
        gl_Position = u_proj * u_view * vec4(in_position, 1.0);
        gl_PointSize = 2.0;
        v_color = in_color;
        }
        """

        FRAGMENT_SOURCE = """
        #version 330

        in vec3 v_color;
        out vec4 fragColor;

        void main() {
        fragColor = vec4(v_color, 1.0);
        }
        """
        return VERTEX_SOURCE, FRAGMENT_SOURCE

    def grid_src(self):
        VERTEX_SOURCE = """
        #version 330

        layout (location = 0) in vec3 in_position;
        layout (location = 1) in vec3 in_color;

        uniform mat4 u_view;
        uniform mat4 u_proj;

        out vec3 v_color;

        void main() {
        gl_Position = u_proj * u_view * vec4(in_position, 1.0);
        gl_PointSize = 2.0;
        v_color = in_color;
        }
        """

        FRAGMENT_SOURCE = """
        #version 330

        in vec3 v_color;
        out vec4 fragColor;
        uniform float u_alpha_multiplier;

        void main() {
            fragColor = vec4(v_color, 1.0 * u_alpha_multiplier);
        }
        """
        return VERTEX_SOURCE, FRAGMENT_SOURCE

    def surface_src(self):
        VERTEX_SHADER = """
            #version 330

            layout (location = 0) in vec3 in_position;
            layout (location = 1) in vec3 in_normal;
            layout (location = 2) in vec3 in_color;

            uniform mat4 u_view;
            uniform mat4 u_proj;
            uniform mat4 u_model;

            out vec3 v_normal;
            out vec3 v_pos;
            out vec3 v_color;

            void main() {
                gl_Position = u_proj * u_view * u_model * vec4(in_position, 1.0);
                v_pos = (u_model * vec4(in_position, 1.0)).xyz;
                v_normal = mat3(transpose(inverse(u_model))) * in_normal;
                v_color = in_color;
            }
        """
        FRAGMENT_SHADER = """
            #version 330

            in vec3 v_normal;
            in vec3 v_pos;
            in vec3 v_color;

            out vec4 fragColor;

            uniform vec3 u_light_pos;
            uniform vec3 u_view_pos;

            void main() {
                vec3 norm = normalize(v_normal);
                vec3 light_dir = normalize(u_light_pos - v_pos);
                
                // Ambient
                float ambient_strength = 0.2;
                vec3 ambient = ambient_strength * vec3(1.0, 1.0, 1.0);
                
                // Diffuse
                float diff = max(dot(norm, light_dir), 0.0);
                vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);
                
                // Specular
                float specular_strength = 0.4;
                vec3 view_dir = normalize(u_view_pos - v_pos);
                vec3 reflect_dir = reflect(-light_dir, norm);
                float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
                vec3 specular = specular_strength * spec * vec3(1.0, 1.0, 1.0);
                
                vec3 result = (ambient + diffuse + specular) * v_color;
                fragColor = vec4(result, 1.0);
            }
        """
        return VERTEX_SHADER, FRAGMENT_SHADER
    
    def lorenz_attractor_src(self):
        VERTEX_SHADER = """
        #version 330

        in vec4 in_position;

        uniform mat4 u_view;
        uniform mat4 u_proj;

        out vec3 frag_pos;

        void main() {
            frag_pos = in_position.xyz;
            gl_Position = u_proj * u_view * vec4(in_position.xyz, 1.0);
            gl_PointSize = 0.5; 
        }
        """
        FRAGMENT_SHADER = """
        #version 330

        in vec3 frag_pos;
        out vec4 fragColor;

        void main() {
            fragColor = vec4(1.0, 0.2, 0.2, 1.0);
        }
        """
        return VERTEX_SHADER, FRAGMENT_SHADER
    
    def lorenz_attractor_compute_src(self):
        COMPUTE_SHADER = """
        #version 430

        layout(local_size_x = 256) in;

        layout(std430, binding = 0) buffer PointsBuffer {
            vec4 points[];
        };

        uniform float dt;
        uniform float sigma;
        uniform float rho;
        uniform float beta;
        uniform int steps;

        void main() {
            uint idx = gl_GlobalInvocationID.x;
            if (idx >= points.length()) return;
            
            vec3 p = points[idx].xyz;
            
            for (int i = 0; i < steps; i++) {
                float dx = sigma * (p.y - p.x);
                float dy = p.x * (rho - p.z) - p.y;
                float dz = p.x * p.y - beta * p.z;
                
                p.x += dx * dt;
                p.y += dy * dt;
                p.z += dz * dt;
            }
            
            points[idx].xyz = p;
        }
        """
        return COMPUTE_SHADER
    
    def build_compute_shader(self, program_id) -> moderngl.ComputeShader:
        if program_id in self.compute_shaders:
            return self.compute_shaders[program_id]

        if program_id == ProgramID.LORENZ_ATTRACTOR:
            COMPUTE_SOURCE = self.lorenz_attractor_compute_src()
        else:
            print('no valid compute shader source code available') 
            return 
        
        compute_shader = self.ctx.compute_shader(COMPUTE_SOURCE)
        
        self.compute_shaders[program_id] = compute_shader
        return compute_shader


    def build_program(self, program_id) -> moderngl.Program: # think of as the material 
        if program_id in self.programs:
            return self.programs[program_id]

        if program_id == ProgramID.BASIC_3D:
            VERTEX_SOURCE, FRAGMENT_SOURCE = self.basic_3d_src()
        elif program_id == ProgramID.LORENZ_ATTRACTOR:
            VERTEX_SOURCE, FRAGMENT_SOURCE = self.lorenz_attractor_src()
        elif program_id == ProgramID.GRID:
            VERTEX_SOURCE, FRAGMENT_SOURCE = self.grid_src()
        elif program_id == ProgramID.SURFACE:
            VERTEX_SOURCE, FRAGMENT_SOURCE = self.surface_src()
        else:
            print('no valid shader source code available') 
            return 
        
        program = self.ctx.program(
            vertex_shader=VERTEX_SOURCE, 
            fragment_shader=FRAGMENT_SOURCE) 
        
        self.programs[program_id] = program
        return program
    
    
class RenderObject:

    def __init__(self,
        program_id: ProgramID,
        vao: moderngl.VertexArray,
        vbo: moderngl.Buffer,
        mode: Mode,
        num_vertexes: int,
        compute_shader: moderngl.ComputeShader = None,
        ) -> None:
        
        self.program_id = program_id
        self.vao = vao
        self.vbo = vbo
        self.mode = mode
        self.num_vertexes = num_vertexes
        self.compute_shader = compute_shader


    def release(self):
        self.vao.release()
        self.vbo.release()

class Renderer:
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.program_manager = ProgramManager(self.ctx)
    
    def create_render_object(self, obj: SceneObject) -> RenderObject:
        program = self.program_manager.build_program(obj.ProgramID)
        
        if obj.ProgramID == ProgramID.LORENZ_ATTRACTOR:
            vbo = self.ctx.buffer(obj.vertices.tobytes())
            vao = self.ctx.vertex_array(program, [(vbo, "4f", "in_position")])
            compute_shader = self.program_manager.build_compute_shader(obj.ProgramID)
            # ... (uniform setup for compute shader)
            return RenderObject(
                program_id=obj.ProgramID,
                vao=vao,
                vbo=vbo,
                mode=obj.Mode,
                num_vertexes=obj.num_points,
                compute_shader=compute_shader
            )
        
        elif obj.ProgramID == ProgramID.SURFACE:
            vbo = self.ctx.buffer(obj.vertices.tobytes())
            vao = self.ctx.vertex_array(program, [(vbo, "3f 3f 3f", "in_position", "in_normal", "in_color")])
            return RenderObject(
                program_id=obj.ProgramID,
                vao=vao,
                vbo=vbo,
                mode=obj.Mode,
                num_vertexes=len(obj.vertices) // 9
            )

        else: # For other objects like BASIC_3D and GRID
            vbo = self.ctx.buffer(obj.vertices.tobytes())
            vao = self.ctx.vertex_array(program, [(vbo, "3f 3f", "in_position", "in_color")])
            return RenderObject(
                program_id=obj.ProgramID,
                vao=vao,
                vbo=vbo,
                mode=obj.Mode,
                num_vertexes=len(obj.vertices) // 6
            )

    def update_render_object(self, ro: RenderObject, obj: SceneObject):
        ro.vbo.write(obj.vertices.tobytes())
        if obj.ProgramID == ProgramID.SURFACE:
            ro.num_vertexes = len(obj.vertices) // 9
        else:
            ro.num_vertexes = len(obj.vertices) // 6


    def render(self, render_objects: list, cam: Camera, width: int, height: int, h_range=None, v_range=None) -> list[RenderObject]:

        for ro in render_objects:
            if ro.compute_shader:
                ro.vbo.bind_to_storage_buffer(0)
                group_size = (ro.num_vertexes + 255) // 256
                ro.compute_shader.run(group_x=group_size)

            program = ro.vao.program
            program["u_view"].write(cam.get_view_matrix())
            program["u_proj"].write(cam.get_projection_matrix(width, height, h_range=h_range, v_range=v_range))
            
            if ro.program_id == ProgramID.SURFACE:
                program["u_model"].write(np.eye(4, dtype=np.float32).tobytes())
                program["u_light_pos"].write(np.array([10.0, 20.0, 10.0], dtype=np.float32).tobytes())
                program["u_view_pos"].write(cam.get_position().tobytes())

            if ro.program_id == ProgramID.GRID:
                alpha_multiplier = 1.0
                if cam.mode == CameraMode.ThreeD:
                    alpha_multiplier = 0.2 
                elif cam.mode == CameraMode.TwoD:
                    alpha_multiplier = 1.0
                if "u_alpha_multiplier" in program:
                    program["u_alpha_multiplier"].value = alpha_multiplier
            else:
                if "u_alpha_multiplier" in program:
                    program["u_alpha_multiplier"].value = 1.0 # Ensure other objects are opaque

            if ro.mode == Mode.POINTS:
                m = moderngl.POINTS
            elif ro.mode == Mode.LINES:
                m = moderngl.LINES
            elif ro.mode == Mode.LINE_STRIP:
                m = moderngl.LINE_STRIP
            elif ro.mode == Mode.LINE_LOOP:
                m = moderngl.LINE_LOOP
            elif ro.mode == Mode.TRIANGLES:
                m = moderngl.TRIANGLES
            else:
                m = moderngl.POINTS

            ro.vao.render(mode=m)


class RenderSpace(QOpenGLWidget):
    manual_range_cleared = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cam: Camera = Camera()
        self.input_state: InputState = InputState()
        self.ctx: moderngl.Context = None
        self.scene = None
        self.screen = None
        self.renderer: Renderer = None
        self.mouse_hovering: bool = False # Track mouse hover state
        self.manual_ranges = {} # e.g. {'x': (-10, 10), 'y': (-10, 10)}

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
        self.ctx.enable(moderngl.BLEND) # Enable blending for transparency
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
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

        h_proj_range, v_proj_range = None, None  # For passing to projection matrix

        if self.cam.mode == CameraMode.TwoD:
            # Determine horizontal and vertical axes from snap mode
            if self.cam.snap_mode == SnapMode.XY:
                h_axis, v_axis, plane_str = 'x', 'y', 'xy'
            elif self.cam.snap_mode == SnapMode.XZ:
                h_axis, v_axis, plane_str = 'x', 'z', 'xz'
            elif self.cam.snap_mode == SnapMode.YZ:
                h_axis, v_axis, plane_str = 'z', 'y', 'yz'
            else:
                h_axis, v_axis, plane_str = None, None, None

            if h_axis and v_axis:
                dynamic_h_range, dynamic_v_range = None, None

                # Check for manual ranges
                if h_axis in self.manual_ranges and v_axis in self.manual_ranges:
                    h_proj_range = self.manual_ranges[h_axis]
                    v_proj_range = self.manual_ranges[v_axis]
                    dynamic_h_range = h_proj_range
                    dynamic_v_range = v_proj_range
                elif height > 0:  # Auto-ranging
                    aspect = width / height
                    view_height = self.cam.distance
                    view_width = view_height * aspect
                    buffer_factor = 1.05

                    axis_map = {'x': 0, 'y': 1, 'z': 2}
                    h_center = self.cam.position_center[axis_map[h_axis]]
                    v_center = self.cam.position_center[axis_map[v_axis]]

                    h_min = h_center - (view_width / 2) * buffer_factor
                    h_max = h_center + (view_width / 2) * buffer_factor
                    dynamic_h_range = (h_min, h_max)

                    v_min = v_center - (view_height / 2) * buffer_factor
                    v_max = v_center + (view_height / 2) * buffer_factor
                    dynamic_v_range = (v_min, v_max)

                # Update scene objects that need ranges
                if dynamic_h_range and dynamic_v_range:
                    for obj in self.scene.objects:
                        if isinstance(obj, Grid):
                            obj.set_ranges(dynamic_h_range, dynamic_v_range, plane_str)
                        elif isinstance(obj, MathFunction):
                            obj.update_for_plane(plane_str, dynamic_h_range, dynamic_v_range)
        else:
            # Revert grid to default state if not in 2D mode
            for obj in self.scene.objects:
                if isinstance(obj, Grid):
                    obj.set_to_default()

        # Create/update render objects
        for obj in self.scene.objects:
            if hasattr(obj, 'vertices') and obj.vertices.size > 0:
                # If the object is dirty, we must recreate its render object
                # to ensure the VBO is correctly sized.
                if getattr(obj, 'is_dirty', False) and obj in self.render_objects:
                    ro = self.render_objects.pop(obj)
                    ro.release() # Release old GPU resources
                    obj.is_dirty = False

                if obj not in self.render_objects:
                    self.render_objects[obj] = self.renderer.create_render_object(obj)

            else: # Object has no vertices, remove its render object if it exists
                if obj in self.render_objects:
                    ro = self.render_objects.pop(obj)
                    ro.release()

        # Handle object removal
        removed_objects = [obj for obj in self.render_objects if obj not in self.scene.objects]
        for obj in removed_objects:
            ro = self.render_objects.pop(obj)
            ro.release()

        # Tell the renderer to draw the objects
        self.renderer.render(list(self.render_objects.values()), self.cam, width, height, h_range=h_proj_range, v_range=v_proj_range)


    def closeEvent(self, event):
        for ro in self.render_objects.values():
            ro.release()
        super().closeEvent(event)


    def update_camera(self, dt: float = 0.016): # dt is roughly 1/60th of a second
        """Updates the camera's position and orientation based on user input."""
        # Clear manual ranges on any interaction
        if self.input_state.left_mouse_pressed or abs(self.input_state.scroll_delta) > 0:
            if self.manual_ranges:
                self.manual_ranges.clear()
                self.manual_range_cleared.emit()

        keys = self.input_state.keys_held

        if self.cam.mode == CameraMode.TwoD:
            # --- Mouse Panning (2D Mode) ---
            if self.input_state.left_mouse_pressed:
                if self.height() > 0:
                    # Adjust sensitivity based on distance (zoom level) and aspect ratio
                    aspect_ratio = self.width() / self.height()
                    pan_speed_h = self.cam.distance * aspect_ratio / self.width()
                    pan_speed_v = self.cam.distance / self.height()

                    if self.cam.snap_mode == SnapMode.XY:
                        self.cam.position_center[0] -= self.input_state.mouse_delta[0] * pan_speed_h
                        self.cam.position_center[1] += self.input_state.mouse_delta[1] * pan_speed_v
                    elif self.cam.snap_mode == SnapMode.XZ:
                        self.cam.position_center[0] -= self.input_state.mouse_delta[0] * pan_speed_h
                        self.cam.position_center[2] -= self.input_state.mouse_delta[1] * pan_speed_v
                    elif self.cam.snap_mode == SnapMode.YZ:
                        self.cam.position_center[2] -= self.input_state.mouse_delta[0] * pan_speed_h
                        self.cam.position_center[1] += self.input_state.mouse_delta[1] * pan_speed_v
        else:
            # --- Mouse Rotation (3D Mode) ---
            if self.input_state.left_mouse_pressed:
                self.cam.rotation[0] += self.input_state.mouse_delta[1] # Pitch
                self.cam.rotation[1] += self.input_state.mouse_delta[0] # Yaw
                self.cam.rotation[0] = np.clip(self.cam.rotation[0], -89.0, 89.0) # Clamp pitch
                self.cam.rotation[1] = self.cam.rotation[1] % 360.0 # Wrap yaw

        # --- Mouse Zoom (Shared) ---
        if abs(self.input_state.scroll_delta) > 0:
            zoom_factor = 1.1 if self.input_state.scroll_delta < 0 else 1/1.1
            self.cam.distance *= zoom_factor
            self.cam.distance = np.clip(self.cam.distance, 1.0, 500.0)

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
        self.input_state.mouse_delta += new_pos - self.input_state.mouse_pos
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
            event.accept()
            return
        
        # Toggle 2D/3D mode with M key
        if event.key() == Qt.Key.Key_M:
            if self.cam.mode == CameraMode.ThreeD:
                self.cam.mode = CameraMode.TwoD
                self.cam.projection = Projection.Orthographic
            event.accept()
            return

        super().keyPressEvent(event)


    def keyReleaseEvent(self, event: QKeyEvent):
        """Handle key release events."""
        self.input_state.keys_held.discard(event.key())
        super().keyReleaseEvent(event)