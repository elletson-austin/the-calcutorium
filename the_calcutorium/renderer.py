import moderngl
import numpy as np
from typing import TYPE_CHECKING

from .scene import SceneObject, ProgramID, RenderMode
from .program_manager import ProgramManager

if TYPE_CHECKING:
    from .camera import Camera3D, Camera2D

class RenderObject:

    def __init__(self,
        program_id: ProgramID,
        vao: moderngl.VertexArray,
        ssbo: moderngl.Buffer,
        Rendermode: RenderMode,
        num_vertexes: int,
        compute_shader: moderngl.ComputeShader = None,
        compute_uniforms: dict = {},
        storage_buffers: list = [], # list of (buffer, binding_index) pairs for SSBOs
        compute_groups: tuple = None,
        compute_local_size_x: int = 256,
        ) -> None:
        
        self.program_id = program_id
        self.vao = vao
        self.ssbo = ssbo  
        self.Rendermode = Rendermode
        self.num_vertexes = num_vertexes
        self.compute_shader = compute_shader
        self.compute_uniforms = compute_uniforms
        self.storage_buffers = storage_buffers  
        self.compute_groups = compute_groups
        self.compute_local_size_x = compute_local_size_x


    def _release_buffer(self, buf):
        """Safely release a single buffer, handling None and already-released buffers."""
        if buf is not None:
            try:
                buf.release()
            except Exception:
                pass

    def release(self):
        """Release all GPU resources held by this render object."""
        self._release_buffer(self.vao)
        self._release_buffer(self.ssbo)
        
        # Release all storage buffers, avoiding double-release of primary ssbo
        if hasattr(self, 'storage_buffers') and self.storage_buffers:
            for buf, _ in self.storage_buffers:
                if buf is not self.ssbo:
                    self._release_buffer(buf)

class Renderer:
    
    def __init__(self, ctx: moderngl.Context = None):
        self.ctx = ctx
        self.program_manager = ProgramManager(self.ctx)
    
    def set_uniform(self, program: moderngl.Program, name=None, value=None, **uniforms) -> None:
        """Set shader uniform(s). Accepts individual or multiple via kwargs."""

        # Single uniform mode
        if name is not None and value is not None:
            self._set_single_uniform(program, name, value)
        
        # Multi uniform mode via kwargs
        for uniform_name, uniform_value in uniforms.items():
            self._set_single_uniform(program, uniform_name, uniform_value)

    def _create_buffer(self, data: bytes, *, dynamic: bool = False):
        try:
            return self.ctx.buffer(data, dynamic=dynamic)
        except Exception:
            try:
                return self.ctx.buffer(data)
            except Exception:
                return None

    def _safe_buffer_write(self, buf, data: bytes):
        if buf is None:
            return
        try:
            buf.write(data)
        except Exception:
            pass

    def _bind_storage_buffers(self, ro):
        if getattr(ro, 'storage_buffers', None):
            for buf, binding in ro.storage_buffers:
                try:
                    buf.bind_to_storage_buffer(binding)
                except Exception:
                    pass
        else:
            try:
                ro.ssbo.bind_to_storage_buffer(0)
            except Exception:
                pass

    def _dispatch_compute(self, ro):
        # Determine dispatch size
        if getattr(ro, 'compute_groups', None) is not None:
            gx, gy, gz = ro.compute_groups
        elif getattr(ro, 'num_vertexes', None) is not None:
            lx = getattr(ro, 'compute_local_size_x', 256)
            gx = (ro.num_vertexes + lx - 1) // lx
            gy, gz = 1, 1
        else:
            gx, gy, gz = 1, 1, 1

        # Dispatch compute shader (try multiple call signatures)
        try:
            ro.compute_shader.run(gx, gy, gz)
        except Exception:
            try:
                ro.compute_shader.run((gx, gy, gz))
            except Exception:
                pass

        # Memory barrier for compute writes
        try:
            self.ctx.memory_barrier()
        except Exception:
            pass

    def _set_single_uniform(self, program: moderngl.Program, name: str, value) -> None:
        """Set a single shader uniform with type handling and fallback."""
        if name not in program:
            return
        
        try:
            # Try writing raw bytes for numpy arrays
            if hasattr(value, 'tobytes') and not isinstance(value, (int, float, bool)):
                program[name].write(value.tobytes())
            else:
                program[name].value = value
        except Exception:
            # Fallback to .value assignment
            try:
                program[name].value = value
            except Exception:
                pass

    def create_render_object(self, obj: SceneObject) -> RenderObject:
        program = self.program_manager.build_program(obj.ProgramID)
        
        if obj.ProgramID == ProgramID.LORENZ_ATTRACTOR:
            ssbo = self._create_buffer(obj.vertices.tobytes(), dynamic=True)
            vao = self.ctx.vertex_array(program, [(ssbo, "4f", "in_position")])
            compute_shader = self.program_manager.build_compute_shader(obj.ProgramID)

            ro = RenderObject(
                program_id=obj.ProgramID,
                vao=vao,
                ssbo=ssbo,
                Rendermode=obj.RenderMode,
                num_vertexes=obj.num_points,
                compute_shader=compute_shader,
                compute_uniforms=obj.uniforms,
                storage_buffers=[(ssbo, 0)],
                compute_local_size_x=256,
            )
            return ro
            
        elif obj.ProgramID == ProgramID.NBODY:
            # Create SSBOs for positions, velocities and masses
            pos_ssbo = self._create_buffer(obj.positions.tobytes(), dynamic=True)
            vel_ssbo = self._create_buffer(obj.velocities.tobytes(), dynamic=True)
            mass_ssbo = self._create_buffer(obj.masses.tobytes(), dynamic=True)

            vao = self.ctx.vertex_array(program, [(pos_ssbo, "4f", "in_position")])
            compute_shader = self.program_manager.build_compute_shader(obj.ProgramID)

            ro = RenderObject(
                program_id=obj.ProgramID,
                vao=vao,
                ssbo=pos_ssbo,
                Rendermode=obj.RenderMode,
                num_vertexes=obj.num_bodies,
                compute_shader=compute_shader,
                compute_uniforms=obj.uniforms,
                storage_buffers=[(pos_ssbo, 0), (vel_ssbo, 1), (mass_ssbo, 2)],
                compute_local_size_x=256,
            )
            # Keep references for updates - stored in storage_buffers but also here for clarity
            ro.vel_ssbo = vel_ssbo
            ro.mass_ssbo = mass_ssbo
            return ro
        
        elif obj.ProgramID == ProgramID.SURFACE:
            ssbo = self._create_buffer(obj.vertices.tobytes())
            vao = self.ctx.vertex_array(program, [(ssbo, "3f 3f 3f", "in_position", "in_normal", "in_color")])
            return RenderObject(
                program_id=obj.ProgramID,
                vao=vao,
                ssbo=ssbo,
                Rendermode=obj.RenderMode,
                num_vertexes=len(obj.vertices) // 9
            )

        elif obj.ProgramID == ProgramID.GRID:
            ssbo = self._create_buffer(obj.vertices.tobytes())
            vao = self.ctx.vertex_array(program, [(ssbo, "3f 3f 1f", "in_position", "in_color", "in_is_major")])
            return RenderObject(
                program_id=obj.ProgramID,
                vao=vao,
                ssbo=ssbo,
                Rendermode=obj.RenderMode,
                num_vertexes=len(obj.vertices) // 7
            )

        else:  # BASIC_3D, etc.
            ssbo = self._create_buffer(obj.vertices.tobytes())
            vao = self.ctx.vertex_array(program, [(ssbo, "3f 3f", "in_position", "in_color")])
            return RenderObject(
                program_id=obj.ProgramID,
                vao=vao,
                ssbo=ssbo,
                Rendermode=obj.RenderMode,
                num_vertexes=len(obj.vertices) // 6
            )

    def update_render_object(self, ro: RenderObject, obj: SceneObject):
        """Update render object with new data from scene object."""
        if obj.ProgramID == ProgramID.NBODY:
            # Update position SSBO
            self._safe_buffer_write(ro.ssbo, obj.positions.tobytes())
            if hasattr(ro, 'vel_ssbo') and hasattr(obj, 'velocities'):
                self._safe_buffer_write(ro.vel_ssbo, obj.velocities.tobytes())
            if hasattr(ro, 'mass_ssbo') and hasattr(obj, 'masses'):
                self._safe_buffer_write(ro.mass_ssbo, obj.masses.tobytes())
            ro.num_vertexes = getattr(obj, 'num_bodies', obj.positions.shape[0])
        else:
            # Update primary SSBO
            self._safe_buffer_write(ro.ssbo, obj.vertices.tobytes())
            
            # Recalculate vertex count based on program type
            if obj.ProgramID == ProgramID.SURFACE:
                ro.num_vertexes = len(obj.vertices) // 9
            elif obj.ProgramID == ProgramID.GRID:
                ro.num_vertexes = len(obj.vertices) // 7
            elif obj.ProgramID == ProgramID.LORENZ_ATTRACTOR:
                ro.num_vertexes = getattr(obj, 'num_points', obj.vertices.shape[0])
            else:
                ro.num_vertexes = len(obj.vertices) // 6

        # Update compute uniforms if available
        if hasattr(obj, 'uniforms'):
            ro.compute_uniforms.update(obj.uniforms)


    def render(self,
           scene,
           camera: 'Camera3D | Camera2D',
           width: int,
           height: int,
           h_range=None,
           v_range=None):
        # Caller (e.g. RenderWindow) passes a list of RenderObjects and optional h_range/v_range
        render_objects_list = scene if isinstance(scene, list) else []
        h_proj_range = h_range
        v_proj_range = v_range

        for ro in render_objects_list:
            # Bind SSBOs for any compute operations
            if getattr(ro, 'compute_shader', None) is not None:
                self._bind_storage_buffers(ro)
                # Apply compute uniforms
                self.set_uniform(ro.compute_shader, **ro.compute_uniforms)
                # Dispatch compute and perform memory barrier
                self._dispatch_compute(ro)

            program = ro.vao.program
            
            self._set_camera_uniforms(program, camera, width, height, h_proj_range, v_proj_range)
            self._set_program_specific_uniforms(program, ro.program_id, camera)
            
            render_mode = self._get_render_mode(ro.Rendermode)
            
            self.ctx.line_width = 2.0
            ro.vao.render(mode=render_mode)

    def _prepare_scene(self, scene, camera, width, height):
        from .camera import Camera2D
        from .scene import Grid

        h_proj_range, v_proj_range = None, None

        if isinstance(camera, Camera2D):
            h_proj_range, v_proj_range = self._update_2d_rendering_params(scene, camera, width, height)
        else:  # 3D Camera
            for obj in scene.objects:
                if isinstance(obj, Grid):
                    obj.update()

        if isinstance(camera, Camera2D):
            filtered_scene_objs = {obj for obj in scene.objects if
                                   obj.is_2d and hasattr(obj, 'vertices') and obj.vertices.size > 0}
        else:  # Camera3D
            filtered_scene_objs = {obj for obj in scene.objects if
                                   hasattr(obj, 'vertices') and obj.vertices.size > 0}

        return filtered_scene_objs, h_proj_range, v_proj_range

    def _update_2d_rendering_params(self, scene, camera, width, height):
        from .render_types import SnapMode

        h_proj_range, v_proj_range = None, None
        cam2d = camera
        snap_map = {SnapMode.XY: ('x', 'y', 'xy'), SnapMode.XZ: ('x', 'z', 'xz'), SnapMode.YZ: ('z', 'y', 'yz')}
        h_axis, v_axis, plane_str = snap_map.get(cam2d.snap_mode, (None, None, None))

        if h_axis and v_axis:
            dynamic_h_range, dynamic_v_range = self._compute_2d_dynamic_ranges(cam2d, h_axis, v_axis, width, height)

            if dynamic_h_range and dynamic_v_range:
                self._apply_dynamic_ranges_to_scene(scene, plane_str, dynamic_h_range, dynamic_v_range)

            if self.has_manual_range(h_axis) and self.has_manual_range(v_axis):
                h_proj_range = self.get_manual_range(h_axis)
                v_proj_range = self.get_manual_range(v_axis)

        return h_proj_range, v_proj_range

    def _compute_2d_dynamic_ranges(self, cam2d, h_axis, v_axis, width, height):
        if self.has_manual_range(h_axis) and self.has_manual_range(v_axis):
            return self.get_manual_range(h_axis), self.get_manual_range(v_axis)

        if height <= 0:
            return None, None

        aspect = width / height
        view_height = cam2d.distance
        view_width = view_height * aspect
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        h_center = cam2d.position_center[axis_map[h_axis]]
        v_center = cam2d.position_center[axis_map[v_axis]]
        h_min, h_max = h_center - (view_width / 2), h_center + (view_width / 2)
        v_min, v_max = v_center - (view_height / 2), v_center + (view_height / 2)
        return (h_min, h_max), (v_min, v_max)

    def _apply_dynamic_ranges_to_scene(self, scene, plane_str, h_range, v_range):
        from .scene import Grid, MathFunction
        for obj in scene.objects:
            if isinstance(obj, Grid):
                obj.update(h_range=h_range, v_range=v_range, plane=plane_str)
            elif isinstance(obj, MathFunction):
                obj.update(plane=plane_str, h_range=h_range, v_range=v_range)


    def _set_camera_uniforms(self, program, camera, width, height, h_range, v_range):
        """Set view and projection matrices."""
        program["u_view"].write(camera.get_view_matrix())
        program["u_proj"].write(camera.get_projection_matrix(width, height, h_range=h_range, v_range=v_range))


    def _set_program_specific_uniforms(self, program, program_id, camera):
        """Set uniforms specific to different shader programs via kwargs."""
        if program_id == ProgramID.SURFACE:
            self.set_uniform(
                program,
                u_model=np.eye(4, dtype=np.float32),
                u_light_pos=np.array([10.0, 20.0, 10.0], dtype=np.float32),
                u_view_pos=camera.get_position(),
            )
        elif program_id == ProgramID.NBODY:
            self.set_uniform(
                program,
                u_model=np.eye(4, dtype=np.float32),
                u_point_size=5.0,
                u_color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            )
        
        # Set alpha multiplier for all program types
        alpha_multiplier = self._get_alpha_multiplier(program_id, camera)
        self.set_uniform(program, "u_alpha_multiplier", alpha_multiplier)

    def _get_alpha_multiplier(self, program_id, camera):
        """Calculate alpha multiplier based on program type and camera."""
        from .camera import Camera3D
        if program_id == ProgramID.GRID and isinstance(camera, Camera3D):
            return 0.2
        return 1.0

    def _get_render_mode(self, render_mode):
        """Map RenderMode enum to moderngl constant."""
        mode_map = {
            RenderMode.POINTS: moderngl.POINTS,
            RenderMode.LINES: moderngl.LINES,
            RenderMode.LINE_STRIP: moderngl.LINE_STRIP,
            RenderMode.LINE_LOOP: moderngl.LINE_LOOP,
            RenderMode.TRIANGLES: moderngl.TRIANGLES,
        }
        return mode_map.get(render_mode, moderngl.POINTS)