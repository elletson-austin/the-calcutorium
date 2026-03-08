import moderngl
import numpy as np
from typing import TYPE_CHECKING, Any

from .scene import SceneObject, ProgramID, RenderMode
from .program_manager import ProgramManager
from .render_object import RenderObject
from .render_adapters import ADAPTERS_BY_PROGRAM_ID, DefaultAdapter, RenderAdapter

if TYPE_CHECKING:
    from .camera import Camera3D, Camera2D

class Renderer:
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx: moderngl.Context = ctx
        self.program_manager: ProgramManager = ProgramManager(self.ctx)
        self._adapters_by_program_id: dict[ProgramID, type[RenderAdapter]] = dict[ProgramID, type[RenderAdapter]](ADAPTERS_BY_PROGRAM_ID) 
        self._default_adapter: type[RenderAdapter] = DefaultAdapter
    
    def set_uniform(self, program: moderngl.Program, name: str | None = None, value: Any | None = None, **uniforms: Any) -> None:
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
        adapter = self._adapters_by_program_id.get(obj.ProgramID, self._default_adapter)
        return adapter().create(renderer=self, obj=obj)

    def update_render_object(self, ro: RenderObject, obj: SceneObject):
        """Update render object with new data from scene object."""
        adapter = self._adapters_by_program_id.get(obj.ProgramID, self._default_adapter)
        adapter().update(renderer=self, ro=ro, obj=obj)


    def render(self,
           scene: list[RenderObject],
           camera: 'Camera3D | Camera2D',
           width: int,
           height: int,
           h_range: tuple[float, float] | None = None,
           v_range: tuple[float, float] | None = None):
        # Caller (e.g. RenderWindow) passes a list of RenderObjects and optional h_range/v_range
        render_objects_list = scene if isinstance(scene, list) else []
        h_proj_range: tuple[float, float] | None = h_range
        v_proj_range: tuple[float, float] | None = v_range

        for ro in render_objects_list:
            # Bind SSBOs for any compute operations
            if ro.compute_shader is not None:
                self._bind_storage_buffers(ro)
                # Apply compute uniforms
                self.set_uniform(program=ro.compute_shader, **ro.compute_uniforms)
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