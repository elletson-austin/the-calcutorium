import logging
from typing import Any, TYPE_CHECKING

import moderngl
import numpy as np

from .program_manager import ProgramManager
from .render_adapters import ADAPTERS_BY_PROGRAM_ID, DefaultAdapter, RenderAdapter
from .render_object import RenderObject
from .scene import ProgramID, RenderMode, SceneObject

if TYPE_CHECKING:
    from .camera import Camera2D, Camera3D

logger = logging.getLogger(__name__)


class Renderer:

    def __init__(self, ctx: moderngl.Context):
        self.ctx: moderngl.Context = ctx
        self.program_manager: ProgramManager = ProgramManager(self.ctx)
        self._adapters_by_program_id: dict[ProgramID, type[RenderAdapter]] = dict(
            ADAPTERS_BY_PROGRAM_ID
        )
        self._default_adapter: type[RenderAdapter] = DefaultAdapter

    def set_uniform(
        self,
        program: moderngl.Program,
        name: str | None = None,
        value: Any | None = None,
        **uniforms: Any,
    ) -> None:
        """Set shader uniform(s). Accepts a single (name, value) or any number of kwargs."""
        if name is not None and value is not None:
            self._set_single_uniform(program, name, value)

        for uniform_name, uniform_value in uniforms.items():
            self._set_single_uniform(program, uniform_name, uniform_value)

    def _create_buffer(self, data: bytes, *, dynamic: bool = False) -> moderngl.Buffer | None:
        try:
            return self.ctx.buffer(data, dynamic=dynamic)
        except Exception as e:
            logger.error("Failed to create GL buffer (dynamic=%s): %s", dynamic, e)
            return None

    def _safe_buffer_write(self, buf: moderngl.Buffer | None, data: bytes) -> None:
        if buf is None:
            return
        try:
            buf.write(data)
        except Exception as e:
            logger.error("Failed to write %d bytes to buffer: %s", len(data), e)

    def _bind_storage_buffers(self, ro: RenderObject) -> None:
        if ro.storage_buffers:
            for buf, binding in ro.storage_buffers:
                buf.bind_to_storage_buffer(binding)
        elif ro.ssbo is not None:
            ro.ssbo.bind_to_storage_buffer(0)

    def _dispatch_compute(self, ro: RenderObject) -> None:
        if ro.compute_shader is None:
            return

        if ro.compute_groups is not None:
            gx, gy, gz = ro.compute_groups
        elif ro.num_vertexes is not None:
            lx = ro.compute_local_size_x or 256
            gx = (ro.num_vertexes + lx - 1) // lx
            gy, gz = 1, 1
        else:
            gx, gy, gz = 1, 1, 1

        ro.compute_shader.run(gx, gy, gz)
        self.ctx.memory_barrier()

    def _set_single_uniform(self, program: moderngl.Program, name: str, value: Any) -> None:
        """Set a single shader uniform with type-aware handling."""
        if name not in program:
            return

        try:
            if hasattr(value, "tobytes") and not isinstance(value, (int, float, bool)):
                program[name].write(value.tobytes())
            else:
                program[name].value = value
        except Exception as e:
            logger.warning("Failed to set uniform %r=%r: %s", name, value, e)

    def create_render_object(self, obj: SceneObject) -> RenderObject:
        adapter = self._adapters_by_program_id.get(obj.program_id, self._default_adapter)
        return adapter().create(renderer=self, obj=obj)

    def update_render_object(self, ro: RenderObject, obj: SceneObject) -> None:
        adapter = self._adapters_by_program_id.get(obj.program_id, self._default_adapter)
        adapter().update(renderer=self, ro=ro, obj=obj)

    def render(
        self,
        scene: list[RenderObject],
        camera: "Camera3D | Camera2D",
        width: int,
        height: int,
        h_range: tuple[float, float] | None = None,
        v_range: tuple[float, float] | None = None,
    ) -> None:
        for ro in scene:
            if ro.compute_shader is not None:
                self._bind_storage_buffers(ro)
                self.set_uniform(program=ro.compute_shader, **ro.compute_uniforms)
                self._dispatch_compute(ro)

            program = ro.vao.program
            self._set_camera_uniforms(program, camera, width, height, h_range, v_range)
            self._set_program_specific_uniforms(program, ro.program_id, camera)

            self.ctx.line_width = 2.0
            ro.vao.render(mode=self._get_render_mode(ro.render_mode))

    def _set_camera_uniforms(self, program, camera, width, height, h_range, v_range):
        program["u_view"].write(camera.get_view_matrix())
        program["u_proj"].write(
            camera.get_projection_matrix(width, height, h_range=h_range, v_range=v_range)
        )

    def _set_program_specific_uniforms(self, program, program_id, camera):
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

        self.set_uniform(program, "u_alpha_multiplier", self._get_alpha_multiplier(program_id, camera))

    def _get_alpha_multiplier(self, program_id, camera):
        from .camera import Camera3D

        if program_id == ProgramID.GRID and isinstance(camera, Camera3D):
            return 0.2
        return 1.0

    def _get_render_mode(self, render_mode: RenderMode) -> int:
        mode_map = {
            RenderMode.POINTS: moderngl.POINTS,
            RenderMode.LINES: moderngl.LINES,
            RenderMode.LINE_STRIP: moderngl.LINE_STRIP,
            RenderMode.LINE_LOOP: moderngl.LINE_LOOP,
            RenderMode.TRIANGLES: moderngl.TRIANGLES,
        }
        return mode_map.get(render_mode, moderngl.POINTS)
