from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING

import numpy as np

from .scene import SceneObject, ProgramID
from .render_object import RenderObject

if TYPE_CHECKING:
    from .renderer import Renderer

class RenderAdapter(Protocol):
    def create(self, renderer: Renderer, obj: SceneObject) -> RenderObject: ...
    def update(self, renderer: Renderer, ro: RenderObject, obj: SceneObject) -> None: ...


@dataclass(frozen=True)
class DefaultAdapter:
    """Default adapter for objects with obj.vertices and basic_3d shader inputs."""

    def create(self, renderer: Renderer, obj: SceneObject) -> RenderObject:
        program = renderer.program_manager.build_program(obj.ProgramID)
        ssbo = renderer._create_buffer(obj.vertices.tobytes())
        vao = renderer.ctx.vertex_array(program, [(ssbo, "3f 3f", "in_position", "in_color")])
        return RenderObject(
            program_id=obj.ProgramID,
            vao=vao,
            ssbo=ssbo,
            Rendermode=obj.RenderMode,
            num_vertexes=len(obj.vertices) // 6,
        )

    def update(self, renderer: "Renderer", ro: RenderObject, obj: SceneObject) -> None:
        renderer._safe_buffer_write(ro.ssbo, obj.vertices.tobytes())
        ro.num_vertexes = len(obj.vertices) // 6
        if hasattr(obj, "uniforms"):
            ro.compute_uniforms.update(obj.uniforms)


@dataclass(frozen=True)
class SurfaceAdapter:
    def create(self, renderer: "Renderer", obj: SceneObject) -> RenderObject:
        program = renderer.program_manager.build_program(obj.ProgramID)
        ssbo = renderer._create_buffer(obj.vertices.tobytes())
        vao = renderer.ctx.vertex_array(program, [(ssbo, "3f 3f 3f", "in_position", "in_normal", "in_color")])
        return RenderObject(
            program_id=obj.ProgramID,
            vao=vao,
            ssbo=ssbo,
            Rendermode=obj.RenderMode,
            num_vertexes=len(obj.vertices) // 9,
        )

    def update(self, renderer: "Renderer", ro: RenderObject, obj: SceneObject) -> None:
        renderer._safe_buffer_write(ro.ssbo, obj.vertices.tobytes())
        ro.num_vertexes = len(obj.vertices) // 9
        if hasattr(obj, "uniforms"):
            ro.compute_uniforms.update(obj.uniforms)


@dataclass(frozen=True)
class GridAdapter:
    def create(self, renderer: "Renderer", obj: SceneObject) -> RenderObject:
        program = renderer.program_manager.build_program(obj.ProgramID)
        ssbo = renderer._create_buffer(obj.vertices.tobytes())
        vao = renderer.ctx.vertex_array(program, [(ssbo, "3f 3f 1f", "in_position", "in_color", "in_is_major")])
        return RenderObject(
            program_id=obj.ProgramID,
            vao=vao,
            ssbo=ssbo,
            Rendermode=obj.RenderMode,
            num_vertexes=len(obj.vertices) // 7,
        )

    def update(self, renderer: "Renderer", ro: RenderObject, obj: SceneObject) -> None:
        renderer._safe_buffer_write(ro.ssbo, obj.vertices.tobytes())
        ro.num_vertexes = len(obj.vertices) // 7
        if hasattr(obj, "uniforms"):
            ro.compute_uniforms.update(obj.uniforms)


@dataclass(frozen=True)
class LorenzAdapter:
    def create(self, renderer: "Renderer", obj: SceneObject) -> RenderObject:
        program = renderer.program_manager.build_program(obj.ProgramID)
        ssbo = renderer._create_buffer(obj.vertices.tobytes(), dynamic=True)
        vao = renderer.ctx.vertex_array(program, [(ssbo, "4f", "in_position")])
        compute_shader = renderer.program_manager.build_compute_shader(obj.ProgramID)

        return RenderObject(
            program_id=obj.ProgramID,
            vao=vao,
            ssbo=ssbo,
            Rendermode=obj.RenderMode,
            num_vertexes=getattr(obj, "num_points", obj.vertices.shape[0]),
            compute_shader=compute_shader,
            compute_uniforms=getattr(obj, "uniforms", {}),
            storage_buffers=[(ssbo, 0)],
            compute_local_size_x=256,
        )

    def update(self, renderer: "Renderer", ro: RenderObject, obj: SceneObject) -> None:
        renderer._safe_buffer_write(ro.ssbo, obj.vertices.tobytes())
        ro.num_vertexes = getattr(obj, "num_points", obj.vertices.shape[0])
        if hasattr(obj, "uniforms"):
            ro.compute_uniforms.update(obj.uniforms)


@dataclass(frozen=True)
class NBodyAdapter:
    def create(self, renderer: "Renderer", obj: SceneObject) -> RenderObject:
        program = renderer.program_manager.build_program(obj.ProgramID)

        pos_ssbo = renderer._create_buffer(obj.positions.tobytes(), dynamic=True)
        vel_ssbo = renderer._create_buffer(obj.velocities.tobytes(), dynamic=True)
        mass_ssbo = renderer._create_buffer(obj.masses.tobytes(), dynamic=True)

        vao = renderer.ctx.vertex_array(program, [(pos_ssbo, "4f", "in_position")])
        compute_shader = renderer.program_manager.build_compute_shader(obj.ProgramID)

        ro = RenderObject(
            program_id=obj.ProgramID,
            vao=vao,
            ssbo=pos_ssbo,
            Rendermode=obj.RenderMode,
            num_vertexes=getattr(obj, "num_bodies", obj.positions.shape[0]),
            compute_shader=compute_shader,
            compute_uniforms=getattr(obj, "uniforms", {}),
            storage_buffers=[(pos_ssbo, 0), (vel_ssbo, 1), (mass_ssbo, 2)],
            compute_local_size_x=256,
        )

        ro.vel_ssbo = vel_ssbo
        ro.mass_ssbo = mass_ssbo
        return ro

    def update(self, renderer: "Renderer", ro: RenderObject, obj: SceneObject) -> None:
        renderer._safe_buffer_write(ro.ssbo, obj.positions.tobytes())
        if hasattr(ro, "vel_ssbo") and hasattr(obj, "velocities"):
            renderer._safe_buffer_write(ro.vel_ssbo, obj.velocities.tobytes())
        if hasattr(ro, "mass_ssbo") and hasattr(obj, "masses"):
            renderer._safe_buffer_write(ro.mass_ssbo, obj.masses.tobytes())
        ro.num_vertexes = getattr(obj, "num_bodies", obj.positions.shape[0])
        if hasattr(obj, "uniforms"):
            ro.compute_uniforms.update(obj.uniforms)


ADAPTERS_BY_PROGRAM_ID: dict[ProgramID, RenderAdapter] = {
    ProgramID.SURFACE: SurfaceAdapter(),
    ProgramID.GRID: GridAdapter(),
    ProgramID.LORENZ_ATTRACTOR: LorenzAdapter(),
    ProgramID.NBODY: NBodyAdapter(),
    ProgramID.BASIC_3D: DefaultAdapter(),
}

