import logging
import os
from typing import Dict

import moderngl

from .scene import ProgramID

logger = logging.getLogger(__name__)


class ProgramManager:
    """Holds compiled GL programs and compute shaders, keyed by ProgramID."""

    def __init__(self, ctx: moderngl.Context):
        self.programs: Dict[ProgramID, moderngl.Program] = {}
        self.compute_shaders: Dict[ProgramID, moderngl.ComputeShader] = {}
        self.ctx = ctx
        self.shader_dir = os.path.join(os.path.dirname(__file__), "shaders")

    def _read_shader_source(self, filename: str) -> str:
        file_path = os.path.join(self.shader_dir, filename)
        with open(file_path, "r") as f:
            return f.read()

    def build_compute_shader(self, program_id: ProgramID) -> moderngl.ComputeShader | None:
        if program_id in self.compute_shaders:
            return self.compute_shaders[program_id]

        compute_shader_files = {
            ProgramID.LORENZ_ATTRACTOR: "lorenz_attractor.comp",
            ProgramID.NBODY: "nbody.comp",
        }
        filename = compute_shader_files.get(program_id)
        if filename is None:
            logger.error("No compute shader registered for %s", program_id)
            return None

        compute_source = self._read_shader_source(filename)
        compute_shader = self.ctx.compute_shader(compute_source)
        self.compute_shaders[program_id] = compute_shader
        return compute_shader

    def build_program(self, program_id: ProgramID) -> moderngl.Program | None:
        if program_id in self.programs:
            return self.programs[program_id]

        shader_map = {
            ProgramID.BASIC_3D: ("basic_3d.vert", "basic_3d.frag"),
            ProgramID.LORENZ_ATTRACTOR: ("lorenz_attractor.vert", "lorenz_attractor.frag"),
            ProgramID.NBODY: ("nbody.vert", "nbody.frag"),
            ProgramID.GRID: ("grid.vert", "grid.frag"),
            ProgramID.SURFACE: ("surface.vert", "surface.frag"),
        }

        if program_id not in shader_map:
            logger.error("No shader sources registered for %s", program_id)
            return None

        vert_filename, frag_filename = shader_map[program_id]
        vertex_source = self._read_shader_source(vert_filename)
        fragment_source = self._read_shader_source(frag_filename)

        program = self.ctx.program(
            vertex_shader=vertex_source,
            fragment_shader=fragment_source,
        )
        self.programs[program_id] = program
        return program
