import moderngl
from typing import Dict
import os

from .scene import ProgramID

class ProgramManager: # holds and stores programs that draw points, lines, etc.

    def __init__(self, ctx: moderngl.Context):
        self.programs: Dict[ProgramID, moderngl.Program] = {}
        self.compute_shaders: Dict[ProgramID, moderngl.ComputeShader] = {}
        self.ctx = ctx
        # Correctly determine the shader directory path
        self.shader_dir = os.path.join(os.path.dirname(__file__), 'shaders')

    def _read_shader_source(self, filename: str) -> str:
        file_path = os.path.join(self.shader_dir, filename)
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: Shader file not found at {file_path}")
            return ""

    def build_compute_shader(self, ProgramID: ProgramID) -> moderngl.ComputeShader:
        if ProgramID in self.compute_shaders:
            return self.compute_shaders[ProgramID]

        if ProgramID == ProgramID.LORENZ_ATTRACTOR:
            COMPUTE_SOURCE = self._read_shader_source('lorenz_attractor.comp')
        elif ProgramID == ProgramID.NBODY:
            COMPUTE_SOURCE = self._read_shader_source('nbody.comp')
        else:
            print('no valid compute shader source code available')
            return None

        if not COMPUTE_SOURCE:
            return None

        compute_shader = self.ctx.compute_shader(COMPUTE_SOURCE)
        self.compute_shaders[ProgramID] = compute_shader
        return compute_shader


    def build_program(self, ProgramID: ProgramID) -> moderngl.Program: # think of as the material
        if ProgramID in self.programs:
            return self.programs[ProgramID]

        shader_map = {
            ProgramID.BASIC_3D: ('basic_3d.vert', 'basic_3d.frag'),
            ProgramID.LORENZ_ATTRACTOR: ('lorenz_attractor.vert', 'lorenz_attractor.frag'),
            ProgramID.NBODY: ('nbody.vert', 'nbody.frag'),
            ProgramID.GRID: ('grid.vert', 'grid.frag'),
            ProgramID.SURFACE: ('surface.vert', 'surface.frag'),
        }

        if ProgramID not in shader_map:
            print('no valid shader source code available')
            return None

        vert_filename, frag_filename = shader_map[ProgramID]
        VERTEX_SOURCE = self._read_shader_source(vert_filename)
        FRAGMENT_SOURCE = self._read_shader_source(frag_filename)

        if not VERTEX_SOURCE or not FRAGMENT_SOURCE:
            return None

        program = self.ctx.program(
            vertex_shader=VERTEX_SOURCE,
            fragment_shader=FRAGMENT_SOURCE)

        self.programs[ProgramID] = program
        return program