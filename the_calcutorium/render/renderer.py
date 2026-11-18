"""OpenGL drawing. The only module (with simulations.py) that talks to the GPU.

Each frame the viewport calls:

    renderer.begin_frame(camera matrices)
    renderer.draw_mesh(key, signature, build)    # for each thing on screen
    renderer.draw_simulation(key, sim)
    renderer.end_frame()                         # frees GPU data for things not drawn

`draw_mesh` caches GPU buffers by `key`. The mesh is rebuilt (by calling `build()`)
only when `signature` changes, e.g. when the plot or the visible range changed.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Hashable

import moderngl
import numpy as np

from ..scene import LorenzAttractor, NBody
from .meshes import Mesh
from .simulations import GpuSimulation, create_simulation

SHADER_DIR = Path(__file__).parent / "shaders"

# program name → (vertex shader, fragment shader, buffer layout, attribute names)
PROGRAMS = {
    "basic": ("basic_3d.vert", "basic_3d.frag", "3f 3f", ("in_position", "in_color")),
    "surface": ("surface.vert", "surface.frag", "3f 3f 3f", ("in_position", "in_normal", "in_color")),
    "grid": ("grid.vert", "grid.frag", "3f 3f 1f", ("in_position", "in_color", "in_is_major")),
}
_MODES = {"lines": moderngl.LINES, "triangles": moderngl.TRIANGLES, "points": moderngl.POINTS}


def load_shader(filename: str) -> str:
    return (SHADER_DIR / filename).read_text(encoding="utf-8")


def set_uniform(program: moderngl.Program, name: str, value: Any) -> None:
    """Set a uniform if the program uses it (unused uniforms are compiled away)."""
    if name not in program:
        return
    if isinstance(value, np.ndarray):
        program[name].write(value.astype(np.float32).tobytes())
    else:
        program[name].value = value


@dataclass
class _GpuMesh:
    signature: Hashable
    vao: moderngl.VertexArray | None
    buffer: moderngl.Buffer | None
    mode: int

    def release(self) -> None:
        if self.vao is not None:
            self.vao.release()
            self.buffer.release()


class Renderer:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._programs: dict[str, moderngl.Program] = {}
        self._meshes: dict[Hashable, _GpuMesh] = {}
        self._simulations: dict[Hashable, GpuSimulation] = {}
        self._drawn: set[Hashable] = set()
        self._frame: dict[str, Any] = {}

    def program(self, name: str) -> moderngl.Program:
        if name not in self._programs:
            vert, frag, _, _ = PROGRAMS[name]
            self._programs[name] = self.ctx.program(vertex_shader=load_shader(vert), fragment_shader=load_shader(frag))
        return self._programs[name]

    # ---- per frame --------------------------------------------------------

    def begin_frame(self, view: np.ndarray, proj: np.ndarray, eye: np.ndarray, grid_alpha: float = 1.0) -> None:
        self._frame = {"u_view": view, "u_proj": proj, "u_view_pos": eye, "u_alpha_multiplier": grid_alpha}
        self._drawn = set()
        self.ctx.line_width = 2.0

    def draw_mesh(self, key: Hashable, signature: Hashable, build: Callable[[], Mesh | None]) -> None:
        self._drawn.add(key)
        cached = self._meshes.get(key)
        if cached is None or cached.signature != signature:
            if cached is not None:
                cached.release()
            cached = self._meshes[key] = self._upload(signature, build())
        if cached.vao is None:
            return  # nothing to draw (e.g. curve not in this plane)

        program = cached.vao.program
        for name, value in self._frame.items():
            set_uniform(program, name, value)
        set_uniform(program, "u_model", np.eye(4, dtype=np.float32))
        set_uniform(program, "u_light_pos", np.array([20.0, 10.0, 40.0], dtype=np.float32))  # above (z-up)
        cached.vao.render(mode=cached.mode)

    def draw_simulation(self, key: Hashable, sim: LorenzAttractor | NBody) -> None:
        self._drawn.add(key)
        gpu = self._simulations.get(key)
        if gpu is not None and not gpu.can_update_to(sim):
            gpu.release()
            gpu = None
        if gpu is None:
            gpu = self._simulations[key] = create_simulation(self.ctx, sim)
        gpu.set_params(sim)
        gpu.step()
        gpu.draw(self._frame["u_view"], self._frame["u_proj"])

    def end_frame(self) -> None:
        """Free GPU memory for everything that wasn't drawn this frame."""
        for cache in (self._meshes, self._simulations):
            for key in [k for k in cache if k not in self._drawn]:
                cache.pop(key).release()

    def release(self) -> None:
        self._drawn = set()
        self.end_frame()

    # ---- helpers ----------------------------------------------------------

    def _upload(self, signature: Hashable, mesh: Mesh | None) -> _GpuMesh:
        if mesh is None or len(mesh.data) == 0:
            return _GpuMesh(signature, None, None, 0)
        _, _, layout, attributes = PROGRAMS[mesh.program]
        buffer = self.ctx.buffer(np.ascontiguousarray(mesh.data, dtype=np.float32).tobytes())
        vao = self.ctx.vertex_array(self.program(mesh.program), [(buffer, layout, *attributes)])
        return _GpuMesh(signature, vao, buffer, _MODES[mesh.mode])
