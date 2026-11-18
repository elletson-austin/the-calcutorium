"""GPU simulations: particles live in GPU buffers and a compute shader steps them each frame.

The scene object (LorenzAttractor) holds the parameters; this module holds the
running GPU state. Changing a parameter like rho just updates a uniform; changing the
particle count or seed rebuilds the simulation from scratch.
"""

from pathlib import Path

import moderngl
import numpy as np

from ..scene import LorenzAttractor

SHADER_DIR = Path(__file__).parent / "shaders"
LOCAL_SIZE = 256  # must match local_size_x in the .comp shaders


def _shader(name: str) -> str:
    return (SHADER_DIR / name).read_text(encoding="utf-8")


def _set(program, name: str, value) -> None:
    if name not in program:
        return
    if isinstance(value, np.ndarray):
        program[name].write(value.astype(np.float32).tobytes())
    else:
        program[name].value = value


class GpuSimulation:
    """Base: a compute shader that updates a positions buffer, and a program that draws it."""

    params: tuple[str, ...] = ()       # scene-object fields passed to the compute shader
    structural: tuple[str, ...] = ()   # fields that require a rebuild when changed

    def __init__(self, ctx: moderngl.Context, sim, compute: str, vert: str, frag: str):
        self.ctx = ctx
        self.sim_type = type(sim)
        self.count = self._count(sim)
        self.key = tuple(getattr(sim, f) for f in self.structural)
        self.compute = ctx.compute_shader(_shader(compute))
        self.program = ctx.program(vertex_shader=_shader(vert), fragment_shader=_shader(frag))
        self.buffers: list[moderngl.Buffer] = []
        self.vao: moderngl.VertexArray | None = None

    def _count(self, sim) -> int:
        raise NotImplementedError

    def can_update_to(self, sim) -> bool:
        return type(sim) is self.sim_type and tuple(getattr(sim, f) for f in self.structural) == self.key

    def set_params(self, sim) -> None:
        for name in self.params:
            _set(self.compute, name, getattr(sim, name))

    def step(self) -> None:
        for binding, buf in enumerate(self.buffers):
            buf.bind_to_storage_buffer(binding)
        self.compute.run((self.count + LOCAL_SIZE - 1) // LOCAL_SIZE)
        self.ctx.memory_barrier()

    def draw(self, view: np.ndarray, proj: np.ndarray) -> None:
        _set(self.program, "u_view", view)
        _set(self.program, "u_proj", proj)
        self.vao.render(mode=moderngl.POINTS, vertices=self.count)

    def release(self) -> None:
        for resource in (self.vao, *self.buffers, self.compute, self.program):
            resource.release()


class GpuLorenz(GpuSimulation):
    params = ("sigma", "rho", "beta", "dt", "steps")
    structural = ("num_points", "seed")

    def __init__(self, ctx: moderngl.Context, sim: LorenzAttractor):
        super().__init__(ctx, sim, "lorenz_attractor.comp", "lorenz_attractor.vert", "lorenz_attractor.frag")
        rng = np.random.default_rng(sim.seed)
        points = np.ones((self.count, 4), dtype=np.float32)
        points[:, :3] = rng.standard_normal((self.count, 3)) * 8.0
        positions = ctx.buffer(points.tobytes())
        self.buffers = [positions]
        self.vao = ctx.vertex_array(self.program, [(positions, "4f", "in_position")])

    def _count(self, sim: LorenzAttractor) -> int:
        return sim.num_points


def create_simulation(ctx: moderngl.Context, sim: LorenzAttractor) -> GpuSimulation:
    if isinstance(sim, LorenzAttractor):
        return GpuLorenz(ctx, sim)
    raise TypeError(f"Not a simulation: {sim!r}")
