"""The scene: WHAT is on screen, as plain data. Nothing here knows how to draw.

Scene objects are immutable. To change one, build a new one and `scene.replace`
the old with it. The scene announces every change through its events, so the UI
and the viewport can react without anyone passing callbacks around.
"""

from dataclasses import dataclass
from typing import ClassVar

from .events import Event
from .mathcore import Equation, MathError

Color = tuple[float, float, float]
Range = tuple[float, float]


# ---- plots ----------------------------------------------------------------------
# One class per kind of plot. Each holds exactly the fields that kind needs, and
# checks in __post_init__ that its equation fits, so an invalid plot can't exist.
# The same equation can be shown several ways at once (e.g. a Surface and a Contour).

@dataclass(frozen=True, eq=False)  # eq=False: two plots with the same equation are still two plots
class Plot:
    """What every plot has. Not used directly: use Curve, Surface, or Contour."""
    kind: ClassVar[str] = "plot"   # short name, shown in the UI and used by commands
    inputs: ClassVar[tuple[int, ...]] = ()  # allowed numbers of input variables

    equation: Equation
    color: Color
    domain: Range = (-10.0, 10.0)  # extent of each input in 3D (2D views use the visible range)

    def __post_init__(self):
        n = len(self.equation.inputs)
        if n not in self.inputs:
            allowed = " or ".join(str(i) for i in self.inputs)
            raise MathError(f"A {self.kind} needs {allowed} input variable(s); '{self.text}' has {n}")

    @property
    def text(self) -> str:
        return self.equation.text


@dataclass(frozen=True, eq=False)
class Curve(Plot):
    """y = x^2, x = y^2, z = sin(x), y = 3"""
    kind: ClassVar[str] = "curve"
    inputs: ClassVar[tuple[int, ...]] = (0, 1)


@dataclass(frozen=True, eq=False)
class Surface(Plot):
    """z = f(x, y) as a shaded surface (3D only)."""
    kind: ClassVar[str] = "surface"
    inputs: ClassVar[tuple[int, ...]] = (2,)


@dataclass(frozen=True, eq=False)
class Contour(Plot):
    """Level curves f(x, y) = c. In 3D each curve floats at its height c."""
    kind: ClassVar[str] = "contour"
    inputs: ClassVar[tuple[int, ...]] = (2,)

    levels: tuple[float, ...] | None = None  # None: pick round-numbered levels automatically


def graph(equation: Equation, color: Color) -> Curve | Surface:
    """The natural picture of an equation: a curve for 0–1 inputs, a surface for 2."""
    return Surface(equation, color) if len(equation.inputs) == 2 else Curve(equation, color)


@dataclass(frozen=True, eq=False)
class LorenzAttractor:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    dt: float = 0.001
    steps: int = 5          # integration steps per frame
    num_points: int = 100_000
    seed: int = 0           # starting points are random, but reproducible


@dataclass(frozen=True, eq=False)
class NBody:
    num_bodies: int = 4000
    G: float = 1.0
    dt: float = 0.01
    softening: float = 1.0
    steps: int = 5
    seed: int = 0


SceneObject = Plot | LorenzAttractor | NBody


class Scene:
    def __init__(self) -> None:
        self._objects: list[SceneObject] = []
        self.added = Event()      # (obj)
        self.removed = Event()    # (obj)
        self.replaced = Event()   # (old, new)

    @property
    def objects(self) -> tuple[SceneObject, ...]:
        return tuple(self._objects)

    def plots(self) -> list[Plot]:
        return [o for o in self._objects if isinstance(o, Plot)]

    def find(self, kind: type) -> SceneObject | None:
        return next((o for o in self._objects if isinstance(o, kind)), None)

    def add(self, obj: SceneObject) -> None:
        if obj in self._objects:
            raise ValueError("Object is already in the scene")
        self._objects.append(obj)
        self.added.emit(obj)

    def remove(self, obj: SceneObject) -> None:
        self._objects.remove(obj)
        self.removed.emit(obj)

    def replace(self, old: SceneObject, new: SceneObject) -> None:
        self._objects[self._objects.index(old)] = new
        self.replaced.emit(old, new)
