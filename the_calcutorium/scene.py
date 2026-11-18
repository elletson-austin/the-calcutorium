"""The scene: WHAT is on screen, as plain data. Nothing here knows how to draw.

Scene objects are immutable. To change one, build a new one and `scene.replace`
the old with it. The scene announces every change through its events, so the UI
and the viewport can react without anyone passing callbacks around.
"""

from dataclasses import dataclass

from .events import Event
from .mathcore import Equation

Color = tuple[float, float, float]


@dataclass(frozen=True, eq=False)  # eq=False: two plots with the same equation are still two plots
class Plot:
    equation: Equation
    color: Color

    @property
    def text(self) -> str:
        return self.equation.text

    @property
    def is_surface(self) -> bool:
        return len(self.equation.inputs) == 2


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
