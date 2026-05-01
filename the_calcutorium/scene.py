from abc import ABC, abstractmethod
from enum import Enum, auto
import numpy as np

from .render_types import AXIS_INDEX, axes_for_plane
from .symbolic import SymbolicFunction


class RenderMode(Enum):
    POINTS = auto()
    LINES = auto()
    LINE_STRIP = auto()
    TRIANGLES = auto()
    LINE_LOOP = auto()


class ProgramID(Enum):
    BASIC_3D = auto()
    LORENZ_ATTRACTOR = auto()
    NBODY = auto()
    GRID = auto()
    SURFACE = auto()


class SceneObject(ABC):
    def __init__(
        self,
        render_mode: RenderMode,
        dynamic: bool = False,
        program_id: ProgramID = ProgramID.BASIC_3D,
        visibility: bool = True,
        is_2d: bool = False,
    ):
        self.render_mode = render_mode
        self.dynamic = dynamic
        self.program_id = program_id
        self.visibility = visibility       # arbitrary visibility
        self.is_2d = is_2d                 # visibility determined by dimension
        self.name: str | None = None
        self.vertices = np.array([], dtype=np.float32)
        self.is_dirty = True

    @abstractmethod
    def update(self, *args, **kwargs):
        """Update scene object, e.g. for dynamic elements or LOD."""


class Scene:
    def __init__(self):
        self.objects: list[SceneObject] = []

    def add(self, obj: SceneObject):
        if obj in self.objects:
            raise ValueError("SceneObject already exists")
        self.objects.append(obj)

    def remove(self, obj: SceneObject):
        if obj not in self.objects:
            raise ValueError("SceneObject doesn't exist")
        self.objects.remove(obj)


class Axes(SceneObject):
    def __init__(self, length: float = 10.0):
        super().__init__(render_mode=RenderMode.LINES, is_2d=False)
        self.length = length

        self.vertices = np.array([  # Each line is two points with RGB color
            -length, 0, 0, 1, 0, 0,
             length, 0, 0, 1, 0, 0,
            0, -length, 0, 0, 1, 0,
            0,  length, 0, 0, 1, 0,
            0, 0, -length, 0, 0, 1,
            0, 0,  length, 0, 0, 1,
        ], dtype=np.float32)

    def update(self, **kwargs):
        pass  # Axes are static


class MathFunction(SceneObject, ABC):
    def __init__(
        self,
        symbolic_func: SymbolicFunction,
        render_mode: RenderMode,
        program_id: ProgramID,
        is_2d: bool,
        points: int = 500,
    ):
        super().__init__(render_mode=render_mode, program_id=program_id, is_2d=is_2d)
        self.symbolic_func = symbolic_func
        self.equation_str = symbolic_func.equation_str
        self.points = points

    def regenerate(self, new_equation_str: str):
        """Re-parse and recompile the equation. Raises ValueError on parse failure."""
        self.symbolic_func = SymbolicFunction(new_equation_str)
        self.equation_str = new_equation_str
        self.name = new_equation_str
        self.update()

    @abstractmethod
    def update(self, *args, **kwargs):
        pass


class LinePlot(MathFunction):
    def __init__(self, symbolic_func: SymbolicFunction, points: int = 500):
        super().__init__(symbolic_func, RenderMode.LINE_STRIP, ProgramID.BASIC_3D, True, points)
        self.current_plane = None
        self.current_domain_range = None
        self.reset_to_3d()

    def reset_to_3d(self, domain_range: tuple = (-10, 10)):
        """Regenerate vertices for 3D display using the equation's actual variables."""
        self.current_plane = None
        self.current_domain_range = None
        if self.symbolic_func.get_num_domain_vars() != 1:
            self.vertices = np.array([], dtype=np.float32)
            self.is_dirty = True
            return
        indep = str(self.symbolic_func.get_domain_vars()[0])
        output = str(self.symbolic_func.get_output_var())
        self._generate_line_vertices(domain_range, indep, output)

    def update(self, plane: str = "xy", h_range: tuple = (-10, 10), v_range: tuple = (-10, 10)):
        if self.symbolic_func.get_num_domain_vars() != 1:
            self.vertices = np.array([], dtype=np.float32)
            self.is_dirty = True
            return

        indep_var_str = str(self.symbolic_func.get_domain_vars()[0])
        output_var_str = str(self.symbolic_func.get_output_var())

        axes = axes_for_plane(plane) or ("x", "y", "z")
        h_axis, v_axis, const_axis = axes

        # Hide if either variable is out-of-plane
        if indep_var_str == const_axis or output_var_str == const_axis:
            self.vertices = np.array([], dtype=np.float32)
            self.is_dirty = True
            return

        if indep_var_str == h_axis:
            domain_range = h_range
        elif indep_var_str == v_axis:
            domain_range = v_range
        else:
            self.vertices = np.array([], dtype=np.float32)
            self.is_dirty = True
            return

        if (
            self.current_plane != plane
            or self.current_domain_range is None
            or not np.allclose(self.current_domain_range, domain_range, atol=1e-2)
        ):
            self.current_plane = plane
            self.current_domain_range = domain_range
            self._generate_line_vertices(domain_range, indep_var_str, output_var_str)

    def _generate_line_vertices(self, domain_range: tuple, indep_axis: str, output_axis: str):
        domain_values = np.linspace(domain_range[0], domain_range[1], self.points).astype(np.float32)

        try:
            out_values = self.symbolic_func.evaluate(domain_values)
        except Exception:
            self.vertices = np.array([], dtype=np.float32)
            self.is_dirty = True
            return

        out_values = np.asarray(out_values, dtype=np.float32)
        if out_values.shape != domain_values.shape:
            out_values = np.broadcast_to(out_values, domain_values.shape).astype(np.float32)

        mask = np.isfinite(out_values)
        domain_values = domain_values[mask]
        out_values = out_values[mask]
        n = len(domain_values)

        if n == 0:
            self.vertices = np.array([], dtype=np.float32)
            self.is_dirty = True
            return

        coords = {
            "x": np.zeros(n, dtype=np.float32),
            "y": np.zeros(n, dtype=np.float32),
            "z": np.zeros(n, dtype=np.float32),
        }
        coords[indep_axis] = domain_values
        coords[output_axis] = out_values

        positions = np.stack([coords["x"], coords["y"], coords["z"]], axis=1)
        colors = np.ones((n, 3), dtype=np.float32)
        self.vertices = np.hstack([positions, colors]).astype(np.float32).flatten()
        self.is_dirty = True


class SurfacePlot(MathFunction):
    def __init__(self, symbolic_func: SymbolicFunction, points: int = 100):
        super().__init__(symbolic_func, RenderMode.TRIANGLES, ProgramID.SURFACE, False, points)
        self.update()

    def update(self, domain1_range=(-10, 10), domain2_range=(-10, 10)):
        if self.symbolic_func.get_num_domain_vars() != 2:
            self.vertices = np.array([], dtype=np.float32)
            self.is_dirty = True
            return
        self._generate_surface_vertices(domain1_range, domain2_range)

    def _generate_surface_vertices(self, domain1_range, domain2_range):
        domain_vars = self.symbolic_func.get_domain_vars()
        domain_var1_name = str(domain_vars[0])
        domain_var2_name = str(domain_vars[1])
        output_var_name = str(self.symbolic_func.get_output_var())

        all_vars = {domain_var1_name, domain_var2_name, output_var_name}
        if len(all_vars) != 3 or not all_vars.issubset({"x", "y", "z"}):
            raise ValueError(
                "Surface plot must be of the form f(a,b)=c where a,b,c are unique x,y,z variables."
            )

        domain1_vals = np.linspace(domain1_range[0], domain1_range[1], self.points)
        domain2_vals = np.linspace(domain2_range[0], domain2_range[1], self.points)
        grid_domain1, grid_domain2 = np.meshgrid(domain1_vals, domain2_vals)

        try:
            grid_output = self.symbolic_func.evaluate(grid_domain1, grid_domain2)
        except Exception as e:
            raise ValueError(f"Error evaluating surface function: {e}") from e

        grid_output = np.asarray(grid_output, dtype=np.float32)
        if grid_output.shape != grid_domain1.shape:
            grid_output = np.broadcast_to(grid_output, grid_domain1.shape).astype(np.float32)

        grids = {
            domain_var1_name: grid_domain1.astype(np.float32),
            domain_var2_name: grid_domain2.astype(np.float32),
            output_var_name: grid_output,
        }
        points3d = np.stack([grids["x"], grids["y"], grids["z"]], axis=-1)

        # Cell corners: TL, TR, BL, BR (shape: (n, n, 3))
        p1 = points3d[:-1, :-1, :]
        p2 = points3d[:-1, 1:, :]
        p3 = points3d[1:, :-1, :]
        p4 = points3d[1:, 1:, :]

        n_a = self._face_normals(p1, p3, p2)
        n_b = self._face_normals(p2, p3, p4)

        color = np.array([1.0, 0.2, 0.2], dtype=np.float32)
        color_arr = np.broadcast_to(color, p1.shape)

        # Each cell emits 6 vertices: (p1,p3,p2) then (p2,p3,p4)
        triangles = np.stack(
            [
                np.concatenate([p1, n_a, color_arr], axis=-1),
                np.concatenate([p3, n_a, color_arr], axis=-1),
                np.concatenate([p2, n_a, color_arr], axis=-1),
                np.concatenate([p2, n_b, color_arr], axis=-1),
                np.concatenate([p3, n_b, color_arr], axis=-1),
                np.concatenate([p4, n_b, color_arr], axis=-1),
            ],
            axis=2,
        )
        self.vertices = triangles.astype(np.float32).flatten()
        self.is_dirty = True

    @staticmethod
    def _face_normals(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n, axis=-1, keepdims=True)
        norm = np.where(norm > 1e-6, norm, 1.0)
        return (n / norm).astype(np.float32)


class Grid(SceneObject):
    def __init__(self, h_range=(-250, 250), v_range=(-250, 250), spacing=1.0, plane="xy"):
        super().__init__(render_mode=RenderMode.LINES, is_2d=True, program_id=ProgramID.GRID)
        self.h_range = h_range
        self.v_range = v_range
        self.spacing = spacing
        self.plane = plane
        self.default_h_range = h_range
        self.default_v_range = v_range
        self.default_spacing = spacing
        self.major_interval = 5  # Major gridline every N minor lines
        self.labels = {}
        self.update()

    def update(self, h_range=None, v_range=None, plane="xy"):
        h_range = h_range if h_range is not None else self.h_range
        v_range = v_range if v_range is not None else self.v_range

        view_size = min(h_range[1] - h_range[0], v_range[1] - v_range[0])
        if view_size <= 0:
            return

        target_spacing = view_size / 10.0
        power_of_10 = 10 ** np.floor(np.log10(target_spacing))
        rescaled_spacing = target_spacing / power_of_10

        if rescaled_spacing < 1.5:
            new_spacing = 1 * power_of_10
        elif rescaled_spacing < 3.5:
            new_spacing = 2 * power_of_10
        elif rescaled_spacing < 7.5:
            new_spacing = 5 * power_of_10
        else:
            new_spacing = 10 * power_of_10

        if (
            not np.isclose(self.spacing, new_spacing)
            or not np.allclose(self.h_range, h_range)
            or not np.allclose(self.v_range, v_range)
            or self.plane != plane
        ):
            self.spacing = new_spacing
            self.h_range = h_range
            self.v_range = v_range
            self.plane = plane

            self.vertices = self._generate_vertices_from_ranges()
            self.is_dirty = True

    def _generate_vertices_from_ranges(self):
        vertices = []
        minor_color = [0.4, 0.4, 0.4]
        major_color = [0.7, 0.7, 0.7]

        start_h = self.spacing * np.floor(self.h_range[0] / self.spacing)
        end_h = self.spacing * np.ceil(self.h_range[1] / self.spacing)
        start_v = self.spacing * np.floor(self.v_range[0] / self.spacing)
        end_v = self.spacing * np.ceil(self.v_range[1] / self.spacing)

        axes = axes_for_plane(self.plane) or ("x", "y", "z")
        h_axis, v_axis, _ = axes
        h_idx = AXIS_INDEX[h_axis]
        v_idx = AXIS_INDEX[v_axis]

        self.labels = {"h_labels": [], "v_labels": []}

        # Lines along the horizontal axis (parallel to h_axis)
        for i in np.arange(start_v, end_v + self.spacing * 0.1, self.spacing):
            is_major = (abs(i) < 1e-6) or (abs(i % (self.spacing * self.major_interval)) < self.spacing * 0.1)
            color = major_color if is_major else minor_color

            p1, p2 = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            p1[h_idx], p2[h_idx] = self.h_range[0], self.h_range[1]
            p1[v_idx], p2[v_idx] = i, i

            vertices.extend(p1 + color + [is_major])
            vertices.extend(p2 + color + [is_major])

            if is_major:
                self.labels["h_labels"].append((i, p1[h_idx], p2[h_idx], v_idx))

        # Lines along the vertical axis (parallel to v_axis)
        for i in np.arange(start_h, end_h + self.spacing * 0.1, self.spacing):
            is_major = (abs(i) < 1e-6) or (abs(i % (self.spacing * self.major_interval)) < self.spacing * 0.1)
            color = major_color if is_major else minor_color

            p1, p2 = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            p1[v_idx], p2[v_idx] = self.v_range[0], self.v_range[1]
            p1[h_idx], p2[h_idx] = i, i

            vertices.extend(p1 + color + [is_major])
            vertices.extend(p2 + color + [is_major])

            if is_major:
                self.labels["v_labels"].append((i, p1[v_idx], p2[v_idx], h_idx))

        return np.array(vertices, dtype=np.float32)

    def set_to_default(self):
        if not (
            np.allclose(self.h_range, self.default_h_range)
            and np.allclose(self.v_range, self.default_v_range)
            and np.isclose(self.spacing, self.default_spacing)
            and self.plane == "xy"
        ):
            self.h_range = self.default_h_range
            self.v_range = self.default_v_range
            self.spacing = self.default_spacing
            self.plane = "xy"
            self.update()
            self.is_dirty = True
