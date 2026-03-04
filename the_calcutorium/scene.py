from abc import ABC, abstractmethod
from enum import Enum, auto
import numpy as np

from the_calcutorium.symbolic import SymbolicFunction


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
    def __init__(self, 
                 RenderMode: RenderMode, 
                 dynamic: bool = False,
                 ProgramID: ProgramID = ProgramID.BASIC_3D,
                 visibility:bool = True, 
                 is_2d = False):
        self.RenderMode = RenderMode
        self.dynamic = dynamic
        self.ProgramID = ProgramID
        self.visibility = visibility # arbitrary visibility 
        self.is_2d = is_2d             # visibility determined by dimension
        self.name = None
        self.vertices = np.array([], dtype=np.float32)
        self.is_dirty = True

    def to_dict(self):
        return {'type': self.__class__.__name__}

    @abstractmethod
    def update(self, **kwargs):
        """Update scene object, e.g. for dynamic elements or LOD."""
        pass


class Scene:
    def __init__(self):
        self.objects: list[SceneObject] = []

    def add(self, obj: SceneObject):
        if obj in self.objects:
            raise ValueError(f"SceneObject already exists")
        self.objects.append(obj)

    def remove(self, obj: SceneObject):
        if obj not in self.objects:
            raise ValueError(f"SceneObject doesn't exist")
        self.objects.remove(obj)

    def to_dict(self):
        from .simulations import LorenzAttractor
        # Don't save the axes, it's a default part of the scene
        return [obj.to_dict() for obj in self.objects if isinstance(obj, (MathFunction, LorenzAttractor))]

    def from_dict(self, scene_data):
        # Clear existing objects except axes
        self.objects = [obj for obj in self.objects if isinstance(obj, Axes)]
        for item_data in scene_data:
            if item_data['type'] in ['LinePlot', 'SurfacePlot', 'MathFunction']: # Compatibility with old saves
                try:
                    symbolic_func = SymbolicFunction(item_data['equation'])
                    if symbolic_func.get_num_domain_vars() == 1:
                        new_func = LinePlot(symbolic_func)
                    elif symbolic_func.get_num_domain_vars() == 2:
                        new_func = SurfacePlot(symbolic_func)
                    else:
                        continue # Skip functions that are not 1D or 2D
                    
                    new_func.name = item_data['equation']
                    self.objects.append(new_func)
                except ValueError:
                    # Ignore functions that fail to parse
                    continue

            elif item_data['type'] == 'LorenzAttractor':
                from .simulations import LorenzAttractor
                # You might want to save/load parameters for this in the future
                lorenz = LorenzAttractor()
                lorenz.name = "Lorenz Attractor"
                self.objects.append(lorenz)


class Axes(SceneObject):

    def __init__(self, length: float = 10.0):
        super().__init__(RenderMode=RenderMode.LINES, is_2d=False)
        self.length = length

        self.vertices = np.array([  # Each line is two points with RGB color
            -length, 0, 0, 1, 0, 0, # X axis (red)
             length, 0, 0, 1, 0, 0, # X axis (red)
            0, -length, 0, 0, 1, 0, # Y axis (green)
            0,  length, 0, 0, 1, 0, # Y axis (green)
            0, 0, -length, 0, 0, 1, # Z axis (blue)
            0, 0,  length, 0, 0, 1, # Z axis (blue)
        ], dtype=np.float32)

        self.render_mode = RenderMode.LINES
        self.dynamic = False
        self.program_id = ProgramID.BASIC_3D

    def update(self, **kwargs):
        pass # Axes are static


class MathFunction(SceneObject, ABC):
    def __init__(self, symbolic_func: SymbolicFunction, RenderMode: RenderMode, ProgramID: ProgramID, is_2d: bool, points: int = 500):
        super().__init__(RenderMode=RenderMode, ProgramID=ProgramID, is_2d=is_2d)
        self.symbolic_func = symbolic_func
        self.equation_str = symbolic_func.equation_str
        self.points = points
        self.is_dirty = True

    def to_dict(self):
        d = super().to_dict()
        d['equation'] = self.equation_str
        return d
    
    def regenerate(self, new_equation_str: str):
        try:
            self.symbolic_func = SymbolicFunction(new_equation_str)
            self.equation_str = new_equation_str
            self.name = new_equation_str
            self.update() # Re-generate vertices
        except ValueError as e:
            # Handle parsing error if needed
            print(f"Error regenerating function: {e}")

    @abstractmethod
    def update(self, **kwargs):
        pass


class LinePlot(MathFunction):
    def __init__(self, symbolic_func: SymbolicFunction, points: int = 500):
        super().__init__(symbolic_func, RenderMode.LINE_STRIP, ProgramID.BASIC_3D, True, points)
        self.current_plane = None
        self.current_domain_range = None
        self.update() # Initial vertex generation

    def update(self, plane: str = 'xy', h_range: tuple = (-10, 10), v_range: tuple = (-10, 10)):
        if self.symbolic_func.get_num_domain_vars() != 1:
            self.vertices = np.array([], dtype=np.float32)
            self.is_dirty = True
            return

        indep_var_str = str(self.symbolic_func.get_domain_vars()[0])
        output_var_str = str(self.symbolic_func.get_output_var())

        plane_axis_map = {'xy': ('x', 'y', 'z'), 'xz': ('x', 'z', 'y'), 'yz': ('z', 'y', 'x')}
        h_axis, v_axis, const_axis = plane_axis_map.get(plane, ('x', 'y', 'z'))

        if indep_var_str == h_axis:
            plot_indep_axis, plot_output_axis, domain_range = h_axis, v_axis, h_range
        elif indep_var_str == v_axis:
            plot_indep_axis, plot_output_axis, domain_range = v_axis, h_axis, v_range
        else:
            self.vertices = np.array([], dtype=np.float32)
            self.is_dirty = True
            return

        if self.current_plane != plane or \
           self.current_domain_range is None or \
           not np.allclose(self.current_domain_range, domain_range, atol=1e-2):
            
            self.current_plane = plane
            self.current_domain_range = domain_range
            self._generate_line_vertices(domain_range, plot_indep_axis, plot_output_axis)

    def _generate_line_vertices(self, domain_range: tuple, indep_axis: str, output_axis: str):
        vertices = []
        domain_values = np.linspace(domain_range[0], domain_range[1], self.points)
        
        for val in domain_values:
            try:
                out_val = self.symbolic_func.evaluate(val)
                if out_val is None: continue

                point = {'x': 0.0, 'y': 0.0, 'z': 0.0}
                point[indep_axis] = val
                point[output_axis] = out_val

                vertices.extend([point['x'], point['y'], point['z'], 1.0, 1.0, 1.0]) # Position and Color
            except Exception:
                pass 
        
        self.vertices = np.array(vertices, dtype=np.float32)
        self.is_dirty = True


class SurfacePlot(MathFunction):
    def __init__(self, symbolic_func: SymbolicFunction, points: int = 100):
        super().__init__(symbolic_func, RenderMode.TRIANGLES, ProgramID.SURFACE, False, points)
        self.update() # Initial vertex generation

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
        if len(all_vars) != 3 or not all_vars.issubset({'x', 'y', 'z'}):
            raise ValueError(f"Surface plot must be of the form f(a,b)=c where a,b,c are unique x,y,z variables.")
        
        domain1_vals = np.linspace(domain1_range[0], domain1_range[1], self.points)
        domain2_vals = np.linspace(domain2_range[0], domain2_range[1], self.points)
        grid_domain1, grid_domain2 = np.meshgrid(domain1_vals, domain2_vals)

        try:
            grid_output = self.symbolic_func.evaluate(grid_domain1, grid_domain2)
        except Exception as e:
            raise ValueError(f"Error evaluating surface function: {e}") from e
        
        grids = {
            domain_var1_name: grid_domain1,
            domain_var2_name: grid_domain2,
            output_var_name: grid_output
        }
        grid_x, grid_y, grid_z = grids['x'], grids['y'], grids['z']

        vertices = []
        color = [1.0, 0.2, 0.2] 

        for i in range(self.points - 1):
            for j in range(self.points - 1):
                p1 = np.array([grid_x[i, j], grid_y[i, j], grid_z[i, j]])
                p2 = np.array([grid_x[i, j + 1], grid_y[i, j + 1], grid_z[i, j + 1]])
                p3 = np.array([grid_x[i + 1, j], grid_y[i + 1, j], grid_z[i + 1, j]])
                p4 = np.array([grid_x[i + 1, j + 1], grid_y[i + 1, j + 1], grid_z[i + 1, j + 1]])

                self._add_triangle_with_normal(vertices, p1, p3, p2, color)
                self._add_triangle_with_normal(vertices, p2, p3, p4, color)

        self.vertices = np.array(vertices, dtype=np.float32)
        self.is_dirty = True

    def _add_triangle_with_normal(self, vertices_list: list, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, color: list):
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n > 1e-6:
            n /= norm_n

        vertices_list.extend([*p1, *n, *color])
        vertices_list.extend([*p2, *n, *color])
        vertices_list.extend([*p3, *n, *color])
    
class Grid(SceneObject):
    def __init__(self, h_range=(-250, 250), v_range=(-250, 250), spacing=1.0, plane='xy'):
        super().__init__(RenderMode=RenderMode.LINES, is_2d=True, ProgramID=ProgramID.GRID)
        self.h_range = h_range
        self.v_range = v_range
        self.spacing = spacing
        self.plane = plane
        self.default_h_range = h_range
        self.default_v_range = v_range
        self.default_spacing = spacing
        self.major_interval = 5  # Major gridline every N minor lines
        self.labels = {}  # Will store label info for rendering
        self.update()

    def update(self, h_range=None, v_range=None, plane='xy'):
        h_range = h_range if h_range is not None else self.h_range
        v_range = v_range if v_range is not None else self.v_range

        view_size = min(h_range[1] - h_range[0], v_range[1] - v_range[0])
        
        if view_size <= 0:
            return
            
        target_spacing = view_size / 10.0
        
        power_of_10 = 10**np.floor(np.log10(target_spacing))
        rescaled_spacing = target_spacing / power_of_10

        if rescaled_spacing < 1.5:
            new_spacing = 1 * power_of_10
        elif rescaled_spacing < 3.5:
            new_spacing = 2 * power_of_10
        elif rescaled_spacing < 7.5:
            new_spacing = 5 * power_of_10
        else:
            new_spacing = 10 * power_of_10
            
        if not np.isclose(self.spacing, new_spacing) or \
           not np.allclose(self.h_range, h_range) or \
           not np.allclose(self.v_range, v_range) or \
           self.plane != plane:
            
            self.spacing = new_spacing
            self.h_range = h_range
            self.v_range = v_range
            self.plane = plane
            
            self.vertices = self._generate_vertices_from_ranges()
            self.is_dirty = True

    def _generate_vertices_from_ranges(self):
        vertices = []
        minor_color = [0.4, 0.4, 0.4]  # Darker grey for minor gridlines
        major_color = [0.7, 0.7, 0.7]  # Brighter grey for major gridlines

        start_h = self.spacing * np.floor(self.h_range[0] / self.spacing)
        end_h = self.spacing * np.ceil(self.h_range[1] / self.spacing)
        
        start_v = self.spacing * np.floor(self.v_range[0] / self.spacing)
        end_v = self.spacing * np.ceil(self.v_range[1] / self.spacing)

        axis_map = {
            'xy': (0, 1, 2),
            'xz': (0, 2, 1),
            'yz': (2, 1, 0),
        }
        h_idx, v_idx, const_idx = axis_map.get(self.plane, (0, 1, 2))
        
        # Generate labels dictionary
        self.labels = {'h_labels': [], 'v_labels': []}
        
        # Lines along the horizontal axis (parallel to h_axis)
        for i in np.arange(start_v, end_v + self.spacing * 0.1, self.spacing):
            is_major = (abs(i) < 1e-6) or (abs(i % (self.spacing * self.major_interval)) < self.spacing * 0.1)
            color = major_color if is_major else minor_color
            
            p1, p2 = [0, 0, 0], [0, 0, 0]
            p1[h_idx], p2[h_idx] = self.h_range[0], self.h_range[1]
            p1[v_idx], p2[v_idx] = i, i
            
            vertices.extend(p1 + color + [is_major])
            vertices.extend(p2 + color + [is_major])
            
            if is_major: self.labels['h_labels'].append((i, p1[h_idx], p2[h_idx], v_idx))
        
        # Lines along the vertical axis (parallel to v_axis)
        for i in np.arange(start_h, end_h + self.spacing * 0.1, self.spacing):
            is_major = (abs(i) < 1e-6) or (abs(i % (self.spacing * self.major_interval)) < self.spacing * 0.1)
            color = major_color if is_major else minor_color

            p1, p2 = [0, 0, 0], [0, 0, 0]
            p1[v_idx], p2[v_idx] = self.v_range[0], self.v_range[1]
            p1[h_idx], p2[h_idx] = i, i
            
            vertices.extend(p1 + color + [is_major])
            vertices.extend(p2 + color + [is_major])
            
            if is_major: self.labels['v_labels'].append((i, p1[v_idx], p2[v_idx], h_idx))
            
        return np.array(vertices, dtype=np.float32)

    def set_to_default(self):
        if not (np.allclose(self.h_range, self.default_h_range) and \
                np.allclose(self.v_range, self.default_v_range) and \
                np.isclose(self.spacing, self.default_spacing) and \
                self.plane == 'xy'):
            self.h_range, self.v_range, self.spacing, self.plane = self.default_h_range, self.default_v_range, self.default_spacing, 'xy'
            self.update()
            self.is_dirty = True


# Re-export so "from .scene import LorenzAttractor, NBody" still works
from .simulations import LorenzAttractor, NBody
