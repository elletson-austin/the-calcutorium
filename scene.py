from enum import Enum, auto
import numpy as np
from sympy import symbols, lambdify, sympify, SympifyError
from render_types import CameraMode


class Mode(Enum):
    POINTS = auto()
    LINES = auto()
    LINE_STRIP = auto()
    TRIANGLES = auto()
    LINE_LOOP = auto()


class ProgramID(Enum):
    BASIC_3D = auto()
    LORENZ_ATTRACTOR = auto()
    GRID = auto()



class SceneObject:
    def __init__(self, 
                 Mode: Mode, 
                 visibility:bool = True, 
                 dynamic: bool = False,
                 ProgramID: ProgramID = ProgramID.BASIC_3D):
        self.visibility = visibility
        self.dynamic = dynamic
        self.ProgramID = ProgramID
        self.Mode = Mode

    def to_dict(self):
        return {'type': self.__class__.__name__}


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
        # Don't save the axes, it's a default part of the scene
        return [obj.to_dict() for obj in self.objects if isinstance(obj, (MathFunction, LorenzAttractor))]

    def from_dict(self, scene_data):
        # Clear existing objects except axes
        self.objects = [obj for obj in self.objects if isinstance(obj, Axes)]
        for item_data in scene_data:
            if item_data['type'] == 'MathFunction':
                new_func = MathFunction(item_data['equation'])
                new_func.name = item_data['equation']
                self.objects.append(new_func)
            elif item_data['type'] == 'LorenzAttractor':
                # You might want to save/load parameters for this in the future
                lorenz = LorenzAttractor()
                lorenz.name = "Lorenz Attractor"
                self.objects.append(lorenz)


class Axes(SceneObject):

    def __init__(self, length: float = 10.0):

        super().__init__(Mode=Mode.LINES)
        self.length = length
        # Each line is two points with RGB color

        self.vertices = np.array([
            # X axis (red)
            -length, 0, 0, 1, 0, 0,
             length, 0, 0, 1, 0, 0,
            # Y axis (green)
            0, -length, 0, 0, 1, 0,
            0,  length, 0, 0, 1, 0,
            # Z axis (blue)
            0, 0, -length, 0, 0, 1,
            0, 0,  length, 0, 0, 1,
        ], dtype=np.float32)

        self.mode = Mode.LINES
        self.dynamic = False
        self.ProgramID = ProgramID.BASIC_3D

class MathFunction(SceneObject):

    def __init__(self, equation_str: str, x_range: tuple = (-10, 10), points: int = 1000, output_widget=None):
        """
        Parses a string to create a plottable math function.
        """
        super().__init__(Mode=Mode.LINE_STRIP)
        self.equation_str = equation_str
        self.x_range = x_range
        self.points = points
        self.output_widget = output_widget
        self._x_symbol = symbols('x') # Define the symbol 'x' for sympy
        self._sympy_expr = None
        self._callable_func = None
        self._parse_and_lambdify() # Parse and lambdify the initial equation
        self.vertices = self._generate_vertices()
        self.mode = Mode.LINE_STRIP
        self.ProgramID = ProgramID.BASIC_3D
        self.is_dirty = False

    def _parse_and_lambdify(self):
        """
        Parses the equation string into a sympy expression and then lambdifies it.
        """
        if not self.equation_str.strip():
            self._sympy_expr = None
            self._callable_func = None
            return

        try:
            # Use sympify to parse the string into a sympy expression
            # Use evaluate=False to prevent immediate evaluation of expressions like sin(pi/2)
            self._sympy_expr = sympify(self.equation_str, evaluate=False) 
            # Lambdify the sympy expression into a callable function for numerical evaluation
            self._callable_func = lambdify(self._x_symbol, self._sympy_expr, 'numpy')
        except SympifyError as e:
            if self.output_widget:
                self.output_widget.append_text(f"MathFunction: Error parsing equation '{self.equation_str}': {e}")
            else:
                print(f"MathFunction: Error parsing equation '{self.equation_str}': {e}")
            self._sympy_expr = None
            self._callable_func = None
        except Exception as e:
            if self.output_widget:
                self.output_widget.append_text(f"MathFunction: Unexpected error during parsing or lambdifying '{self.equation_str}': {e}")
            else:
                print(f"MathFunction: Unexpected error during parsing or lambdifying '{self.equation_str}': {e}")
            self._sympy_expr = None
            self._callable_func = None

    def to_dict(self):
        d = super().to_dict()
        d['equation'] = self.equation_str
        return d

    def _generate_vertices(self):
        vertices = []
        if not self.equation_str.strip() or self._callable_func is None:
            if self.output_widget and not self.equation_str.strip():
                self.output_widget.append_text("MathFunction: Equation string is empty, cannot generate vertices.")
            return np.array([], dtype=np.float32)

        x_values = np.linspace(self.x_range[0], self.x_range[1], self.points)
        for x_val in x_values:
            try:
                y = self._callable_func(x_val)
                z = 0  # For now, plot in the XY plane
                r, g, b = 1, 1, 1  # White color for the plot
                vertices.extend([x_val, y, z, r, g, b])
            except Exception as e:
                if self.output_widget:
                    self.output_widget.append_text(f"MathFunction: Could not evaluate equation '{self.equation_str}' at x={x_val}: {e}")
                else:
                    print(f"MathFunction: Could not evaluate equation '{self.equation_str}' at x={x_val}: {e}")
        return np.array(vertices, dtype=np.float32)

    def regenerate(self, new_equation_str: str):
        self.equation_str = new_equation_str
        self._parse_and_lambdify() # Re-parse and re-lambdify the new equation
        self.vertices = self._generate_vertices()
        self.is_dirty = True

    def set_x_range(self, x_range: tuple):
        # Add a small tolerance to avoid regenerating on tiny pans
        if not np.allclose(self.x_range, x_range, atol=1e-3):
            self.x_range = x_range
            self.vertices = self._generate_vertices()
            self.is_dirty = True
    
class Grid(SceneObject):
    def __init__(self, x_range=(-250, 250), y_range=(-250, 250), spacing=1.0):
        super().__init__(Mode=Mode.LINES)
        self.x_range = x_range
        self.y_range = y_range
        self.spacing = spacing
        self.default_x_range = x_range
        self.default_y_range = y_range
        self.default_spacing = spacing
        self.vertices = self._generate_vertices_from_ranges()
        self.ProgramID = ProgramID.GRID
        self.is_dirty = False

    def _generate_vertices_from_ranges(self):
        vertices = []
        color = [0.5, 0.5, 0.5]  # Grey color for grid lines

        start_x = self.spacing * np.floor(self.x_range[0] / self.spacing)
        end_x = self.spacing * np.ceil(self.x_range[1] / self.spacing)
        
        start_y = self.spacing * np.floor(self.y_range[0] / self.spacing)
        end_y = self.spacing * np.ceil(self.y_range[1] / self.spacing)
        
        # Horizontal lines
        for i in np.arange(start_y, end_y, self.spacing):
            vertices.extend([self.x_range[0], i, 0] + color)
            vertices.extend([self.x_range[1], i, 0] + color)

        # Vertical lines
        for i in np.arange(start_x, end_x, self.spacing):
            vertices.extend([i, self.y_range[0], 0] + color)
            vertices.extend([i, self.y_range[1], 0] + color)
            
        return np.array(vertices, dtype=np.float32)

    def set_ranges(self, x_range, y_range):
        view_size = min(x_range[1] - x_range[0], y_range[1] - y_range[0])
        
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
           not np.allclose(self.x_range, x_range) or \
           not np.allclose(self.y_range, y_range):
            
            self.spacing = new_spacing
            self.x_range = x_range
            self.y_range = y_range
            
            self.vertices = self._generate_vertices_from_ranges()
            self.is_dirty = True

    def set_to_default(self):
        if not (np.allclose(self.x_range, self.default_x_range) and \
                np.allclose(self.y_range, self.default_y_range) and \
                np.isclose(self.spacing, self.default_spacing)):
            self.x_range = self.default_x_range
            self.y_range = self.default_y_range
            self.spacing = self.default_spacing
            self.vertices = self._generate_vertices_from_ranges()
            self.is_dirty = True

class LorenzAttractor(SceneObject):
    def __init__(self, num_points: int = 100_000, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0, dt: float = 0.001, steps:int = 5):
        super().__init__(Mode=Mode.POINTS, ProgramID=ProgramID.LORENZ_ATTRACTOR, dynamic=True)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.steps = steps
        self.num_points = num_points
        self.vertices = self.create_initial_points(num_points=self.num_points)

    def to_dict(self):
        return super().to_dict()

    def create_initial_points(self,num_points: int) -> np.ndarray:
        # Note: We are not including color data here, as the fragment shader will assign a color.
        # The vertex format is just position (x, y, z, w).
        initial_points = np.random.randn(num_points, 4).astype(np.float32)
        initial_points[:, :3] *= 2.0
        initial_points[:, :3] += [1.0, 1.0, 1.0]
        initial_points[:, 3] = 1.0
        return initial_points