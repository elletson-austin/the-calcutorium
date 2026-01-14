from enum import Enum, auto
import numpy as np


class Mode(Enum):
    POINTS = auto()
    LINES = auto()
    LINE_STRIP = auto()
    TRIANGLES = auto()
    LINE_LOOP = auto()


class ProgramID(Enum):
    BASIC_3D = auto()
    LORENZ_ATTRACTOR = auto()



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

    def __init__(self, equation_str: str, x_range: tuple = (-10, 10), points: int = 100, output_widget=None):
        """
        Parses a string to create a plottable math function.
        WARNING: This uses eval() and is not safe with untrusted input.
        """
        super().__init__(Mode=Mode.LINE_STRIP)
        self.equation_str = equation_str
        self.x_range = x_range
        self.points = points
        self.output_widget = output_widget
        self.vertices = self._generate_vertices()
        self.mode = Mode.LINE_STRIP
        self.ProgramID = ProgramID.BASIC_3D
        self.is_dirty = False

    def to_dict(self):
        d = super().to_dict()
        d['equation'] = self.equation_str
        return d

    def _generate_vertices(self):
        vertices = []
        if not self.equation_str.strip(): # Check if equation string is empty or just whitespace
            if self.output_widget:
                self.output_widget.append_text("MathFunction: Equation string is empty, cannot generate vertices.")
            return np.array([], dtype=np.float32)

        x_values = np.linspace(self.x_range[0], self.x_range[1], self.points)
        for x in x_values:
            try:
                # WARNING: eval is a security risk. Do not use with untrusted input.
                y = eval(self.equation_str, {"x": x, "np": np})
                z = 0  # For now, plot in the XY plane
                r, g, b = 1, 1, 1  # White color for the plot
                vertices.extend([x, y, z, r, g, b])
            except Exception as e:
                if self.output_widget:
                    self.output_widget.append_text(f"MathFunction: Could not evaluate equation '{self.equation_str}' at x={x}: {e}")
                else:
                    print(f"MathFunction: Could not evaluate equation '{self.equation_str}' at x={x}: {e}")
        return np.array(vertices, dtype=np.float32)

    def regenerate(self, new_equation_str: str):
        self.equation_str = new_equation_str
        self.vertices = self._generate_vertices()
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