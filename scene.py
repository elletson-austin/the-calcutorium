import numpy as np
from render_types import ProgramID, Mode


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
        self.program_id = ProgramID.BASIC_3D

class MathFunction(SceneObject):

    def __init__(self, equation_str: str, x_range: tuple = (-10, 10), points: int = 100):
        """
        Parses a string to create a plottable math function.
        WARNING: This uses eval() and is not safe with untrusted input.
        """
        super().__init__(Mode=Mode.LINE_STRIP)
        self.equation_str = equation_str
        self.x_range = x_range
        self.points = points
        self.vertices = self._generate_vertices()
        self.mode = Mode.LINE_STRIP
        self.program_id = ProgramID.BASIC_3D

    def _generate_vertices(self):
        vertices = []
        x_values = np.linspace(self.x_range[0], self.x_range[1], self.points)
        for x in x_values:
            try:
                # WARNING: eval is a security risk. Do not use with untrusted input.
                y = eval(self.equation_str, {"x": x, "np": np})
                z = 0  # For now, plot in the XY plane
                r, g, b = 1, 1, 1  # White color for the plot
                vertices.extend([x, y, z, r, g, b])
            except Exception as e:
                print(f"Could not evaluate equation at x={x}: {e}")
        return np.array(vertices, dtype=np.float32)