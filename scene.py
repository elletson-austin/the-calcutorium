import numpy as np
from abc import ABC, abstractmethod
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

if __name__ == '__main__':
    scene = Scene()
    axes = Axes(length=10.0)
    scene.add(axes)

    print(scene.objects)
    print(axes.vertices)