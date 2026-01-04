import numpy as np
from abc import ABC, abstractmethod
'''The scene owns references and intent
The renderer owns GPU buffers
The simulation writes directly to those buffers'''
class SceneObject:

    def __init__(self, visibility: bool = False):
        self.vbo = None
        self.vao = None

class Scene:
    def __init__(self):
        self.objects: list[SceneObject] = []

    def add(self, obj: SceneObject):
        if obj.name in self.objects:
            raise ValueError(f"SceneObject '{obj.name}' already exists")
        self.objects[obj.name] = obj

    def remove(self, obj: SceneObject):
        if obj.name not in self.objects:
            raise ValueError(f"SceneObject '{obj.name}' doesn't exist")
        del self.objects[obj.name]



class Curve(SceneObject):
    def __init__(self):
        pass

class Axes(SceneObject):
    def __init__(
        self,
        axis_length: float = 10.0,
        tick_interval: float = 1.0,
        visible: bool = True
    ):
        super().__init__(visibility=visible)

        self.axis_length = axis_length
        self.tick_interval = tick_interval

        self.show_ticks = True
        self.show_labels = False   # future
        self.color_xyz = (
            (1.0, 0.0, 0.0),  # X
            (0.0, 1.0, 0.0),  # Y
            (0.0, 0.0, 1.0),  # Z
        )

class ParticleSim(SceneObject):
    pass