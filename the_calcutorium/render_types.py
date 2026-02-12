from enum import Enum, auto

class Projection(Enum):
    Perspective = auto()
    Orthographic = auto()

class SnapMode(Enum):
    NONE = auto()
    XY = auto()
    XZ = auto()
    YZ = auto()

