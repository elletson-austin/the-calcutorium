from enum import Enum, auto

class Mode(Enum):
    POINTS = auto()
    LINES = auto()
    LINE_STRIP = auto()
    TRIANGLES = auto()
    LINE_LOOP = auto()

class ProgramID(Enum):
    BASIC_3D = auto()