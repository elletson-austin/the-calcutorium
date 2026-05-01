from enum import Enum, auto


class Projection(Enum):
    Perspective = auto()
    Orthographic = auto()


class SnapMode(Enum):
    NONE = auto()
    XY = auto()
    XZ = auto()
    YZ = auto()


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

# (horizontal, vertical, out-of-plane) axis names per 2D plane
PLANE_AXES: dict[str, tuple[str, str, str]] = {
    "xy": ("x", "y", "z"),
    "xz": ("x", "z", "y"),
    "yz": ("z", "y", "x"),
}

SNAP_MODE_TO_PLANE: dict[SnapMode, str] = {
    SnapMode.XY: "xy",
    SnapMode.XZ: "xz",
    SnapMode.YZ: "yz",
}


def plane_for_snap_mode(snap: SnapMode) -> str | None:
    return SNAP_MODE_TO_PLANE.get(snap)


def axes_for_plane(plane: str) -> tuple[str, str, str] | None:
    """Return (horizontal, vertical, out-of-plane) axis names; None for unknown plane."""
    return PLANE_AXES.get(plane)
