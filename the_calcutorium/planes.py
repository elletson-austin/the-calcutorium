"""Axis and plane names shared by every layer. Plain constants, no dependencies."""

AXES = ("x", "y", "z")
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

PLANES = ("xy", "xz", "yz")

# (horizontal axis, vertical axis, out-of-plane axis) for each 2D view
PLANE_AXES: dict[str, tuple[str, str, str]] = {
    "xy": ("x", "y", "z"),
    "xz": ("x", "z", "y"),
    "yz": ("z", "y", "x"),
}
