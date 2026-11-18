"""Cameras turn 3D world coordinates into screen coordinates (view + projection matrices).

Camera3D orbits around a center point. Camera2D looks straight at one plane (xy, xz, yz).
Matrices are returned column-major and flattened, which is what OpenGL expects.
"""

from enum import Enum, auto

import numpy as np

from ..planes import AXIS_INDEX, PLANE_AXES


class Projection(Enum):
    Perspective = auto()
    Orthographic = auto()


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def _look_at(eye: np.ndarray, target: np.ndarray, world_up) -> np.ndarray:
    forward = _normalize(target - eye)
    right = _normalize(np.cross(forward, world_up))
    up = np.cross(right, forward)

    view = np.eye(4, dtype=np.float32)
    view[0, :3], view[1, :3], view[2, :3] = right, up, -forward
    view[0, 3] = -np.dot(right, eye)
    view[1, 3] = -np.dot(up, eye)
    view[2, 3] = np.dot(forward, eye)
    return view.T.flatten()


def _orthographic(left: float, right: float, bottom: float, top: float) -> np.ndarray:
    near, far = -1000.0, 1000.0
    proj = np.eye(4, dtype=np.float32)
    proj[0, 0] = 2 / (right - left)
    proj[1, 1] = 2 / (top - bottom)
    proj[2, 2] = -2 / (far - near)
    proj[0, 3] = -(right + left) / (right - left)
    proj[1, 3] = -(top + bottom) / (top - bottom)
    proj[2, 3] = -(far + near) / (far - near)
    return proj.T.flatten()


def _perspective(fov_degrees: float, aspect: float) -> np.ndarray:
    near, far = 1.0, 500.0
    focal = 1.0 / np.tan(np.radians(fov_degrees) / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = focal / aspect
    proj[1, 1] = focal
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2.0 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj.T.flatten()


_Z_UP = np.array([0, 0, 1], dtype=np.float32)
ORBIT_DEGREES_PER_PIXEL = 0.4


class Camera3D:
    """Orbits around `center` with z pointing up, like a calculus textbook.

    yaw is the angle around the z-axis measured from +x toward +y; pitch is the
    angle above the xy-plane. The default looks from the first octant, so x points
    toward you, y to the right, and z up.
    """

    def __init__(self, center=None, distance: float = 40.0, fov: float = 60.0):
        self.center = np.array(center if center is not None else (0, 0, 0), dtype=np.float32)
        self.pitch, self.yaw = 25.0, 30.0  # degrees
        self.distance = distance
        self.fov = fov
        self.projection = Projection.Perspective

    def position(self) -> np.ndarray:
        pitch, yaw = np.radians(self.pitch), np.radians(self.yaw)
        offset = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        return (self.center + self.distance * offset).astype(np.float32)

    def view_matrix(self) -> np.ndarray:
        return _look_at(self.position(), self.center, _Z_UP)

    def projection_matrix(self, width: int, height: int) -> np.ndarray:
        aspect = width / height if height > 0 else 1.0
        if self.projection == Projection.Perspective:
            return _perspective(self.fov, aspect)
        half_h = self.distance / 2.0
        return _orthographic(-half_h * aspect, half_h * aspect, -half_h, half_h)

    def toggle_projection(self) -> None:
        self.projection = (
            Projection.Orthographic if self.projection == Projection.Perspective else Projection.Perspective
        )

    def orbit(self, dx_pixels: float, dy_pixels: float) -> None:
        """Mouse drag rotates the scene: drag right spins it right, drag down tilts it toward you."""
        self.yaw = (self.yaw - dx_pixels * ORBIT_DEGREES_PER_PIXEL) % 360.0
        self.pitch = float(np.clip(self.pitch + dy_pixels * ORBIT_DEGREES_PER_PIXEL, -89.0, 89.0))

    def zoom(self, factor: float) -> None:
        self.distance = float(np.clip(self.distance * factor, 1.0, 500.0))

    def move(self, forward_amt: float, right_amt: float) -> None:
        """Slide the center across the xy-plane, relative to where the camera faces."""
        forward = self.center - self.position()
        forward[2] = 0
        forward = _normalize(forward)
        right = _normalize(np.cross(forward, _Z_UP))
        self.center += forward * forward_amt + right * right_amt


class Camera2D:
    """Looks straight at a plane with the horizontal axis pointing right and the vertical up.

    For plane (h, v) the camera sits on the h × v side, so the view is right-handed:
    xy is seen from +z, xz from −y, yz from −x.
    """

    def __init__(self, plane: str = "xy", center=None, distance: float = 50.0):
        self.plane = plane
        h, v, _ = PLANE_AXES[plane]
        self.h_index, self.v_index = AXIS_INDEX[h], AXIS_INDEX[v]
        self.center = np.array(center if center is not None else (0, 0, 0), dtype=np.float32)
        self.distance = distance  # visible height in world units

    def _unit(self, index: int) -> np.ndarray:
        u = np.zeros(3, dtype=np.float32)
        u[index] = 1.0
        return u

    def position(self) -> np.ndarray:
        toward_camera = np.cross(self._unit(self.h_index), self._unit(self.v_index))
        return self.center + self.distance * toward_camera

    def view_matrix(self) -> np.ndarray:
        return _look_at(self.position(), self.center, self._unit(self.v_index))

    def visible_ranges(self, width: int, height: int):
        """The (h_range, v_range) in world units the camera shows at this window size."""
        aspect = width / height if height > 0 else 1.0
        half_h, half_w = self.distance / 2, self.distance * aspect / 2
        h, v = float(self.center[self.h_index]), float(self.center[self.v_index])
        return (h - half_w, h + half_w), (v - half_h, v + half_h)

    def fit(self, h_range, v_range) -> None:
        """Center on the given world ranges (used when the user fixes both ranges)."""
        self.center[self.h_index] = (h_range[0] + h_range[1]) / 2
        self.center[self.v_index] = (v_range[0] + v_range[1]) / 2
        self.distance = v_range[1] - v_range[0]

    def projection_matrix(self, width: int, height: int, ranges=None) -> np.ndarray:
        """Orthographic. With explicit world `ranges`, shows exactly those (may stretch)."""
        if ranges is None:
            ranges = self.visible_ranges(width, height)
        (left, right), (bottom, top) = ranges
        # The view matrix puts self.center at the origin, so shift world ranges by it.
        ch, cv = float(self.center[self.h_index]), float(self.center[self.v_index])
        return _orthographic(left - ch, right - ch, bottom - cv, top - cv)

    def pan(self, dx_pixels: float, dy_pixels: float, width: int, height: int) -> None:
        """Drag the plane with the mouse (Qt's y grows downward)."""
        if width <= 0 or height <= 0:
            return
        units_per_pixel = self.distance / height
        self.center[self.h_index] -= dx_pixels * units_per_pixel
        self.center[self.v_index] += dy_pixels * units_per_pixel

    def zoom(self, factor: float) -> None:
        self.distance = float(np.clip(self.distance * factor, 1e-3, 1e5))
