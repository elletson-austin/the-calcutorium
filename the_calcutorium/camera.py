import numpy as np

from .render_types import Projection, SnapMode


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n > 1e-6:
        return v / n
    return v


def _make_view_matrix(
    cam_pos: np.ndarray, target: np.ndarray, world_up: np.ndarray
) -> np.ndarray:
    """Compute a view matrix (column-major, flattened) from camera position, target and up vector."""
    forward = target - cam_pos
    forward = _normalize(forward)

    right = np.cross(forward, world_up)
    right = _normalize(right)

    up = np.cross(right, forward)

    view = np.eye(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = up
    view[2, :3] = -forward
    view[0, 3] = -np.dot(right, cam_pos)
    view[1, 3] = -np.dot(up, cam_pos)
    view[2, 3] = np.dot(forward, cam_pos)

    return view.T.flatten()


class Camera3D:
    def __init__(
        self,
        position_center: np.ndarray | None = None,
        rotation: np.ndarray | None = None,
        distance: float = 50.0,
        fov: float = 60.0,
    ):

        self.position_center = (
            np.array([0.0, 0.0, 0.0], dtype=np.float32)
            if position_center is None
            else np.array(position_center, dtype=np.float32)
        )
        self.rotation = (
            np.array([0.0, 45.0, 0.0], dtype=np.float32)
            if rotation is None
            else np.array(rotation, dtype=np.float32)
        )
        self.distance = distance
        self.fov = fov
        self.projection = Projection.Perspective

    def get_position(self) -> np.ndarray:
        pitch = np.radians(self.rotation[0])
        yaw = np.radians(self.rotation[1])

        x = self.distance * np.cos(pitch) * np.sin(yaw)
        y = self.distance * np.sin(pitch)
        z = self.distance * np.cos(pitch) * np.cos(yaw)

        return self.position_center + np.array([x, y, z], dtype=np.float32)

    def get_view_matrix(self) -> np.ndarray:
        cam_pos = self.get_position()
        target = self.position_center
        world_up = np.array([0, 1, 0], dtype=np.float32)
        return _make_view_matrix(cam_pos, target, world_up)

    def get_projection_matrix(self, width, height, h_range=None, v_range=None):
        if self.projection == Projection.Perspective:
            aspect_ratio = width / height if height > 0 else 1.0
            fov_rad = np.radians(self.fov)
            near, far = 1.0, 500.0
            focal_len = 1.0 / np.tan(fov_rad / 2.0)

            proj = np.zeros((4, 4), dtype=np.float32)
            proj[0, 0] = focal_len / aspect_ratio
            proj[1, 1] = focal_len
            proj[2, 2] = (far + near) / (near - far)
            proj[2, 3] = (2.0 * far * near) / (near - far)
            proj[3, 2] = -1.0
            return proj.T.flatten()

        # Orthographic
        aspect = width / height if height > 0 else 1.0
        view_height = self.distance
        top, bottom = view_height / 2.0, -view_height / 2.0
        right, left = top * aspect, -top * aspect
        near, far = -1000.0, 1000.0

        proj = np.eye(4, dtype=np.float32)
        proj[0, 0] = 2 / (right - left)
        proj[1, 1] = 2 / (top - bottom)
        proj[2, 2] = -2 / (far - near)
        proj[0, 3] = -(right + left) / (right - left)
        proj[1, 3] = -(top + bottom) / (top - bottom)
        proj[2, 3] = -(far + near) / (far - near)
        return proj.T.flatten()

    def orbit(self, dyaw: float, dpitch: float) -> None:
        self.rotation[0] = np.clip(self.rotation[0] + dpitch, -89.0, 89.0)
        self.rotation[1] = (self.rotation[1] + dyaw) % 360.0

    def zoom(self, factor: float) -> None:
        self.distance = float(np.clip(self.distance * factor, 1.0, 500.0))

    def move(self, forward_amt: float, right_amt: float) -> None:
        """Translate the camera target on the ground plane by world-space amounts."""
        cam_pos = self.get_position()
        forward = self.position_center - cam_pos
        forward[1] = 0
        forward = _normalize(forward)

        right = np.cross(forward, np.array([0, 1, 0], dtype=np.float32))
        right = _normalize(right)

        self.position_center += forward * forward_amt + right * right_amt


class Camera2D:
    def __init__(
        self,
        position_center: np.ndarray | None = None,
        distance: float = 50.0,
        snap_mode: SnapMode = SnapMode.XY,
    ):
        self.position_center = (
            np.array([0.0, 0.0, 0.0], dtype=np.float32)
            if position_center is None
            else np.array(position_center, dtype=np.float32)
        )
        self.distance = distance
        self.snap_mode = snap_mode
        self.projection = Projection.Orthographic

    def get_position(self) -> np.ndarray:
        pos = self.position_center.copy()
        if self.snap_mode == SnapMode.XY:
            pos[2] += self.distance
        elif self.snap_mode == SnapMode.XZ:
            pos[1] += self.distance
        elif self.snap_mode == SnapMode.YZ:
            pos[0] += self.distance
        return pos

    def get_view_matrix(self) -> np.ndarray:
        cam_pos = self.get_position()
        target = self.position_center
        world_up = (
            np.array([0, 0, 1], dtype=np.float32)
            if self.snap_mode == SnapMode.XZ
            else np.array([0, 1, 0], dtype=np.float32)
        )
        return _make_view_matrix(cam_pos, target, world_up)

    def get_projection_matrix(self, width, height, h_range=None, v_range=None):
        if h_range is not None and v_range is not None:
            left, right = h_range
            bottom, top = v_range
        else:
            aspect = width / height if height > 0 else 1.0
            view_height = self.distance
            top, bottom = view_height / 2.0, -view_height / 2.0
            right, left = top * aspect, -top * aspect

        near, far = -1000.0, 1000.0
        proj = np.eye(4, dtype=np.float32)
        proj[0, 0] = 2 / (right - left)
        proj[1, 1] = 2 / (top - bottom)
        proj[2, 2] = -2 / (far - near)
        proj[0, 3] = -(right + left) / (right - left)
        proj[1, 3] = -(top + bottom) / (top - bottom)
        proj[2, 3] = -(far + near) / (far - near)
        return proj.T.flatten()

    def pan(self, dx_pixels: float, dy_pixels: float, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            return
        aspect_ratio = width / height
        pan_speed_h = self.distance * aspect_ratio / width
        pan_speed_v = self.distance / height
        if self.snap_mode == SnapMode.XY:
            self.position_center[0] -= dx_pixels * pan_speed_h
            self.position_center[1] += dy_pixels * pan_speed_v
        elif self.snap_mode == SnapMode.XZ:
            self.position_center[0] -= dx_pixels * pan_speed_h
            self.position_center[2] -= dy_pixels * pan_speed_v
        elif self.snap_mode == SnapMode.YZ:
            self.position_center[2] -= dx_pixels * pan_speed_h
            self.position_center[1] += dy_pixels * pan_speed_v

    def zoom(self, factor: float) -> None:
        self.distance = float(np.clip(self.distance * factor, 1.0, 500.0))
