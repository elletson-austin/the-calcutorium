import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field

class Projection(Enum):
    Perspective = auto()
    Orthographic = auto()

class SnapMode(Enum):
    NONE = auto()
    XY = auto()
    XZ = auto()
    YZ = auto()


@dataclass
class InputState: # Tracks the state of the input
    mouse_pos: np.ndarray = field(
        default_factory=lambda: np.array([0, 0], dtype=np.float32)
    )
    mouse_delta: np.ndarray = field(
        default_factory=lambda: np.array([0, 0], dtype=np.float32)
    )
    keys_held: set = field(default_factory=set)
    left_mouse_pressed: bool = False
    right_mouse_pressed: bool = False
    scroll_delta: float = 0.0


class Camera: # TODO It would make sense to have two different camera objects between 2d and 3d visualization.
    
    def __init__(self, 
                 position_center: np.ndarray = None,
                 rotation: np.ndarray = None,
                 fov: float = 60.0, distance: float = 50.0):
        
        if position_center is not None:
            self.position_center = position_center
        else:
            self.position_center = np.array([0.0, 20.0, 25.0], dtype=np.float32)

        if rotation is not None:
            self.rotation = rotation
        else:
            self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.fov = fov
        self.distance = distance
        self.projection = Projection.Orthographic
        self.snap_mode = SnapMode.XY

    # Calculate the camera position based on the orientation, distance and center
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
        
        forward = target - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        view = np.eye(4, dtype=np.float32)
        
        # Rotation part
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        
        # Translation part - ALL NEGATIVE
        view[0, 3] = -np.dot(right, cam_pos)
        view[1, 3] = -np.dot(up, cam_pos)
        view[2, 3] = np.dot(forward, cam_pos)  
        
        # Bottom row should be [0, 0, 0, 1] - already set by np.eye()
        
        return view.T.flatten()
    

    def _perspective_matrix(self, width, height) -> np.ndarray: 
        if height == 0:
            aspect_ratio = 1.0
        else:
            aspect_ratio = width / height
        fov_rad = np.radians(self.fov)  # Field of view in radians
        near = 1.0     # Near clipping plane (minimum render distance)
        far = 500.0    # Far clipping plane (maximum render distance)
        focal_len = 1.0 / np.tan(fov_rad / 2.0)  # Calculate focal length (controls zoom)

        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = focal_len / aspect_ratio
        proj[1, 1] = focal_len
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2.0 * far * near) / (near - far)
        proj[3, 2] = -1.0
        
        return proj.T.flatten()
        

    def _orthographic_matrix(self, width, height):
        if height == 0:
            aspect = 1.0
        else:
            aspect = width / height
        
        view_height = self.distance
        top = view_height / 2.0
        bottom = -view_height / 2.0
        right = top * aspect
        left = -top * aspect

        near   = -1000.0
        far    =  1000.0

        proj = np.eye(4, dtype=np.float32)
        proj[0, 0] = 2 / (right - left)
        proj[1, 1] = 2 / (top - bottom)
        proj[2, 2] = -2 / (far - near)
        proj[0, 3] = -(right + left) / (right - left)
        proj[1, 3] = -(top + bottom) / (top - bottom)
        proj[2, 3] = -(far + near) / (far - near)

        return proj.T.flatten()
    

    def get_projection_matrix(self, width, height):
        
        if self.projection == Projection.Perspective:
            return self._perspective_matrix(width, height)
        else:
            return self._orthographic_matrix(width, height)
