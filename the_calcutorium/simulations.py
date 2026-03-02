import numpy as np

from .scene import SceneObject, RenderMode, ProgramID


class LorenzAttractor(SceneObject):
    def __init__(self, 
                num_points: int = 100_000, 
                sigma: float = 10.0, 
                rho: float = 28.0, 
                beta: float = 8.0 / 3.0, 
                dt: float = 0.001, 
                steps: int = 5):

        super().__init__(RenderMode=RenderMode.POINTS, ProgramID=ProgramID.LORENZ_ATTRACTOR, dynamic=True)
        self.uniforms: dict = {
            'sigma': sigma,
            'rho': rho,
            'beta': beta,
            'dt': dt,
            'steps': steps}
        self.num_points = num_points
        self.vertices = self.create_initial_points(num_points=num_points)

    def to_dict(self):
        d = super().to_dict()
        d['uniforms'] = self.uniforms
        return d

    def create_initial_points(self, num_points: int) -> np.ndarray:
        initial_points = np.random.randn(num_points, 4).astype(np.float32)
        initial_points[:, :3] *= 2.0
        initial_points[:, :3] += [1.0, 1.0, 1.0]
        initial_points[:, 3] = 1.0
        return initial_points

    def update(self, **kwargs):
        pass # Handled by the compute shader in the renderer
        

class NBody(SceneObject):
    def __init__(self, 
                num_bodies: int = 4000, 
                dt: float = 0.01, 
                G: float = 1.0, 
                softening: float = 1.0, 
                steps: int = 5):
        super().__init__(RenderMode=RenderMode.POINTS, ProgramID=ProgramID.NBODY, dynamic=True)
        self.num_bodies = num_bodies
        self.uniforms: dict = {
            'dt': dt,
            'G': G,
            'softening': softening,
            'num_bodies': num_bodies,
            'steps': steps}
        self.positions = self._create_initial_positions(num_bodies)
        self.velocities = self._create_initial_velocities(num_bodies)
        self.masses = self._create_initial_masses(num_bodies)
        self.vertices = self.positions.flatten()

    def to_dict(self):
        d = super().to_dict()
        d['uniforms'] = self.uniforms
        return d

    def _create_initial_positions(self, n: int) -> np.ndarray:
        p = np.random.randn(n, 4).astype(np.float32)
        p[:, :3] *= 5.0
        p[:, 3] = 1.0
        return p

    def _create_initial_velocities(self, n: int) -> np.ndarray:
        v = (np.random.randn(n, 4) * 0.1).astype(np.float32)
        v[:, 3] = 0.0
        return v

    def _create_initial_masses(self, n: int) -> np.ndarray:
        return (np.abs(np.random.randn(n)) * 1.0).astype(np.float32)

    def update(self, **kwargs):
        pass # Handled by the compute shader in the renderer
        
