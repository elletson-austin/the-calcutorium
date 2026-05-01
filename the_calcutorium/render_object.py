from dataclasses import dataclass, field
import logging

import moderngl

from .scene import ProgramID, RenderMode

logger = logging.getLogger(__name__)


@dataclass
class RenderObject:
    program_id: ProgramID
    vao: moderngl.VertexArray
    ssbo: moderngl.Buffer
    render_mode: RenderMode
    num_vertexes: int
    compute_shader: moderngl.ComputeShader | None = None
    compute_uniforms: dict = field(default_factory=dict)
    storage_buffers: list | None = None  # list of (buffer, binding_index) pairs for SSBOs
    compute_groups: tuple | None = None
    compute_local_size_x: int = 256

    @staticmethod
    def _release(resource) -> None:
        if resource is None:
            return
        try:
            resource.release()
        except Exception as e:
            logger.warning("Failed to release GL resource %r: %s", resource, e)

    def release(self) -> None:
        self._release(self.vao)
        self._release(self.ssbo)

        # Release any extra storage buffers, avoiding double-release of primary ssbo
        if self.storage_buffers:
            for buf, _ in self.storage_buffers:
                if buf is not self.ssbo:
                    self._release(buf)
