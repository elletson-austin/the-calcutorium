from dataclasses import dataclass
import moderngl

from .scene import ProgramID, RenderMode

@dataclass
class RenderObject:
    program_id: ProgramID
    vao: moderngl.VertexArray
    ssbo: moderngl.Buffer
    Rendermode: RenderMode
    num_vertexes: int
    compute_shader: moderngl.ComputeShader | None = None
    compute_uniforms: dict | None = None
    storage_buffers: list | None = None  # list of (buffer, binding_index) pairs for SSBOs
    compute_groups: tuple | None = None
    compute_local_size_x: int = 256


    @staticmethod
    def _release_buffer(buf: moderngl.Buffer | None) -> None:
        if buf is not None:
            try:
                buf.release()
            except Exception:
                pass

    def release(self) -> None:
        RenderObject._release_buffer(self.vao)
        RenderObject._release_buffer(self.ssbo)

        # Release all storage buffers, avoiding double-release of primary ssbo
        if self.storage_buffers:
            for buf, _ in self.storage_buffers:
                if buf is not self.ssbo:
                    self._release_buffer(buf)

