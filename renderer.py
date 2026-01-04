import moderngl
from render_space import *
from scene import *

'''it owns:
Context (borrowed, not created)
Program
Buffer (VBO)
VertexArray (VAO)

It does not own:
Scene logic
Simulation state
Camera movement logic'''
class Renderer:
    
    def __init__(self, render_space: RenderSpace):

        self.ctx = render_space.ctx

        self.program = None
        self.vbo = None
        self.vao = None

        self.VERTEX_SOURCE = """
        #version 330

        in vec3 in_position;
        in vec3 in_color;

        uniform mat4 u_view;
        uniform mat4 u_proj;

        out vec3 v_color;

        void main() {
        gl_Position = u_proj * u_view * vec4(in_position, 1.0);
        v_color = in_color;
        }
        """

        self.FRAGMENT_SOURCE = """
        #version 330

        in vec3 v_color;
        out vec4 fragColor;

        void main() {
        fragColor = vec4(v_color, 1.0);
        }
        """
        
        self.build_program()
        self.build_axes()

    def build_program(self):
        self.program = self.ctx.program(
            vertex_shader=self.VERTEX_SOURCE,
            fragment_shader=self.FRAGMENT_SOURCE,
        )

    def build_axes(self, length: float = 10.0): # TODO move to scene object eventually
        vertices = np.array([
            # X axis (red)
            -length, 0, 0,  1, 0, 0,
            length, 0, 0,  1, 0, 0,

            # Y axis (green)
            0, -length, 0,  0, 1, 0,
            0,  length, 0,  0, 1, 0,

            # Z axis (blue)
            0, 0, -length,  0, 0, 1,
            0, 0,  length,  0, 0, 1,
        ], dtype=np.float32)

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, "3f 3f", "in_position", "in_color")]
        )
    def render(self, cam: Camera, width: int, height: int):
        self.program["u_view"].write(cam.get_view_matrix())
        self.program["u_proj"].write(cam.get_projection_matrix(width, height))

        self.vao.render(mode=moderngl.LINES)