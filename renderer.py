import moderngl
from render_space import *
from scene import *
from render_types import ProgramID, Mode
'''it owns:
Context (borrowed, not created)
Program
Buffer (VBO)
VertexArray (VAO)

It does not own:
Scene logic
Simulation state
Camera movement logic'''

'''class Mode(Enum): # Moderngl draw modes
    POINTS = auto()
    LINES = auto()
    LINE_STRIP = auto()
    TRIANGLES = auto()
    LINE_LOOP = auto()

class ProgramID(Enum): # supported shader programs
    BASIC_3D = auto()'''

class ProgramManager: # holds and stores programs that draw points, lines, etc.

    def __init__(self, ctx: moderngl.Context):
        self.programs: dict[ProgramID, moderngl.Program] = {}
        self.ctx = ctx
        self.programs[ProgramID.BASIC_3D] = self.basic_3d_src()
    
    def basic_3d_src(self):
        VERTEX_SOURCE = """
        #version 330

        layout (location = 0) in vec3 in_position;
        layout (location = 1) in vec3 in_color;

        uniform mat4 u_view;
        uniform mat4 u_proj;

        out vec3 v_color;

        void main() {
        gl_Position = u_proj * u_view * vec4(in_position, 1.0);
        gl_PointSize = 2.0;
        v_color = in_color;
        }
        """

        FRAGMENT_SOURCE = """
        #version 330

        in vec3 v_color;
        out vec4 fragColor;

        void main() {
        fragColor = vec4(v_color, 1.0);
        }
        """
        return VERTEX_SOURCE, FRAGMENT_SOURCE
    
    def build_program(self, program_id) -> moderngl.Program: # think of as the material 
        if program_id == ProgramID.BASIC_3D:
            VERTEX_SOURCE, FRAGMENT_SOURCE = self.basic_3d_src()
        else:
            print('no valid shader source code available') 
            return 
        program = self.ctx.program(
            vertex_shader=VERTEX_SOURCE, 
            fragment_shader=FRAGMENT_SOURCE) 
        
        self.programs[program_id] = program
        return program
    
class RenderObject:
    def __init__(self,
        program_id: ProgramID,
        vao: moderngl.VertexArray,
        vbo: moderngl.Buffer,
        num_vertexes: int,
        mode: Mode,
        dynamic: bool = False,
        ) -> None:
        
        self.program_id = program_id
        self.num_vertexes = num_vertexes
        self.mode = mode
        self.dynamic = dynamic
        self.vao = vao
        self.vbo = vbo

    def release(self):
        self.vao.release()
        self.vbo.release()

class Renderer:
    
    def __init__(self, scene: Scene,render_space: RenderSpace):

        self.ctx = render_space.ctx
        self.program_manager = ProgramManager(self.ctx)
        self.scene = scene
        self.objects_to_render = list[RenderObject]
    
    def create_render_object(self, obj: SceneObject) -> RenderObject:
        program = self.program_manager.build_program(obj.program_id)
        vbo = self.ctx.buffer(obj.vertices.tobytes())
        vao = self.ctx.vertex_array(
            program,
            [(vbo, "3f 3f", "in_position", "in_color")]
        )
        render_object = RenderObject(
            program_id=obj.program_id,
            vao=vao,
            vbo=vbo,
            num_vertexes=len(obj.vertices)//6,
            mode=obj.mode,
            dynamic=obj.dynamic
        )
        return render_object
    
    def render(self, cam: Camera, width: int, height: int):
        """
        Renders all RenderObjects in the scene.
        Each object uses its own VAO and program.
        """
        for ro in self.objects_to_render:
            # Get the program associated with this object
            program = self.program_manager.programs.get(ro.program_id)
            if not program:
                continue  # skip objects with no valid program

            # Write camera uniforms
            program["u_view"].write(cam.get_view_matrix())
            program["u_proj"].write(cam.get_projection_matrix(width, height))

            # Choose the correct draw mode
            if ro.mode == Mode.POINTS:
                m = moderngl.POINTS
            elif ro.mode == Mode.LINES:
                m = moderngl.LINES
            elif ro.mode == Mode.LINE_STRIP:
                m = moderngl.LINE_STRIP
            elif ro.mode == Mode.LINE_LOOP:
                m = moderngl.LINE_LOOP
            elif ro.mode == Mode.TRIANGLES:
                m = moderngl.TRIANGLES
            else:
                m = moderngl.POINTS  # fallback

            # Render the object
            ro.vao.render(mode=m, vertices=ro.num_vertexes)