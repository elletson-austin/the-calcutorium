import moderngl
import numpy as np

from camera import Camera
from scene import SceneObject, LorenzAttractor
from render_types import ProgramID, Mode


class ProgramManager: # holds and stores programs that draw points, lines, etc.

    def __init__(self, ctx: moderngl.Context):
        self.programs: dict[ProgramID, moderngl.Program] = {}
        self.compute_shaders: dict[ProgramID, moderngl.ComputeShader] = {}
        self.ctx = ctx

    
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
    
    def lorenz_attractor_src(self):
        VERTEX_SHADER = """
        #version 330

        in vec4 in_position;

        uniform mat4 u_view;
        uniform mat4 u_proj;

        out vec3 frag_pos;

        void main() {
            frag_pos = in_position.xyz;
            gl_Position = u_proj * u_view * vec4(in_position.xyz, 1.0);
            gl_PointSize = 0.5; 
        }
        """
        FRAGMENT_SHADER = """
        #version 330

        in vec3 frag_pos;
        out vec4 fragColor;

        void main() {
            fragColor = vec4(1.0, 0.2, 0.2, 1.0);
        }
        """
        return VERTEX_SHADER, FRAGMENT_SHADER
    
    def lorenz_attractor_compute_src(self):
        COMPUTE_SHADER = """
        #version 430

        layout(local_size_x = 256) in;

        layout(std430, binding = 0) buffer PointsBuffer {
            vec4 points[];
        };

        uniform float dt;
        uniform float sigma;
        uniform float rho;
        uniform float beta;
        uniform int steps;

        void main() {
            uint idx = gl_GlobalInvocationID.x;
            if (idx >= points.length()) return;
            
            vec3 p = points[idx].xyz;
            
            for (int i = 0; i < steps; i++) {
                float dx = sigma * (p.y - p.x);
                float dy = p.x * (rho - p.z) - p.y;
                float dz = p.x * p.y - beta * p.z;
                
                p.x += dx * dt;
                p.y += dy * dt;
                p.z += dz * dt;
            }
            
            points[idx].xyz = p;
        }
        """
        return COMPUTE_SHADER
    
    def build_compute_shader(self, program_id) -> moderngl.ComputeShader:
        if program_id in self.compute_shaders:
            return self.compute_shaders[program_id]

        if program_id == ProgramID.LORENZ_ATTRACTOR:
            COMPUTE_SOURCE = self.lorenz_attractor_compute_src()
        else:
            print('no valid compute shader source code available') 
            return 
        
        compute_shader = self.ctx.compute_shader(COMPUTE_SOURCE)
        
        self.compute_shaders[program_id] = compute_shader
        return compute_shader


    def build_program(self, program_id) -> moderngl.Program: # think of as the material 
        if program_id in self.programs:
            return self.programs[program_id]

        if program_id == ProgramID.BASIC_3D:
            VERTEX_SOURCE, FRAGMENT_SOURCE = self.basic_3d_src()
        elif program_id == ProgramID.LORENZ_ATTRACTOR:
            VERTEX_SOURCE, FRAGMENT_SOURCE = self.lorenz_attractor_src()
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
        mode: Mode,
        num_vertexes: int,
        compute_shader: moderngl.ComputeShader = None,
        ) -> None:
        
        self.program_id = program_id
        self.vao = vao
        self.vbo = vbo
        self.mode = mode
        self.num_vertexes = num_vertexes
        self.compute_shader = compute_shader


    def release(self):
        self.vao.release()
        self.vbo.release()

class Renderer:
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.program_manager = ProgramManager(self.ctx)
    
    def create_render_object(self, obj: SceneObject) -> RenderObject:
        program = self.program_manager.build_program(obj.ProgramID)
        #mode
        if isinstance(obj, LorenzAttractor):
            vbo = self.ctx.buffer(obj.vertices.tobytes())
            vao = self.ctx.vertex_array(
                program,
                [(vbo, "4f", "in_position")]
            )
            compute_shader = self.program_manager.build_compute_shader(obj.ProgramID)
            compute_shader['sigma'] = obj.sigma
            compute_shader['rho'] = obj.rho
            compute_shader['beta'] = obj.beta
            compute_shader['dt'] = obj.dt
            compute_shader['steps'] = obj.steps

            return RenderObject(
                program_id=obj.ProgramID,
                vao=vao,
                vbo=vbo,
                mode=obj.Mode,
                num_vertexes=obj.num_points,
                compute_shader=compute_shader,
            )
        else: # For other objects
            vbo = self.ctx.buffer(obj.vertices.tobytes())
            vao = self.ctx.vertex_array(
                program,
                [(vbo, "3f 3f", "in_position", "in_color")]
            )
            return RenderObject(
                program_id=obj.ProgramID,
                vao=vao,
                vbo=vbo,
                mode=obj.Mode,
                num_vertexes=len(obj.vertices) // 6,
            )

    def render(self, render_objects: list, cam: Camera, width: int, height: int) -> list[RenderObject]:

        for ro in render_objects:
            if ro.compute_shader:
                ro.vbo.bind_to_storage_buffer(0)
                group_size = (ro.num_vertexes + 255) // 256
                ro.compute_shader.run(group_x=group_size)

            program = ro.vao.program
            program["u_view"].write(cam.get_view_matrix())
            program["u_proj"].write(cam.get_projection_matrix(width, height))

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
                m = moderngl.POINTS

            ro.vao.render(mode=m)