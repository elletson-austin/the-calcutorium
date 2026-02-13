import moderngl
from typing import Dict

from .scene import ProgramID

class ProgramManager: # holds and stores programs that draw points, lines, etc.

    def __init__(self, ctx: moderngl.Context):
        self.programs: Dict[ProgramID, moderngl.Program] = {}
        self.compute_shaders: Dict[ProgramID, moderngl.ComputeShader] = {}
        self.ctx = ctx

    def _read_shader_source(self, filename: str) -> str:
        # Placeholder for reading shader source from external files
        # In the future, this will read from .glsl files
        shader_dir = "the_calcutorium/shaders" # Assuming a shaders directory
        file_path = f"{shader_dir}/{filename}"
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: Shader file not found at {file_path}")
            return ""

    def basic_3d_src(self):
        # Temporarily keep the strings here, will move to file later
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

    def grid_src(self):
        # Temporarily keep the strings here, will move to file later
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
        uniform float u_alpha_multiplier;

        void main() {
            fragColor = vec4(v_color, 1.0 * u_alpha_multiplier);
        }
        """
        return VERTEX_SOURCE, FRAGMENT_SOURCE

    def surface_src(self):
        # Temporarily keep the strings here, will move to file later
        VERTEX_SHADER = """
            #version 330

            layout (location = 0) in vec3 in_position;
            layout (location = 1) in vec3 in_normal;
            layout (location = 2) in vec3 in_color;

            uniform mat4 u_view;
            uniform mat4 u_proj;
            uniform mat4 u_model;

            out vec3 v_normal;
            out vec3 v_pos;
            out vec3 v_color;

            void main() {
                gl_Position = u_proj * u_view * u_model * vec4(in_position, 1.0);
                v_pos = (u_model * vec4(in_position, 1.0)).xyz;
                v_normal = mat3(transpose(inverse(u_model))) * in_normal;
                v_color = in_color;
            }
        """
        FRAGMENT_SHADER = """
            #version 330

            in vec3 v_normal;
            in vec3 v_pos;
            in vec3 v_color;

            out vec4 fragColor;

            uniform vec3 u_light_pos;
            uniform vec3 u_view_pos;

            void main() {
                vec3 norm = normalize(v_normal);
                vec3 light_dir = normalize(u_light_pos - u_view_pos); // Changed from v_pos
                
                // Ambient
                float ambient_strength = 0.2;
                vec3 ambient = ambient_strength * vec3(1.0, 1.0, 1.0);
                
                // Diffuse
                float diff = max(dot(norm, light_dir), 0.0);
                vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);
                
                // Specular
                float specular_strength = 0.4;
                vec3 view_dir = normalize(u_view_pos - v_pos);
                vec3 reflect_dir = reflect(-light_dir, norm);
                float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
                vec3 specular = specular_strength * spec * vec3(1.0, 1.0, 1.0);
                
                vec3 result = (ambient + diffuse + specular) * v_color;
                fragColor = vec4(result, 1.0);
            }
        """
        return VERTEX_SHADER, FRAGMENT_SHADER
    
    def lorenz_attractor_src(self):
        # Temporarily keep the strings here, will move to file later
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
        # Temporarily keep the strings here, will move to file later
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
    
    def build_compute_shader(self, ProgramID: ProgramID) -> moderngl.ComputeShader:
        if ProgramID in self.compute_shaders:
            return self.compute_shaders[ProgramID]

        if ProgramID == ProgramID.LORENZ_ATTRACTOR:
            COMPUTE_SOURCE = self.lorenz_attractor_compute_src()
        else:
            print('no valid compute shader source code available') 
            return None # Changed to return None for clarity
        
        compute_shader = self.ctx.compute_shader(COMPUTE_SOURCE)
        
        self.compute_shaders[ProgramID] = compute_shader
        return compute_shader


    def build_program(self, ProgramID: ProgramID) -> moderngl.Program: # think of as the material 
        if ProgramID in self.programs:
            return self.programs[ProgramID]

        if ProgramID == ProgramID.BASIC_3D:
            VERTEX_SOURCE, FRAGMENT_SOURCE = self.basic_3d_src()
        elif ProgramID == ProgramID.LORENZ_ATTRACTOR:
            VERTEX_SOURCE, FRAGMENT_SOURCE = self.lorenz_attractor_src()
        elif ProgramID == ProgramID.GRID:
            VERTEX_SOURCE, FRAGMENT_SOURCE = self.grid_src()
        elif ProgramID == ProgramID.SURFACE:
            VERTEX_SOURCE, FRAGMENT_SOURCE = self.surface_src()
        else:
            print('no valid shader source code available') 
            return None # Changed to return None for clarity
        
        program = self.ctx.program(
            vertex_shader=VERTEX_SOURCE, 
            fragment_shader=FRAGMENT_SOURCE) 
        
        self.programs[ProgramID] = program
        return program