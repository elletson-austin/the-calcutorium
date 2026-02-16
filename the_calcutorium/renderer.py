import moderngl
import numpy as np
from typing import TYPE_CHECKING

from .scene import SceneObject, ProgramID, RenderMode
from .program_manager import ProgramManager

if TYPE_CHECKING:
    from .camera import Camera3D, Camera2D

class RenderObject:

    def __init__(self,
        program_id: ProgramID,
        vao: moderngl.VertexArray,
        vbo: moderngl.Buffer,
        Rendermode: RenderMode,
        num_vertexes: int,
        compute_shader: moderngl.ComputeShader = None,
        compute_uniforms: dict = None, # Add compute_uniforms parameter
        ) -> None:
        
        self.program_id = program_id
        self.vao = vao
        self.vbo = vbo
        self.Rendermode = Rendermode
        self.num_vertexes = num_vertexes
        self.compute_shader = compute_shader
        self.compute_uniforms = compute_uniforms if compute_uniforms is not None else {}


    def release(self):
        self.vao.release()
        self.vbo.release()

class Renderer:
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.program_manager = ProgramManager(self.ctx)
    
    def _apply_compute_uniforms(self, compute_shader: moderngl.ComputeShader, uniforms: dict):
        for name, value in uniforms.items():
            if name in compute_shader:
                compute_shader[name].value = value

    def create_render_object(self, obj: SceneObject) -> RenderObject:
        program = self.program_manager.build_program(obj.ProgramID)
        
        if obj.ProgramID == ProgramID.LORENZ_ATTRACTOR:
            vbo = self.ctx.buffer(obj.vertices.tobytes(), dynamic=True)
            vao = self.ctx.vertex_array(program, [(vbo, "4f", "in_position")])
            compute_shader = self.program_manager.build_compute_shader(obj.ProgramID)
            return RenderObject(
                program_id=obj.ProgramID,
                vao=vao,
                vbo=vbo,
                Rendermode=obj.RenderMode,
                num_vertexes=obj.num_points,
                compute_shader=compute_shader,
                compute_uniforms=obj.uniforms # Pass compute_uniforms here
            )
        
        elif obj.ProgramID == ProgramID.SURFACE:
            vbo = self.ctx.buffer(obj.vertices.tobytes())
            vao = self.ctx.vertex_array(program, [(vbo, "3f 3f 3f", "in_position", "in_normal", "in_color")])
            return RenderObject(
                program_id=obj.ProgramID,
                vao=vao,
                vbo=vbo,
                Rendermode=obj.RenderMode,
                num_vertexes=len(obj.vertices) // 9
            )

        else: # For other objects like BASIC_3D and GRID
            vbo = self.ctx.buffer(obj.vertices.tobytes())
            vao = self.ctx.vertex_array(program, [(vbo, "3f 3f", "in_position", "in_color")])
            return RenderObject(
                program_id=obj.ProgramID,
                vao=vao,
                vbo=vbo,
                Rendermode=obj.RenderMode,
                num_vertexes=len(obj.vertices) // 6
            )

    def update_render_object(self, ro: RenderObject, obj: SceneObject):
        ro.vbo.write(obj.vertices.tobytes())
        if obj.ProgramID == ProgramID.SURFACE:
            ro.num_vertexes = len(obj.vertices) // 9
        elif obj.ProgramID == ProgramID.LORENZ_ATTRACTOR:
            # For Lorenz attractor vertices are N x 4 float32 points
            # and the SceneObject stores the intended number as num_points.
            ro.num_vertexes = getattr(obj, 'num_points', obj.vertices.shape[0])
        else:
            ro.num_vertexes = len(obj.vertices) // 6
        
        # Update compute uniforms if available and applicable
        if obj.ProgramID == ProgramID.LORENZ_ATTRACTOR and hasattr(obj, 'uniforms'):
            ro.compute_uniforms.update(obj.uniforms)


    def render(self, 
           render_objects: list, 
           camera: 'Camera3D | Camera2D', 
           width: int, 
           height: int, 
           h_range=None, 
           v_range=None):
    
        for ro in render_objects:
            # If this render object has a compute shader (e.g. Lorenz attractor),
            # bind its VBO as a storage buffer, apply uniforms and dispatch.
            if getattr(ro, 'compute_shader', None) is not None:
                # Bind the vertex buffer as an SSBO at binding point 0 which
                # matches the layout(binding = 0) in the compute shader.
                ro.vbo.bind_to_storage_buffer(0)

                # Apply any uniforms supplied for the compute shader
                self._apply_compute_uniforms(ro.compute_shader, ro.compute_uniforms)

                # Dispatch compute shader. local_size_x in shader is 256.
                groups = (ro.num_vertexes + 255) // 256
                ro.compute_shader.run(groups, 1, 1)

                # Try to issue a memory barrier if available so writes are visible
                # to subsequent rendering. Not all moderngl versions expose it.
                try:
                    self.ctx.memory_barrier()
                except Exception:
                    pass

            program = ro.vao.program
            
            self._set_camera_uniforms(program, camera, width, height, h_range, v_range)
            self._set_program_specific_uniforms(program, ro.program_id, camera)
            
            render_mode = self._get_render_mode(ro.Rendermode)
            
            self.ctx.line_width = 2.0
            ro.vao.render(mode=render_mode)


    def _set_camera_uniforms(self, program, camera, width, height, h_range, v_range):
        """Set view and projection matrices."""
        program["u_view"].write(camera.get_view_matrix())
        program["u_proj"].write(camera.get_projection_matrix(
            width, height, h_range=h_range, v_range=v_range
        ))


    def _set_program_specific_uniforms(self, program, program_id, camera):
        """Set uniforms specific to different shader programs."""
        if program_id == ProgramID.SURFACE:
            self._set_surface_uniforms(program, camera)
        
        # Set alpha multiplier based on program type and camera
        alpha_multiplier = self._get_alpha_multiplier(program_id, camera)
        if "u_alpha_multiplier" in program:
            program["u_alpha_multiplier"].value = alpha_multiplier


    def _set_surface_uniforms(self, program, camera):
        """Set uniforms for surface rendering."""
        program["u_model"].write(np.eye(4, dtype=np.float32).tobytes())
        program["u_light_pos"].write(np.array([10.0, 20.0, 10.0], dtype=np.float32).tobytes())
        program["u_view_pos"].write(camera.get_position().tobytes())


    def _get_alpha_multiplier(self, program_id, camera):
        """Calculate alpha multiplier based on program type and camera."""
        from .camera import Camera3D
        if program_id == ProgramID.GRID and isinstance(camera, Camera3D):
            return 0.2
        return 1.0


    def _get_render_mode(self, render_mode):
        """Map RenderMode enum to moderngl constant."""
        mode_map = {
            RenderMode.POINTS: moderngl.POINTS,
            RenderMode.LINES: moderngl.LINES,
            RenderMode.LINE_STRIP: moderngl.LINE_STRIP,
            RenderMode.LINE_LOOP: moderngl.LINE_LOOP,
            RenderMode.TRIANGLES: moderngl.TRIANGLES,
        }
        return mode_map.get(render_mode, moderngl.POINTS)