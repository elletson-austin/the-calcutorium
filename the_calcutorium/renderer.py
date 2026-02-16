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
                compute_uniforms=obj.compute_uniforms # Pass compute_uniforms here
            )
        
        elif obj.ProgramID == ProgramID.SURFACE:
            vbo = self.ctx.buffer(obj.vertices.tobytes())
            vao = self.ctx.vertex_array(program, [(vbo, "3f 3f 3f", "in_position", "in_normal", "in_color")])
            return RenderObject(
                program_id=obj.ProgramID,
                vao=vbo,
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
        else:
            ro.num_vertexes = len(obj.vertices) // 6
        
        # Update compute uniforms if available and applicable
        if obj.ProgramID == ProgramID.LORENZ_ATTRACTOR and hasattr(obj, 'compute_uniforms'):
            ro.compute_uniforms.update(obj.compute_uniforms)


    def render(self, render_objects: list, camera: 'Camera3D | Camera2D', width: int, height: int, h_range=None, v_range=None):

        for ro in render_objects:

            program = ro.vao.program
            program["u_view"].write(camera.get_view_matrix())
            program["u_proj"].write(camera.get_projection_matrix(width, height, h_range=h_range, v_range=v_range))
            
            if ro.program_id == ProgramID.SURFACE:
                program["u_model"].write(np.eye(4, dtype=np.float32).tobytes())
                program["u_light_pos"].write(np.array([10.0, 20.0, 10.0], dtype=np.float32).tobytes())
                program["u_view_pos"].write(camera.get_position().tobytes())
                
            if ro.program_id == ProgramID.GRID:
                # Import Camera3D for isinstance check
                from .camera import Camera3D
                alpha_multiplier = 0.2 if isinstance(camera, Camera3D) else 1.0
                if "u_alpha_multiplier" in program:
                    program["u_alpha_multiplier"].value = alpha_multiplier
            else:
                if "u_alpha_multiplier" in program:
                    program["u_alpha_multiplier"].value = 1.0

            if ro.Rendermode == RenderMode.POINTS:
                m = moderngl.POINTS
            elif ro.Rendermode == RenderMode.LINES:
                m = moderngl.LINES
            elif ro.Rendermode == RenderMode.LINE_STRIP:
                m = moderngl.LINE_STRIP
            elif ro.Rendermode == RenderMode.LINE_LOOP:
                m = moderngl.LINE_LOOP
            elif ro.Rendermode == RenderMode.TRIANGLES:
                m = moderngl.TRIANGLES
            else:
                m = moderngl.POINTS

            self.ctx.line_width = 2.0
            ro.vao.render(mode=m)