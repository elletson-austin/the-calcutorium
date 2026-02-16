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
        compute_bindings: list = None, # list of (buffer, binding_index) pairs for SSBOs
        compute_groups: tuple = None,  # explicit (groups_x, groups_y, groups_z)
        compute_local_size_x: int = 256, # fallback local size x used to compute groups
        ) -> None:
        
        self.program_id = program_id
        self.vao = vao
        self.vbo = vbo
        self.Rendermode = Rendermode
        self.num_vertexes = num_vertexes
        self.compute_shader = compute_shader
        self.compute_uniforms = compute_uniforms if compute_uniforms is not None else {}
        self.compute_bindings = compute_bindings if compute_bindings is not None else []
        self.compute_groups = compute_groups
        self.compute_local_size_x = compute_local_size_x


    def release(self):
        self.vao.release()
        try:
            self.vbo.release()
        except Exception:
            pass
        # Release any extra buffers provided in compute_bindings
        if getattr(self, 'compute_bindings', None):
            for buf, _ in self.compute_bindings:
                try:
                    # avoid double releasing the primary vbo
                    if buf is not self.vbo:
                        buf.release()
                except Exception:
                    pass

class Renderer:
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.program_manager = ProgramManager(self.ctx)
    
    def _apply_compute_uniforms(self, compute_shader: moderngl.ComputeShader, uniforms: dict):
        for name, value in uniforms.items():
            if name not in compute_shader:
                continue

            # Accept numpy arrays for uniform buffers or plain scalars
            try:
                # If it's a numpy array or has tobytes, write raw bytes
                if hasattr(value, 'tobytes') and not isinstance(value, (int, float, bool)):
                    compute_shader[name].write(value.tobytes())
                else:
                    compute_shader[name].value = value
            except Exception:
                # Best-effort: set .value fallback
                try:
                    compute_shader[name].value = value
                except Exception:
                    pass

    def create_render_object(self, obj: SceneObject) -> RenderObject:
        program = self.program_manager.build_program(obj.ProgramID)
        
        if obj.ProgramID == ProgramID.LORENZ_ATTRACTOR:
            vbo = self.ctx.buffer(obj.vertices.tobytes(), dynamic=True)
            vao = self.ctx.vertex_array(program, [(vbo, "4f", "in_position")])
            compute_shader = self.program_manager.build_compute_shader(obj.ProgramID)
            # Bind the vbo as the default SSBO at binding 0; for more
            # complicated sims (N-Body) callers can supply additional
            # compute_bindings when constructing the RenderObject.
            ro = RenderObject(
                program_id=obj.ProgramID,
                vao=vao,
                vbo=vbo,
                Rendermode=obj.RenderMode,
                num_vertexes=obj.num_points,
                compute_shader=compute_shader,
                compute_uniforms=obj.uniforms, # Pass compute_uniforms here
                compute_bindings=[(vbo, 0)],
                compute_local_size_x=256,
            )
            return ro
        elif obj.ProgramID == ProgramID.NBODY:
            # Create SSBOs for positions, velocities and masses
            pos_buf = self.ctx.buffer(obj.positions.tobytes(), dynamic=True)
            vel_buf = self.ctx.buffer(obj.velocities.tobytes(), dynamic=True)
            mass_buf = self.ctx.buffer(obj.masses.tobytes())

            vao = self.ctx.vertex_array(program, [(pos_buf, "4f", "in_position")])
            compute_shader = self.program_manager.build_compute_shader(obj.ProgramID)

            ro = RenderObject(
                program_id=obj.ProgramID,
                vao=vao,
                vbo=pos_buf,
                Rendermode=obj.RenderMode,
                num_vertexes=obj.num_bodies,
                compute_shader=compute_shader,
                compute_uniforms=obj.uniforms,
                compute_bindings=[(pos_buf, 0), (vel_buf, 1), (mass_buf, 2)],
                compute_local_size_x=256,
            )
            # Keep references to other buffers for updates
            ro.vel_buffer = vel_buf
            ro.mass_buffer = mass_buf
            return ro
        
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
        # Handle NBody objects (positions/velocities/masses) specially
        if obj.ProgramID == ProgramID.NBODY:
            # Update position buffer
            try:
                ro.vbo.write(obj.positions.tobytes())
            except Exception:
                pass
            # Update velocity buffer if present on the render object
            if hasattr(ro, 'vel_buffer') and hasattr(obj, 'velocities'):
                try:
                    ro.vel_buffer.write(obj.velocities.tobytes())
                except Exception:
                    pass
            # Update mass buffer if present
            if hasattr(ro, 'mass_buffer') and hasattr(obj, 'masses'):
                try:
                    ro.mass_buffer.write(obj.masses.tobytes())
                except Exception:
                    pass
            ro.num_vertexes = getattr(obj, 'num_bodies', obj.positions.shape[0])
        else:
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
        if hasattr(obj, 'uniforms'):
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
            # bind its SSBOs, apply uniforms and dispatch. Supports multiple
            # bindings to make it easy to add N-Body style sims later.
            if getattr(ro, 'compute_shader', None) is not None:
                # Bind any configured buffers to their binding points. If none
                # are provided, fall back to binding the primary vbo at 0.
                if getattr(ro, 'compute_bindings', None):
                    for buf, binding in ro.compute_bindings:
                        try:
                            buf.bind_to_storage_buffer(binding)
                        except Exception:
                            pass
                else:
                    # Backwards-compatible single buffer binding
                    try:
                        ro.vbo.bind_to_storage_buffer(0)
                    except Exception:
                        pass

                # Apply any uniforms supplied for the compute shader
                self._apply_compute_uniforms(ro.compute_shader, ro.compute_uniforms)

                # Determine dispatch size. If explicit groups provided, use them.
                if getattr(ro, 'compute_groups', None) is not None:
                    gx, gy, gz = ro.compute_groups
                elif getattr(ro, 'num_vertexes', None) is not None:
                    lx = getattr(ro, 'compute_local_size_x', 256)
                    gx = (ro.num_vertexes + lx - 1) // lx
                    gy, gz = 1, 1
                else:
                    gx, gy, gz = 1, 1, 1

                # Dispatch compute shader
                try:
                    ro.compute_shader.run(gx, gy, gz)
                except Exception:
                    # If run fails with ints vs tuples, try alternative call
                    try:
                        ro.compute_shader.run((gx, gy, gz))
                    except Exception:
                        pass

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
        elif program_id == ProgramID.NBODY:
            # Basic uniforms for N-body rendering: model matrix, point size and color
            if "u_model" in program:
                program["u_model"].write(np.eye(4, dtype=np.float32).tobytes())
            if "u_point_size" in program:
                try:
                    program["u_point_size"].value = 5.0
                except Exception:
                    pass
            if "u_color" in program:
                try:
                    program["u_color"].write(np.array([1.0, 1.0, 1.0], dtype=np.float32).tobytes())
                except Exception:
                    try:
                        program["u_color"].value = (1.0, 1.0, 1.0)
                    except Exception:
                        pass
        
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