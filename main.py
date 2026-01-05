from render_space import RenderSpace, glfw_init
from scene import Scene, Axes
from renderer import Renderer
import glfw

def main():
    # 1️⃣ Initialize render space and GLFW
    render_space = RenderSpace()
    glfw_init(render_space)  # sets up window and ModernGL context

    # 2️⃣ Create scene and add axes
    scene = Scene()
    axes = Axes(length=10.0)
    axes.name = "axes"  # for scene management
    scene.objects.append(axes)

    # 3️⃣ Create renderer and build render objects
    renderer = Renderer(scene, render_space)
    render_objects = []
    for obj in scene.objects:
        ro = renderer.create_render_object(obj)
        render_objects.append(ro)

    width, height = render_space.ctx.fbo.size  # get framebuffer size

    # 4️⃣ Main loop
    while not glfw.window_should_close(render_space.window):
        glfw.poll_events()

        # Clear screen
        render_space.ctx.clear(0.1, 0.1, 0.1, 1.0)

        # Update camera (optional movement handled in render_space.update)
        render_space.update(0.01)
        cam = render_space.cam

        # Render all objects
        for ro in render_objects:
            program = renderer.program_manager.programs[ro.program_id]
            program["u_view"].write(cam.get_view_matrix())
            program["u_proj"].write(cam.get_projection_matrix(width, height))
            ro.vao.render(mode=render_space.ctx.LINES)

        # Swap buffers
        glfw.swap_buffers(render_space.window)

    # Cleanup
    for ro in render_objects:
        ro.release()
    glfw.terminate()


if __name__ == "__main__":
    main()