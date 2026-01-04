import glfw
import time

from render_space import RenderSpace, glfw_init
from renderer import Renderer
from scene import *

def main():
    render_space = RenderSpace()
    glfw_init(render_space_global=render_space)
    renderer = Renderer(render_space)

    last_time = time.perf_counter()

    while not glfw.window_should_close(render_space.window):
        current_time = time.perf_counter()
        dt = current_time - last_time
        last_time = current_time

        # Poll input
        glfw.poll_events()

        # Update camera
        render_space.update(dt)

        # Clear screen
        render_space.ctx.clear(0.0, 0.2, 0.2)
        render_space.ctx.enable_only(render_space.ctx.DEPTH_TEST)

        width, height = glfw.get_framebuffer_size(render_space.window)
        
        renderer.render(render_space.cam, width, height)

        # Swap buffers
        glfw.swap_buffers(render_space.window)

    # Cleanup
    glfw.terminate()


if __name__ == "__main__":
    main()