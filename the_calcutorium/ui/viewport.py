"""The OpenGL view. Reads the scene and view state each frame and draws them.

It never changes the scene. The only thing it tells the app is "the user dragged
or zoomed", which clears manual ranges.

Controls:
    3D: left-drag orbits, scroll zooms, WASD moves, Tab toggles perspective/orthographic
    2D: left-drag pans, scroll zooms, hovering shows the values under the cursor
"""

from dataclasses import dataclass, field

import moderngl
import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from ..app import App, ViewState
from ..render.camera import Camera2D, Camera3D
from ..render.meshes import GridLabels, Mesh, axes_mesh, grid_mesh, plot_mesh
from ..render.readout import ReadoutLine, plane_point, readout
from ..render.renderer import Renderer
from ..scene import Plot
from .overlay import draw_overlay

ZOOM_STEP = 1.1
MOVE_SPEED = 20.0  # world units per second, WASD in 3D
FRAME_SECONDS = 0.016


@dataclass
class _Input:
    """Mouse/keyboard activity since the last frame."""
    mouse_pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    mouse_delta: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    left_down: bool = False
    scroll: float = 0.0
    keys: set = field(default_factory=set)


class Viewport(QOpenGLWidget):
    def __init__(self, app: App, parent=None):
        super().__init__(parent)
        self.app = app
        self.camera: Camera3D | Camera2D = Camera3D()
        self.input = _Input()
        self.hovering = False
        self.ctx: moderngl.Context | None = None
        self.renderer: Renderer | None = None
        self._grid_labels = GridLabels((), ())
        self._anchors: dict[Plot, np.ndarray | None] = {}

        app.view_changed.subscribe(self._on_view_changed)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(int(FRAME_SECONDS * 1000))

    # ---- view state → camera ----------------------------------------------

    def _on_view_changed(self, view: ViewState) -> None:
        cam = self.camera
        if view.mode == "3d" and not isinstance(cam, Camera3D):
            self.camera = Camera3D(center=cam.center, distance=cam.distance)
        elif view.is_2d and (not isinstance(cam, Camera2D) or cam.plane != view.mode):
            self.camera = Camera2D(view.mode, center=cam.center, distance=cam.distance)
        if view.fixed_ranges is not None:
            self.camera.fit(*view.fixed_ranges)

    # ---- GL lifecycle ------------------------------------------------------

    def initializeGL(self) -> None:
        self.ctx = moderngl.create_context()
        self.renderer = Renderer(self.ctx)

    def paintGL(self) -> None:
        if self.ctx is None:
            return
        width, height = self.width(), self.height()
        if width == 0 or height == 0:
            return
        self._apply_input()

        fbo = self.ctx.detect_framebuffer(self.defaultFramebufferObject())
        fbo.use()
        fbo.clear(0.1, 0.1, 0.1, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        view = self.app.view
        if isinstance(self.camera, Camera2D):
            ranges = view.fixed_ranges or self.camera.visible_ranges(width, height)
            proj = self.camera.projection_matrix(width, height, ranges)
        else:
            ranges = None
            proj = self.camera.projection_matrix(width, height)

        if ranges is not None:
            self.ctx.disable(moderngl.DEPTH_TEST)  # 2D is flat: later draws simply go on top

        r = self.renderer
        r.begin_frame(self.camera.view_matrix(), proj, self.camera.position())
        self._draw_scene(view, ranges)
        r.end_frame()

        # Unbind our vertex array before Qt draws anything. Otherwise QPainter (the
        # overlay text) and Qt's own compositing write into whichever VAO we drew
        # last, corrupting that plot for every frame after. Qt's docs require this.
        self.context().extraFunctions().glBindVertexArray(0)

        if ranges is not None:
            labels = [(p.text, p.color, a) for p, a in self._anchors.items() if a is not None]
            lines = self._readout(view.mode, ranges) if self.hovering else ()
            draw_overlay(self, view.mode, *ranges, self._grid_labels, labels, lines)

    def _readout(self, plane: str, ranges) -> tuple[ReadoutLine, ...]:
        """Mouse pixel → plane point → readout lines (all the work is in render/readout.py)."""
        (h_range, v_range), size = ranges, (self.width(), self.height())
        mouse = (float(self.input.mouse_pos[0]), float(self.input.mouse_pos[1]))
        point = plane_point(mouse, size, h_range, v_range)
        step = min((h_range[1] - h_range[0]) / size[0], (v_range[1] - v_range[0]) / size[1])
        return readout(self.app.scene.plots(), plane, point, step)

    def _draw_scene(self, view: ViewState, ranges) -> None:
        r = self.renderer
        plane = view.mode if view.is_2d else None

        if plane is None:
            r.draw_mesh("axes", (), axes_mesh)
        else:
            r.draw_mesh("grid", (plane, ranges), lambda: self._build_grid(plane, ranges))

        plots = self.app.scene.plots()
        self._anchors = {p: self._anchors.get(p) for p in plots}
        for plot in plots:
            signature = (plane, ranges) if plane is not None else None
            r.draw_mesh(plot, signature, lambda p=plot: self._build_plot(p, plane, ranges))

        for obj in self.app.scene.objects:
            if not isinstance(obj, Plot) and plane is None:
                r.draw_simulation(type(obj), obj)  # keyed by type: GPU state survives param edits

    def _build_grid(self, plane: str, ranges) -> Mesh:
        mesh, self._grid_labels = grid_mesh(plane, *ranges)
        return mesh

    def _build_plot(self, plot: Plot, plane: str | None, ranges) -> Mesh | None:
        mesh = plot_mesh(plot, plane, *ranges) if ranges else plot_mesh(plot)
        self._anchors[plot] = mesh.data[-1, :3] if mesh is not None and plane is not None else None
        return mesh

    def closeEvent(self, event) -> None:
        if self.renderer is not None:
            self.makeCurrent()
            self.renderer.release()
        super().closeEvent(event)

    # ---- input -------------------------------------------------------------

    def _apply_input(self) -> None:
        state, cam = self.input, self.camera
        dragged = state.left_down and bool(state.mouse_delta.any())
        zoom = (1 / ZOOM_STEP if state.scroll > 0 else ZOOM_STEP) if state.scroll else None

        if isinstance(cam, Camera3D):
            if dragged:
                cam.orbit(state.mouse_delta[0], state.mouse_delta[1])
            forward = (Qt.Key.Key_W in state.keys) - (Qt.Key.Key_S in state.keys)
            right = (Qt.Key.Key_D in state.keys) - (Qt.Key.Key_A in state.keys)
            if forward or right:
                step = FRAME_SECONDS * MOVE_SPEED
                cam.move(forward * step, right * step)
        else:
            if (dragged or zoom) and self.app.view.ranges:
                self.app.auto_range()  # user took control of the view
            if dragged:
                cam.pan(state.mouse_delta[0], state.mouse_delta[1], self.width(), self.height())
        if zoom:
            cam.zoom(zoom)

        state.mouse_delta[:] = 0
        state.scroll = 0.0

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.input.left_down = True
        self.input.mouse_pos[:] = (event.position().x(), event.position().y())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.input.left_down = False

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = np.array([event.position().x(), event.position().y()], dtype=np.float32)
        self.input.mouse_delta += pos - self.input.mouse_pos
        self.input.mouse_pos = pos

    def wheelEvent(self, event: QWheelEvent) -> None:
        self.input.scroll += event.angleDelta().y() / 120.0

    def enterEvent(self, event) -> None:
        self.hovering = True
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.hovering = False
        super().leaveEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        self.input.keys.add(event.key())
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        self.input.keys.discard(event.key())
        super().keyReleaseEvent(event)

    def toggle_projection(self) -> None:
        if isinstance(self.camera, Camera3D):
            self.camera.toggle_projection()


class TabToggleFilter(QObject):
    """Qt uses Tab to move focus between widgets. While the mouse is over the
    viewport, steal it to toggle perspective/orthographic instead."""

    def __init__(self, viewport: Viewport):
        super().__init__(viewport)
        self.viewport = viewport

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Tab and self.viewport.hovering:
            self.viewport.toggle_projection()
            return True
        return False
