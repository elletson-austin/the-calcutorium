"""End-to-end: open the real window, type commands, and check the pixels.

These open a window and need a GPU, so they're skipped by default. Run them with:

    uv run pytest -m ui

Screenshots are saved to tests/output/ so you can look at them.
"""

import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.ui

QtGui = pytest.importorskip("PySide6.QtGui")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

OUTPUT = Path(__file__).parent / "output"


@pytest.fixture(scope="module")
def window():
    from the_calcutorium.ui.main_window import MainWindow

    fmt = QtGui.QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)
    qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    win.show()
    pump(qt, 1.0)
    yield win
    win.close()
    pump(qt, 0.2)


@pytest.fixture
def run(window):
    """Clear the scene, then return a function that types commands into the window."""
    window.app.clear()
    window.app.set_view("3d")

    def type_commands(*lines: str):
        for line in lines:
            window._run(line)
    return type_commands


def pump(qt, seconds: float = 0.4) -> None:
    end = time.time() + seconds
    while time.time() < end:
        qt.processEvents()
        time.sleep(0.01)


def screenshot(window, name: str) -> np.ndarray:
    """Render the viewport and return it as an (h, w, 3) uint8 array."""
    pump(QtWidgets.QApplication.instance())
    image = window.viewport.grabFramebuffer()
    OUTPUT.mkdir(exist_ok=True)
    image.save(str(OUTPUT / f"{name}.png"))
    image = image.convertToFormat(QtGui.QImage.Format.Format_RGB888)
    w, h, stride = image.width(), image.height(), image.bytesPerLine()
    raw = np.frombuffer(image.constBits(), np.uint8, count=stride * h).reshape(h, stride)
    # .copy(): the array must own its memory; the QImage is freed when this function returns
    return raw[:, : w * 3].reshape(h, w, 3).copy()


def pixels_of(pixels: np.ndarray, color, tolerance: int = 25) -> int:
    target = np.array(color) * 255
    return int((np.abs(pixels.astype(int) - target) <= tolerance).all(axis=-1).sum())


def assert_all_plots_visible(window, pixels):
    for plot in window.app.scene.plots():
        assert pixels_of(pixels, plot.color) > 30, f"{plot.kind} '{plot.text}' isn't on screen"


def test_every_plot_is_drawn_in_2d(window, run):
    # Regression test: the last-drawn plot used to vanish (Qt corrupted its vertex array).
    run("view xy", "y = x", "y = -x/2", "y = 5.5", "y = x^2/4", "contour z = x^2 + y^2")
    assert_all_plots_visible(window, screenshot(window, "2d_xy"))


def test_xz_view(window, run):
    run("view xz", "z = 3x^2", "x = z^2/4")
    assert_all_plots_visible(window, screenshot(window, "2d_xz"))


def test_contour_in_3d(window, run):
    run("contour z = x^2 + y^2")
    assert_all_plots_visible(window, screenshot(window, "3d_contour"))


def test_surface_and_contour_together(window, run):
    run("z = sin(x)*cos(y)*3", "contour z = sin(x)*cos(y)*3")
    pixels = screenshot(window, "3d_surface_contour")
    assert pixels.std() > 10  # something was drawn (lit surfaces don't keep their exact color)


def test_tab_and_ranges_dont_crash(window, run):
    run("view xy", "range x 0 10", "range y -2 8", "y = sin(x)", "range auto", "view 3d")
    screenshot(window, "after_ranges")
