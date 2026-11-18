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
    window.app.auto_range()
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


def test_hover_readout(window, run):
    run("view xy", "y = x^2/4", "contour z = x^2 + y^2")
    viewport = window.viewport
    w, h = viewport.width(), viewport.height()
    (h0, h1), (v0, v1) = viewport.camera.visible_ranges(w, h)
    viewport.hovering = False  # the real mouse may happen to be over the window
    before = screenshot(window, "hover_off")

    viewport.hovering = True  # what moving the mouse in does
    viewport.input.mouse_pos[:] = ((4 - h0) / (h1 - h0) * w, h / 2)  # at x = 4: the dot is at (4, 4)
    after = screenshot(window, "hover_readout")

    # Screenshots are in physical pixels (w, h above are logical), so scale by the image's size
    top_right = (slice(0, after.shape[0] // 8), slice(after.shape[1] // 2, None))
    assert (after[top_right] != before[top_right]).any(), "no readout box drawn"

    # Nothing else differs between the two shots, so pixels that changed around the dot's
    # spot were changed by the dot. (Counting curve-colored pixels doesn't work: at 1x
    # scale the dot's dark outline covers about as many line pixels as the dot adds.)
    scale = after.shape[1] / w  # physical pixels per logical pixel: 1, 1.5, 2, ...
    dot_x = round((4 - h0) / (h1 - h0) * after.shape[1])
    dot_y = round((v1 - 4) / (v1 - v0) * after.shape[0])
    r = round(8 * scale)
    around = (slice(dot_y - r, dot_y + r), slice(dot_x - r, dot_x + r))
    changed = int((after[around] != before[around]).any(axis=-1).sum())
    assert changed > 20 * scale**2, f"no dot on the curve at the cursor ({changed} pixels changed)"


def test_contour_in_3d(window, run):
    run("contour z = x^2 + y^2")
    assert_all_plots_visible(window, screenshot(window, "3d_contour"))


def test_surface_and_contour_together(window, run):
    run("z = sin(x)*cos(y)*3", "contour z = sin(x)*cos(y)*3")
    pixels = screenshot(window, "3d_surface_contour")
    assert pixels.std() > 10  # something was drawn (lit surfaces don't keep their exact color)


def contour_row(window):
    from the_calcutorium.ui.plot_list import PlotList
    plot_list = window.findChild(PlotList)
    (row,) = plot_list.rows.values()
    return row


def test_contour_interval_box(window, run):
    run("view xy", "contour z = x^2 + y^2")
    row = contour_row(window)
    assert row.contour_settings.isVisibleTo(row) and row.interval.text() == "auto"

    row.interval.setValue(50)  # what typing 50 + Enter does
    (plot,) = window.app.scene.plots()
    assert plot.interval == 50
    assert_all_plots_visible(window, screenshot(window, "contour_every_50"))

    run("contour z = x^2 + y^2 every 200")  # a typed command updates the box too
    assert contour_row(window).interval.value() == 200

    contour_row(window).interval.setValue(0)
    assert window.app.scene.plots()[0].interval is None


def test_curve_rows_have_no_interval_box(window, run):
    run("y = x^2")
    assert not contour_row(window).contour_settings.isVisibleTo(contour_row(window))


def test_tab_and_ranges_dont_crash(window, run):
    run("view xy", "range x 0 10", "range y -2 8", "y = sin(x)", "range auto", "view 3d")
    screenshot(window, "after_ranges")
