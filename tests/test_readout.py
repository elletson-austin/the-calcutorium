import math

import pytest

from the_calcutorium.mathcore import parse_equation
from the_calcutorium.render.readout import ReadoutLine, plane_point, readout
from the_calcutorium.scene import Contour, graph

RED = (1.0, 0.0, 0.0)
BLUE = (0.0, 0.0, 1.0)


def plot(text: str, color=RED):
    return graph(parse_equation(text), color)


def contour(text: str, color=RED):
    return Contour(parse_equation(text), color)


# ---- pixel → plane ----------------------------------------------------------------

def test_plane_point_corners():
    size, h_range, v_range = (200, 100), (-10, 10), (0, 5)
    assert plane_point((0, 0), size, h_range, v_range) == (-10, 5)        # top-left
    assert plane_point((200, 100), size, h_range, v_range) == (10, 0)     # bottom-right
    assert plane_point((100, 50), size, h_range, v_range) == (0, 2.5)     # center


# ---- the cursor line ----------------------------------------------------------------

def test_first_line_is_the_cursor_in_plane_axes():
    lines = readout([], "xz", (1.5, -2.25), step=0.01)
    assert lines == (ReadoutLine("x = 1.50,  z = -2.25"),)


def test_cursor_rounds_to_pixel_size():
    assert readout([], "xy", (3.14159, 2.71828), step=0.1)[0].text == "x = 3.1,  y = 2.7"
    assert readout([], "xy", (1234.5, -0.4), step=5)[0].text == "x = 1234,  y = 0"  # no "-0"


def test_yz_view_puts_z_first():
    assert readout([], "yz", (1, 2), step=1)[0].text == "z = 1,  y = 2"


# ---- curves ------------------------------------------------------------------------

def test_curve_value_at_cursor():
    lines = readout([plot("y = x^2")], "xy", (1.5, 0), step=0.01)
    assert lines[1] == ReadoutLine("y = x^2:  y = 2.25", RED, marker=(1.5, 2.25))


def test_curve_is_evaluated_at_the_shown_coordinate():
    # The cursor shows x = 1.5, so y must be 1.5^2, not 1.5003^2
    lines = readout([plot("y = x^2")], "xy", (1.5003, 0), step=0.01)
    assert lines[1].text == "y = x^2:  y = 2.25"


def test_sideways_curve_reads_along_vertical_axis():
    lines = readout([plot("x = y^2")], "xy", (0, 3), step=1)
    assert lines[1].text == "x = y^2:  x = 9"
    assert lines[1].marker == (9, 3)  # (h, v): x is horizontal


def test_constant_curve():
    lines = readout([plot("y = 3")], "xy", (7, 0), step=1)
    assert lines[1].text == "y = 3:  y = 3"
    assert lines[1].marker == (7, 3)


def test_curve_not_in_plane_is_skipped():
    assert len(readout([plot("z = x")], "xy", (1, 1), step=1)) == 1


def test_undefined_value():
    lines = readout([plot("y = sqrt(x)"), plot("y = 1/x")], "xy", (-1, 0), step=1)
    assert lines[1] == ReadoutLine("y = sqrt(x):  undefined", RED, marker=None)
    lines = readout([plot("y = 1/x")], "xy", (0, 0), step=1)
    assert lines[1].text == "y = 1/x:  undefined"


def test_integral_curve():
    lines = readout([plot("y = integrate(cos(t), t, 0, x)")], "xy", (math.pi / 2, 0), step=1e-6)
    assert lines[1].text.endswith("y = 1")


# ---- surfaces and contours -----------------------------------------------------------

def test_surface_value_at_cursor():
    lines = readout([plot("z = x^2 + y^2")], "xy", (1, 2), step=1)
    assert lines[1] == ReadoutLine("z = x^2 + y^2:  z = 5", RED, marker=None)


def test_surface_inputs_follow_axis_names_not_screen_order():
    # yz view: z is horizontal, y vertical. x = y - 10z must still get y and z right.
    lines = readout([plot("x = y - 10z")], "yz", (1, 3), step=1)  # z = 1, y = 3
    assert lines[1].text == "x = y - 10z:  x = -7"


def test_surface_in_other_plane_is_skipped():
    assert len(readout([plot("z = x*y")], "xz", (1, 1), step=1)) == 1


def test_surface_and_contour_of_same_equation_shown_once():
    plots = [plot("z = x*y", BLUE), contour("z = x*y")]
    lines = readout(plots, "xy", (2, 3), step=1)
    assert lines[1:] == (ReadoutLine("z = x*y:  z = 6", BLUE),)


def test_values_use_significant_digits():
    lines = readout([plot("z = x/y")], "xy", (1, 3), step=1)
    assert lines[1].text == "z = x/y:  z = 0.33333"


def test_order_follows_scene():
    lines = readout([plot("y = x", BLUE), plot("y = 2x")], "xy", (1, 0), step=1)
    assert [line.text for line in lines[1:]] == ["y = x:  y = 1", "y = 2x:  y = 2"]


@pytest.mark.parametrize("plane", ["xy", "xz", "yz"])
def test_readout_is_pure(plane):
    plots = [plot("y = x^2"), plot("z = x*y"), plot("x = sin(z)")]
    assert readout(plots, plane, (0.5, 0.25), 0.01) == readout(plots, plane, (0.5, 0.25), 0.01)
