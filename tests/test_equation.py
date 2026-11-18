import numpy as np
import pytest

from the_calcutorium.mathcore import MathError, parse_equation, sample_curve, sample_surface


@pytest.mark.parametrize("text, output, inputs", [
    ("x^2", "y", ("x",)),
    ("y = x^2", "y", ("x",)),
    ("y^2", "x", ("y",)),
    ("z = sin(x)", "z", ("x",)),
    ("x*y", "z", ("x", "y")),
    ("x = y*z", "x", ("y", "z")),
    ("3", "y", ()),
    ("y = integrate(t^2, t, 0, x)", "y", ("x",)),
])
def test_output_and_inputs(text, output, inputs):
    eq = parse_equation(text)
    assert (eq.output, eq.inputs) == (output, inputs)


@pytest.mark.parametrize("bad, message", [
    ("y = a*x", "Unknown variable"),
    ("x*y*z", "at most 2"),
    ("y = y + x", "both sides"),
    ("q = x", "Left side"),
    ("y = x = 2", "at most one"),
])
def test_invalid_equations(bad, message):
    with pytest.raises(MathError, match=message):
        parse_equation(bad)


def test_sample_curve_places_points_on_the_right_axes():
    pts = sample_curve(parse_equation("z = 2x"), (0, 1), n=3)
    np.testing.assert_allclose(pts, [[0, 0, 0], [0.5, 0, 1], [1, 0, 2]])


def test_sample_constant():
    pts = sample_curve(parse_equation("y = 3"), (-1, 1), n=2)
    np.testing.assert_allclose(pts, [[-1, 3, 0], [1, 3, 0]])


def test_sample_surface_shape_and_values():
    pts = sample_surface(parse_equation("x*y"), (0, 1), (0, 2), n=3)
    assert pts.shape == (3, 3, 3)
    np.testing.assert_allclose(pts[..., 2], pts[..., 0] * pts[..., 1])


def test_sampling_is_deterministic():
    eq = parse_equation("sin(x)")
    np.testing.assert_array_equal(sample_curve(eq, (0, 1)), sample_curve(eq, (0, 1)))
