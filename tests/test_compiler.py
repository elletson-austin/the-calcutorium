import math

import numpy as np
import pytest

from the_calcutorium.mathcore import MathError, compile_expr, parse


def f(text: str, *inputs: str):
    return compile_expr(parse(text), inputs)


def test_evaluates_over_arrays():
    np.testing.assert_allclose(f("x^2 + 1", "x")(np.array([0, 1, 2])), [1, 2, 5])


def test_constants_broadcast_to_input_shape():
    assert f("5", "x")(np.zeros(4)).shape == (4,)


def test_two_inputs_broadcast():
    out = f("x*y", "x", "y")(np.array([[1], [2]]), np.array([10, 20, 30]))
    assert out.shape == (2, 3)


def test_argument_order_follows_inputs():
    assert f("x - y", "x", "y")(5.0, 3.0) == 2.0
    assert f("x - y", "y", "x")(5.0, 3.0) == -2.0


def test_undefined_points_are_nan_not_errors():
    out = f("sqrt(x)", "x")(np.array([-1.0, 4.0]))
    assert math.isnan(out[0]) and out[1] == 2.0


def test_unknown_variable():
    with pytest.raises(MathError, match="Unknown variable"):
        f("x + q", "x")


def test_wrong_argument_count():
    with pytest.raises(MathError):
        f("sin(x, 2)", "x")


def test_definite_integral():
    assert f("integrate(t^2, t, 0, 3)")() == pytest.approx(9.0)


def test_integral_with_variable_bound_is_an_antiderivative():
    xs = np.linspace(-2, 2, 9)
    np.testing.assert_allclose(f("integrate(cos(t), t, 0, x)", "x")(xs), np.sin(xs), atol=1e-12)


def test_integrand_can_use_other_variables():
    # ∫_0^1 x*t dt = x/2
    np.testing.assert_allclose(f("integrate(x*t, t, 0, 1)", "x")(np.array([2.0, 4.0])), [1.0, 2.0])


def test_double_integral():
    # ∫_0^1 ∫_0^2 (s*t) ds dt = 1/2 * 2 = 1
    assert f("integrate(integrate(s*t, s, 0, 2), t, 0, 1)")() == pytest.approx(1.0)


def test_integrate_requires_explicit_variable_and_bounds():
    with pytest.raises(MathError, match="integrate"):
        f("integrate(x)", "x")
