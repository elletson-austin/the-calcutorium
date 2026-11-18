import numpy as np
import pytest

from the_calcutorium.mathcore import MathError, compile_expr, diff, parse, simplify, to_string


def d(text: str, var: str) -> str:
    return to_string(diff(parse(text), var))


def numeric_check(f_text: str, var: str, at: dict[str, float]):
    """Compare the symbolic derivative to a central finite difference."""
    names = tuple(sorted(at))
    f = compile_expr(parse(f_text), names)
    df = compile_expr(diff(parse(f_text), var), names)
    h = 1e-6
    point = [at[n] for n in names]
    plus = [v + h if n == var else v for n, v in zip(names, point)]
    minus = [v - h if n == var else v for n, v in zip(names, point)]
    expected = (f(*plus) - f(*minus)) / (2 * h)
    assert df(*point) == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_basic_rules():
    assert d("x^3", "x") == "3*x^2"
    assert d("5", "x") == "0"
    assert d("x", "x") == "1"
    assert d("sin(x)", "x") == "cos(x)"


def test_partial_derivatives_use_the_variable_you_give():
    assert d("x^2*y", "x") == "2*x*y"
    assert d("x^2*y", "y") == "x^2"


def test_mixed_partials_are_equal():
    f = parse("sin(x*y) + x^3*y^2")
    fxy = compile_expr(diff(diff(f, "x"), "y"), ("x", "y"))
    fyx = compile_expr(diff(diff(f, "y"), "x"), ("x", "y"))
    xs, ys = np.meshgrid(np.linspace(-2, 2, 7), np.linspace(-2, 2, 7))
    np.testing.assert_allclose(fxy(xs, ys), fyx(xs, ys))


def test_diff_is_pure():
    f = parse("x^2*y")
    before = repr(f)
    diff(f, "x")
    assert repr(f) == before


@pytest.mark.parametrize("f", [
    "sin(x)*cos(x)", "x^x", "exp(2x)/x", "ln(x^2 + 1)", "sqrt(x)", "cbrt(x)",
    "tan(x)", "asin(x/2)", "acos(x/2)", "atan(x)", "sinh(x)", "cosh(x)", "tanh(x)",
    "asinh(x)", "acosh(x + 2)", "atanh(x/2)", "sec(x)", "csc(x)", "cot(x)", "|x|",
    "2^x", "-x^2",
])
def test_derivatives_match_finite_differences(f):
    numeric_check(f, "x", {"x": 0.7})


def test_partial_derivative_numerically():
    numeric_check("x^2*sin(y) + exp(x*y)", "y", {"x": 0.3, "y": 1.1})


def test_diff_inside_an_expression_with_explicit_variable():
    f = compile_expr(parse("diff(x^2*y, y)"), ("x", "y"))
    assert f(3.0, 5.0) == pytest.approx(9.0)


def test_single_variable_diff_needs_no_variable():
    f = compile_expr(parse("diff(x^3)"), ("x",))
    assert f(2.0) == pytest.approx(12.0)


def test_diff_refuses_to_guess_the_variable():
    with pytest.raises(MathError, match="needs a variable"):
        compile_expr(parse("diff(x*y)"), ("x", "y"))


def test_nested_diff():
    f = compile_expr(parse("diff(diff(x^4, x), x)"), ("x",))
    assert f(1.0) == pytest.approx(12.0)


def test_fundamental_theorem_of_calculus():
    # d/dx ∫_0^x t^2 dt = x^2
    assert d("integrate(t^2, t, 0, x)", "x") == "x^2"


def test_simplify():
    assert to_string(simplify(parse("0 + x*1"))) == "x"
    assert to_string(simplify(parse("x - x"))) == "0"
    assert to_string(simplify(parse("2*3 + x^0"))) == "7"
    assert to_string(simplify(parse("x/x"))) == "x/x"  # keeps the hole at x = 0
