import pytest

from the_calcutorium.mathcore import BinOp, Call, MathError, Neg, Num, Var, parse, to_string


def test_precedence():
    assert parse("1 + 2*3") == BinOp("+", Num(1), BinOp("*", Num(2), Num(3)))


def test_left_associative_subtraction():
    assert parse("1 - 2 - 3") == BinOp("-", BinOp("-", Num(1), Num(2)), Num(3))


def test_power_is_right_associative():
    assert parse("2^3^2") == BinOp("^", Num(2), BinOp("^", Num(3), Num(2)))


def test_double_star_is_power():
    assert parse("x**2") == parse("x^2")


def test_unary_minus_binds_looser_than_power():
    # -x^2 is -(x^2), like on paper (the old parser got this wrong)
    assert parse("-x^2") == Neg(BinOp("^", Var("x"), Num(2)))


def test_negative_exponent():
    assert parse("2^-x") == BinOp("^", Num(2), Neg(Var("x")))


def test_negative_literal_folds():
    assert parse("-3") == Num(-3)


@pytest.mark.parametrize("text, expected", [
    ("2x", "2*x"),
    ("xy", "x*y"),
    ("2(x+1)", "2*(x + 1)"),
    ("(x)(y)", "x*y"),
    ("x sin(y)", "x*sin(y)"),
    ("xsin(y)", "x*sin(y)"),
    ("2pi", "2*pi"),
    ("2|x|", "2*abs(x)"),
    ("|x||y|", "abs(x)*abs(y)"),
])
def test_implicit_multiplication(text, expected):
    assert parse(text) == parse(expected.replace("pi", "3.141592653589793"))


def test_absolute_value_bars():
    assert parse("|x - 1|") == Call("abs", (BinOp("-", Var("x"), Num(1)),))


def test_scientific_notation():
    assert parse("3e2") == Num(300.0)
    assert parse("1.5e-1") == Num(0.15)


def test_e_without_exponent_digits_is_eulers_number():
    assert parse("3e") == BinOp("*", Num(3), Num(2.718281828459045))


def test_longest_function_name_wins():
    assert parse("asinh(x)") == Call("asinh", (Var("x"),))
    assert parse("exp(x)") == Call("exp", (Var("x"),))


def test_function_with_several_args():
    assert parse("max(x, 1)") == Call("max", (Var("x"), Num(1)))


@pytest.mark.parametrize("bad", ["", "1 +", "(x", "x)", "sin x", "2 $ 3", "|x"])
def test_errors(bad):
    with pytest.raises(MathError):
        parse(bad)


@pytest.mark.parametrize("text", [
    "x + y*z", "(x + y)*z", "x - (y - z)", "-x^2", "(-x)^2", "2^3^2", "(2^3)^2",
    "sin(x)/cos(y)", "x/(y*z)", "integrate(t^2, t, 0, x)", "-3*x",
])
def test_to_string_round_trips(text):
    assert parse(to_string(parse(text))) == parse(text)
