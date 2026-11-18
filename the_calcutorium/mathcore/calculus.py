"""Symbolic calculus on expression trees. Every function here is pure."""

from .errors import MathError
from .expr import BinOp, Call, Expr, Neg, Num, Var, free_vars, substitute

ZERO, ONE, TWO = Num(0.0), Num(1.0), Num(2.0)


# ---- tiny constructors, to keep the derivative table readable ---------------

def _add(a, b): return BinOp("+", a, b)
def _sub(a, b): return BinOp("-", a, b)
def _mul(a, b): return BinOp("*", a, b)
def _div(a, b): return BinOp("/", a, b)
def _pow(a, b): return BinOp("^", a, b)
def _call(name, u): return Call(name, (u,))


# d/du f(u), for each built-in single-argument function f
_DERIVATIVES = {
    "sin":   lambda u: _call("cos", u),
    "cos":   lambda u: Neg(_call("sin", u)),
    "tan":   lambda u: _div(ONE, _pow(_call("cos", u), TWO)),
    "asin":  lambda u: _div(ONE, _call("sqrt", _sub(ONE, _pow(u, TWO)))),
    "acos":  lambda u: Neg(_div(ONE, _call("sqrt", _sub(ONE, _pow(u, TWO))))),
    "atan":  lambda u: _div(ONE, _add(ONE, _pow(u, TWO))),
    "sinh":  lambda u: _call("cosh", u),
    "cosh":  lambda u: _call("sinh", u),
    "tanh":  lambda u: _sub(ONE, _pow(_call("tanh", u), TWO)),
    "asinh": lambda u: _div(ONE, _call("sqrt", _add(_pow(u, TWO), ONE))),
    "acosh": lambda u: _div(ONE, _call("sqrt", _sub(_pow(u, TWO), ONE))),
    "atanh": lambda u: _div(ONE, _sub(ONE, _pow(u, TWO))),
    "sec":   lambda u: _mul(_call("sec", u), _call("tan", u)),
    "csc":   lambda u: Neg(_mul(_call("csc", u), _call("cot", u))),
    "cot":   lambda u: Neg(_pow(_call("csc", u), TWO)),
    "exp":   lambda u: _call("exp", u),
    "log":   lambda u: _div(ONE, u),
    "ln":    lambda u: _div(ONE, u),
    "sqrt":  lambda u: _div(ONE, _mul(TWO, _call("sqrt", u))),
    "cbrt":  lambda u: _div(ONE, _mul(Num(3.0), _pow(_call("cbrt", u), TWO))),
    "abs":   lambda u: _call("sign", u),
    "sign":  lambda u: ZERO,
    "floor": lambda u: ZERO,
    "ceil":  lambda u: ZERO,
}


def diff(expr: Expr, var: str) -> Expr:
    """∂expr/∂var, simplified."""
    return simplify(_diff(expand_calculus(expr), var))


def _diff(expr: Expr, var: str) -> Expr:
    # Anything that doesn't depend on var is a constant.
    if var not in free_vars(expr):
        return ZERO

    match expr:
        case Var():
            return ONE  # it must be var itself, or free_vars would have caught it
        case Neg(f):
            return Neg(_diff(f, var))
        case BinOp("+" | "-" as op, f, g):
            return BinOp(op, _diff(f, var), _diff(g, var))
        case BinOp("*", f, g):  # product rule
            return _add(_mul(_diff(f, var), g), _mul(f, _diff(g, var)))
        case BinOp("/", f, g):  # quotient rule
            return _div(_sub(_mul(_diff(f, var), g), _mul(f, _diff(g, var))), _pow(g, TWO))
        case BinOp("^", f, g) if var not in free_vars(g):  # power rule: n f^(n-1) f'
            return _mul(_mul(g, _pow(f, _sub(g, ONE))), _diff(f, var))
        case BinOp("^", f, g):  # general: f^g (g' ln f + g f'/f)
            return _mul(expr, _add(_mul(_diff(g, var), _call("ln", f)), _mul(g, _div(_diff(f, var), f))))
        case Call("integrate", (f, Var(t), a, b)):
            return _diff_integral(f, t, a, b, var)
        case Call(name, (u,)) if name in _DERIVATIVES:  # chain rule
            return _mul(_DERIVATIVES[name](u), _diff(u, var))
        case Call(name, _):
            raise MathError(f"Can't differentiate {name}() with respect to {var}")
    raise TypeError(f"Not an expression: {expr!r}")


def _diff_integral(f: Expr, t: str, a: Expr, b: Expr, var: str) -> Expr:
    """Fundamental theorem of calculus: d/dx ∫_a^b f(t) dt = f(b) b' − f(a) a'."""
    if var in free_vars(f) - {t}:
        raise MathError(f"Can't differentiate an integral whose integrand depends on {var}")
    return _sub(_mul(substitute(f, t, b), _diff(b, var)), _mul(substitute(f, t, a), _diff(a, var)))


def expand_calculus(expr: Expr) -> Expr:
    """Replace every diff(f, x) in the tree with the actual derivative.

    integrate(...) is left in place: it's evaluated numerically when compiled.
    """
    match expr:
        case Num() | Var():
            return expr
        case Neg(f):
            return Neg(expand_calculus(f))
        case BinOp(op, f, g):
            return BinOp(op, expand_calculus(f), expand_calculus(g))
        case Call("diff", (f, Var(x))):
            return diff(f, x)
        case Call("diff", (f,)):
            variables = free_vars(f)
            if len(variables) != 1:
                names = ", ".join(sorted(variables)) or "none"
                raise MathError(
                    f"diff() needs a variable when the expression has variables {names}: write diff(f, x)"
                )
            return diff(f, next(iter(variables)))
        case Call("diff", _):
            raise MathError("diff() is written diff(f) or diff(f, x)")
        case Call("integrate", (f, Var() as t, a, b)):
            return Call("integrate", (expand_calculus(f), t, expand_calculus(a), expand_calculus(b)))
        case Call("integrate", _):
            raise MathError("integrate() is written integrate(f, t, a, b): ∫ from a to b of f dt")
        case Call(name, args):
            return Call(name, tuple(expand_calculus(a) for a in args))
    raise TypeError(f"Not an expression: {expr!r}")


# ---- simplification ---------------------------------------------------------

_FOLD = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b if b != 0 else None,
    "^": lambda a, b: a ** b if not (a < 0 and b != int(b)) and not (a == 0 and b < 0) else None,
}


def simplify(expr: Expr) -> Expr:
    """Light algebraic cleanup: constant folding and identities like x*1 = x."""
    match expr:
        case Num() | Var():
            return expr
        case Neg(f):
            f = simplify(f)
            match f:
                case Num(v):
                    return Num(-v)
                case Neg(inner):
                    return inner
            return Neg(f)
        case BinOp(op, left, right):
            return _simplify_binop(op, simplify(left), simplify(right))
        case Call(name, args):
            return Call(name, tuple(simplify(a) for a in args))
    raise TypeError(f"Not an expression: {expr!r}")


def _simplify_binop(op: str, a: Expr, b: Expr) -> Expr:
    if isinstance(a, Num) and isinstance(b, Num):
        try:
            value = _FOLD[op](a.value, b.value)
        except OverflowError:
            value = None
        if value is not None:
            return Num(float(value))

    match op, a, b:
        case "+", Num(0.0), _: return b
        case "+", _, Num(0.0): return a
        case "+", _, Neg(inner): return simplify(BinOp("-", a, inner))
        case "-", _, Num(0.0): return a
        case "-", Num(0.0), _: return simplify(Neg(b))
        case "-", _, Neg(inner): return simplify(BinOp("+", a, inner))
        case "*", Num(0.0), _: return ZERO
        case "*", _, Num(0.0): return ZERO
        case "*", Num(1.0), _: return b
        case "*", _, Num(1.0): return a
        case "*", Num(-1.0), _: return simplify(Neg(b))
        case "*", _, Num(-1.0): return simplify(Neg(a))
        case "*", Neg(x), Neg(y): return simplify(BinOp("*", x, y))
        case "*", Neg(x), _: return simplify(Neg(BinOp("*", x, b)))
        case "*", _, Neg(y): return simplify(Neg(BinOp("*", a, y)))
        case "*", _, Num() if not isinstance(a, Num): return BinOp("*", b, a)  # constants first: 2*x
        case "/", Num(0.0), _: return ZERO
        case "/", _, Num(1.0): return a
        case "^", _, Num(0.0): return ONE
        case "^", _, Num(1.0): return a
        case "^", Num(1.0), _: return ONE
    if op == "-" and a == b:
        return ZERO
    # Note: no x/x → 1. That would silently remove the hole at x = 0.
    return BinOp(op, a, b)
