"""Expression trees.

An expression like `2*sin(x) + y` is stored as an immutable tree:

    BinOp("+", BinOp("*", Num(2), Call("sin", (Var("x"),))), Var("y"))

Nodes are frozen dataclasses: they can't be modified after creation. Every
operation (diff, simplify, substitute, ...) builds a *new* tree.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Num:
    value: float


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Neg:
    operand: "Expr"


@dataclass(frozen=True)
class BinOp:
    op: str  # one of + - * / ^
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple["Expr", ...]


Expr = Num | Var | Neg | BinOp | Call


def free_vars(expr: Expr) -> frozenset[str]:
    """Variables the expression's value depends on.

    Bound variables don't count: in integrate(t^2, t, 0, x), `t` is bound and
    only `x` is free.
    """
    match expr:
        case Num():
            return frozenset()
        case Var(name):
            return frozenset({name})
        case Neg(operand):
            return free_vars(operand)
        case BinOp(_, left, right):
            return free_vars(left) | free_vars(right)
        case Call("integrate", (f, Var(t), a, b)):
            return (free_vars(f) - {t}) | free_vars(a) | free_vars(b)
        case Call("diff", (f, Var())):
            return free_vars(f)
        case Call(_, args):
            return frozenset().union(*(free_vars(a) for a in args))
    raise TypeError(f"Not an expression: {expr!r}")


def substitute(expr: Expr, var: str, value: Expr) -> Expr:
    """Replace every free occurrence of `var` with `value`."""
    match expr:
        case Num():
            return expr
        case Var(name):
            return value if name == var else expr
        case Neg(operand):
            return Neg(substitute(operand, var, value))
        case BinOp(op, left, right):
            return BinOp(op, substitute(left, var, value), substitute(right, var, value))
        case Call("integrate", (f, Var(t) as tv, a, b)):
            inner = f if t == var else substitute(f, var, value)
            return Call("integrate", (inner, tv, substitute(a, var, value), substitute(b, var, value)))
        case Call(name, args):
            return Call(name, tuple(substitute(a, var, value) for a in args))
    raise TypeError(f"Not an expression: {expr!r}")


# ---- printing -------------------------------------------------------------

_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 4}
_NEG_PRECEDENCE = 3  # unary minus binds tighter than * but looser than ^


def to_string(expr: Expr) -> str:
    """Render an expression as text that `parse` reads back to the same tree."""
    return _to_string(expr, 0)


def _format_number(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def _to_string(expr: Expr, parent_prec: int) -> str:
    match expr:
        case Num(value):
            text = _format_number(value)
            # a negative literal behaves like unary minus
            return f"({text})" if value < 0 and parent_prec >= _NEG_PRECEDENCE else text
        case Var(name):
            return name
        case Neg(operand):
            text = "-" + _to_string(operand, _NEG_PRECEDENCE)
            return f"({text})" if parent_prec >= _NEG_PRECEDENCE else text
        case BinOp(op, left, right):
            prec = _PRECEDENCE[op]
            if op == "^":  # right-associative
                left_s, right_s = _to_string(left, prec + 1), _to_string(right, prec)
            else:  # left-associative
                left_s, right_s = _to_string(left, prec), _to_string(right, prec + 1)
            text = f"{left_s} {op} {right_s}" if prec == 1 else f"{left_s}{op}{right_s}"
            return f"({text})" if prec < parent_prec else text
        case Call(name, args):
            return f"{name}({', '.join(_to_string(a, 0) for a in args)})"
    raise TypeError(f"Not an expression: {expr!r}")
