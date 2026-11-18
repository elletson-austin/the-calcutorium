"""Expression tree → fast numpy function.

    f = compile_expr(parse("x^2 + y"), ("x", "y"))
    f(np.array([1, 2]), np.array([10, 10]))   # → array([11., 14.])

The compiled function is pure: the same arrays in give the same array out.
"""

from typing import Callable

import numpy as np

from .calculus import expand_calculus
from .errors import MathError
from .expr import BinOp, Call, Expr, Neg, Num, Var, free_vars

Env = dict[str, np.ndarray]

_UNARY_FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "asinh": np.arcsinh, "acosh": np.arccosh, "atanh": np.arctanh,
    "sec": lambda u: 1.0 / np.cos(u),
    "csc": lambda u: 1.0 / np.sin(u),
    "cot": lambda u: np.cos(u) / np.sin(u),
    "exp": np.exp, "log": np.log, "ln": np.log,
    "sqrt": np.sqrt, "cbrt": np.cbrt, "abs": np.abs,
    "floor": np.floor, "ceil": np.ceil, "sign": np.sign,
}
_BINARY_FUNCTIONS = {"min": np.minimum, "max": np.maximum}
_OPERATORS = {
    "+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide, "^": np.power,
}

# Gauss–Legendre quadrature: 64 nodes on [-1, 1]. Exact for polynomials up to degree 127.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(64)


def compile_expr(expr: Expr, inputs: tuple[str, ...]) -> Callable[..., np.ndarray]:
    """Compile `expr` into f(*arrays), with arguments in the order of `inputs`.

    The result always has the broadcast shape of the inputs, even for constants,
    and undefined points (log(-1), 1/0, ...) come out as nan/inf instead of raising.
    """
    expr = expand_calculus(expr)
    unknown = free_vars(expr) - set(inputs)
    if unknown:
        raise MathError(f"Unknown variable(s): {', '.join(sorted(unknown))}")
    body = _compile(expr)

    def evaluate(*args) -> np.ndarray:
        if len(args) != len(inputs):
            raise TypeError(f"Expected {len(inputs)} argument(s) {inputs}, got {len(args)}")
        arrays = [np.asarray(a, dtype=np.float64) for a in args]
        shape = np.broadcast_shapes(*(a.shape for a in arrays))
        with np.errstate(all="ignore"):
            result = body(dict(zip(inputs, arrays)))
        return np.array(np.broadcast_to(result, shape), dtype=np.float64)

    return evaluate


def _compile(expr: Expr) -> Callable[[Env], np.ndarray]:
    match expr:
        case Num(value):
            return lambda env: np.float64(value)
        case Var(name):
            return lambda env: env[name]
        case Neg(operand):
            f = _compile(operand)
            return lambda env: -f(env)
        case BinOp(op, left, right):
            ufunc, f, g = _OPERATORS[op], _compile(left), _compile(right)
            return lambda env: ufunc(f(env), g(env))
        case Call("integrate", (integrand, Var(t), lower, upper)):
            return _compile_integral(integrand, t, lower, upper)
        case Call(name, (arg,)) if name in _UNARY_FUNCTIONS:
            func, f = _UNARY_FUNCTIONS[name], _compile(arg)
            return lambda env: func(f(env))
        case Call(name, (arg1, arg2)) if name in _BINARY_FUNCTIONS:
            func, f, g = _BINARY_FUNCTIONS[name], _compile(arg1), _compile(arg2)
            return lambda env: func(f(env), g(env))
        case Call(name, args):
            raise MathError(f"{name}() can't take {len(args)} argument(s)")
    raise TypeError(f"Not an expression: {expr!r}")


def _compile_integral(integrand: Expr, t: str, lower: Expr, upper: Expr) -> Callable[[Env], np.ndarray]:
    """∫_lower^upper integrand dt, by Gauss–Legendre quadrature.

    Works point-by-point over arrays: every other variable gets an extra trailing
    axis holding the quadrature nodes, and we sum along that axis. Because of that,
    integrals can nest (double integrals) and bounds can depend on other variables.
    """
    f, a, b = _compile(integrand), _compile(lower), _compile(upper)

    def integral(env: Env) -> np.ndarray:
        lo = np.asarray(a(env))[..., None]
        hi = np.asarray(b(env))[..., None]
        half_width, midpoint = (hi - lo) / 2, (hi + lo) / 2

        inner_env = {name: np.asarray(value)[..., None] for name, value in env.items()}
        inner_env[t] = midpoint + half_width * _GL_NODES
        values = f(inner_env)
        return np.sum(values * _GL_WEIGHTS, axis=-1) * half_width[..., 0]

    return integral
