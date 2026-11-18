"""Plot equations: "z = sin(x)*y" → which axis is the output, which are inputs.

Rules:
    - Plot variables are the axes x, y, z.
    - "out = expr" names the output axis explicitly.
    - With no "=", the output is inferred:
        0 or 1 input  → y  (or x if the input is y: "y^2" means x = y^2)
        2 inputs      → the remaining axis ("x*y" means z = x*y)
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .compiler import compile_expr
from .errors import MathError
from .expr import Expr, free_vars
from .parser import parse

AXES = ("x", "y", "z")  # same as planes.AXES; repeated so mathcore stays dependency-free


@dataclass(frozen=True)
class Equation:
    text: str                  # what the user typed
    output: str                # the axis the value goes on
    inputs: tuple[str, ...]    # sorted input axes; 1 = curve, 2 = surface
    expr: Expr                 # right-hand side
    fn: Callable[..., np.ndarray] = field(compare=False, repr=False)

    def __call__(self, *args) -> np.ndarray:
        return self.fn(*args)


def parse_equation(text: str) -> Equation:
    text = text.strip()
    if text.count("=") > 1:
        raise MathError("An equation can have at most one '='")

    lhs, _, rhs = text.rpartition("=")
    lhs = lhs.strip() or None
    expr = parse(rhs)
    inputs = tuple(sorted(free_vars(expr)))

    bad = [v for v in inputs if v not in AXES]
    if bad:
        raise MathError(f"Unknown variable(s) {', '.join(bad)}: plots use x, y, z")
    if len(inputs) > 2:
        raise MathError("A plot can have at most 2 input variables")

    output = lhs if lhs is not None else _infer_output(inputs)
    if output not in AXES:
        raise MathError(f"Left side must be x, y, or z, not '{output}'")
    if output in inputs:
        raise MathError(f"'{output}' can't appear on both sides")

    return Equation(text, output, inputs, expr, compile_expr(expr, inputs))


def _infer_output(inputs: tuple[str, ...]) -> str:
    if len(inputs) == 2:
        return next(a for a in AXES if a not in inputs)
    return "x" if inputs == ("y",) else "y"
