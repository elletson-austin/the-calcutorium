"""The pure math core. No Qt, no OpenGL, no state: every function maps input → output.

    text ──parse──▶ Expr ──diff/simplify──▶ Expr ──compile_expr──▶ numpy function
    text ──parse_equation──▶ Equation ──sample_curve/sample_surface──▶ points
"""

from .calculus import diff, expand_calculus, simplify
from .compiler import compile_expr
from .equation import AXES, Equation, parse_equation
from .errors import MathError, ParseError
from .expr import BinOp, Call, Expr, Neg, Num, Var, free_vars, substitute, to_string
from .parser import parse
from .sampling import sample_curve, sample_surface

__all__ = [
    "AXES", "BinOp", "Call", "Equation", "Expr", "MathError", "Neg", "Num", "ParseError", "Var",
    "compile_expr", "diff", "expand_calculus", "free_vars", "parse", "parse_equation",
    "sample_curve", "sample_surface", "simplify", "substitute", "to_string",
]
