"""The pure math core. No Qt, no OpenGL, no state: every function maps input → output.

    text ──parse──▶ Expr ──diff/simplify──▶ Expr ──compile_expr──▶ numpy function
    text ──parse_equation──▶ Equation ──sample_curve/sample_surface──▶ points
                                       ──sample_contours──▶ level-curve segments
"""

from .calculus import diff, expand_calculus, simplify
from .compiler import compile_expr
from .contours import auto_levels, marching_squares, sample_contours
from .equation import AXES, Equation, parse_equation
from .errors import MathError, ParseError
from .expr import BinOp, Call, Expr, Neg, Num, Var, free_vars, substitute, to_string
from .parser import parse
from .sampling import sample_curve, sample_surface
from .ticks import nice_spacing

__all__ = [
    "AXES", "BinOp", "Call", "Equation", "Expr", "MathError", "Neg", "Num", "ParseError", "Var",
    "auto_levels", "compile_expr", "diff", "expand_calculus", "free_vars", "marching_squares",
    "nice_spacing", "parse", "parse_equation", "sample_contours", "sample_curve", "sample_surface",
    "simplify", "substitute", "to_string",
]
