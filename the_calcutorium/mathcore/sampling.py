"""Equation → points in 3D space. Pure: same arguments, same arrays.

Points that are undefined (sqrt(-1), 1/0) come back as nan, so callers can
decide how to break the curve or surface there.
"""

import numpy as np

from .equation import AXES, Equation

Range = tuple[float, float]


def sample_curve(eq: Equation, domain: Range, n: int = 500, along: str | None = None) -> np.ndarray:
    """Sample a 0- or 1-input equation. Returns shape (n, 3): rows of (x, y, z).

    A constant like "y = 3" has no input, so `along` says which axis to draw it
    along. By default that's the first axis that isn't the output (x for "y = 3").
    """
    if len(eq.inputs) > 1:
        raise ValueError(f"'{eq.text}' is a surface, not a curve")
    if eq.inputs and along not in (None, eq.inputs[0]):
        raise ValueError(f"'{eq.text}' varies along {eq.inputs[0]}, not {along}")
    t = np.linspace(domain[0], domain[1], n)
    input_axis = eq.inputs[0] if eq.inputs else along or next(a for a in AXES if a != eq.output)
    values = eq(t) if eq.inputs else eq()
    return _place({input_axis: t, eq.output: np.broadcast_to(values, t.shape)}, t.shape)


def sample_surface(eq: Equation, range1: Range, range2: Range, n: int = 100) -> np.ndarray:
    """Sample a 2-input equation on an n×n grid. Returns shape (n, n, 3).

    range1 is for eq.inputs[0], range2 for eq.inputs[1] (inputs are sorted).
    """
    if len(eq.inputs) != 2:
        raise ValueError(f"'{eq.text}' is not a surface")
    u, v = np.meshgrid(np.linspace(*range1, n), np.linspace(*range2, n))
    a, b = eq.inputs
    return _place({a: u, b: v, eq.output: eq(u, v)}, u.shape)


def _place(coords: dict[str, np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    """Stack named coordinates into (..., 3) xyz order, filling missing axes with 0."""
    zeros = np.zeros(shape)
    return np.stack([coords.get(axis, zeros) for axis in AXES], axis=-1)
