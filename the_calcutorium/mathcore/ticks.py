"""Round-number steps, for grid lines and contour levels."""

import numpy as np


def nice_spacing(span: float, target_count: int = 10) -> float:
    """A step of 1, 2 or 5 × 10^k that divides `span` into about `target_count` pieces."""
    raw = span / target_count
    power = 10 ** np.floor(np.log10(raw))
    for step, below in ((1, 1.5), (2, 3.5), (5, 7.5)):
        if raw / power < below:
            return float(step * power)
    return float(10 * power)
