"""View state: which view is active and any manually-set axis ranges.

Immutable, like scene objects: actions build a new ViewState and announce it.
The viewport turns this into a camera; the view state itself knows nothing of GL.
"""

from dataclasses import dataclass, field, replace

from ..planes import PLANE_AXES, PLANES


@dataclass(frozen=True)
class ViewState:
    mode: str = "3d"   # "3d" or one of PLANES
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)  # manual ranges, 2D only

    @property
    def is_2d(self) -> bool:
        return self.mode in PLANES

    @property
    def plane_axes(self) -> tuple[str, str, str] | None:
        return PLANE_AXES.get(self.mode)

    @property
    def fixed_ranges(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """(h_range, v_range) if both in-plane axes have manual ranges, else None."""
        axes = self.plane_axes
        if axes is None or axes[0] not in self.ranges or axes[1] not in self.ranges:
            return None
        return self.ranges[axes[0]], self.ranges[axes[1]]

    def with_mode(self, mode: str) -> "ViewState":
        return replace(self, mode=mode)

    def with_range(self, axis: str, lo: float, hi: float) -> "ViewState":
        return replace(self, ranges={**self.ranges, axis: (lo, hi)})

    def without_ranges(self) -> "ViewState":
        return replace(self, ranges={})
