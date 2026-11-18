"""Every action the user can take. GUI buttons and typed commands both end up here.

No Qt in this file: the App reports results through events, so it can be driven
and tested from plain Python.
"""

from dataclasses import fields, replace

from ..events import Event
from ..mathcore import (
    MathError,
    Num,
    compile_expr,
    expand_calculus,
    free_vars,
    parse,
    parse_equation,
    simplify,
    to_string,
)
from ..scene import LorenzAttractor, NBody, Plot, Scene
from .view import PLANES, ViewState

PALETTE = [
    (0.36, 0.62, 0.98),  # blue
    (0.98, 0.45, 0.38),  # red
    (0.40, 0.85, 0.50),  # green
    (0.98, 0.78, 0.30),  # yellow
    (0.75, 0.50, 0.98),  # purple
    (0.30, 0.85, 0.85),  # cyan
]

SIMULATIONS = {"lorenz": LorenzAttractor, "nbody": NBody}
_SIMULATION_LABELS = {"lorenz": "Lorenz attractor", "nbody": "N-body simulation"}


class App:
    def __init__(self) -> None:
        self.scene = Scene()
        self.view = ViewState()
        self.view_changed = Event()  # (ViewState)
        self.message = Event()       # (level: "info" | "error", text)

    # ---- messages -------------------------------------------------------

    def info(self, text: str) -> None:
        self.message.emit("info", text)

    def error(self, text: str) -> None:
        self.message.emit("error", text)

    # ---- plots ----------------------------------------------------------

    def add_plot(self, text: str) -> Plot | None:
        if any(p.text == text.strip() for p in self.scene.plots()):
            self.info(f"'{text.strip()}' is already plotted.")
            return None
        try:
            equation = parse_equation(text)
        except MathError as e:
            self.error(f"Can't plot '{text}': {e}")
            return None
        plot = Plot(equation, self._next_color())
        self.scene.add(plot)
        self.info(f"Plotted {plot.text}")
        return plot

    def edit_plot(self, plot: Plot, text: str) -> Plot | None:
        """Replace a plot's equation. Returns the new plot, or None if the text is invalid."""
        try:
            equation = parse_equation(text)
        except MathError as e:
            self.error(f"Can't plot '{text}': {e}")
            return None
        new_plot = replace(plot, equation=equation)
        self.scene.replace(plot, new_plot)
        self.info(f"Changed {plot.text} → {new_plot.text}")
        return new_plot

    def remove_plot(self, plot: Plot) -> None:
        self.scene.remove(plot)
        self.info(f"Removed {plot.text}")

    def remove_plot_by_text(self, text: str) -> None:
        plot = next((p for p in self.scene.plots() if p.text == text.strip()), None)
        if plot is None:
            self.error(f"No plot '{text.strip()}'")
        else:
            self.remove_plot(plot)

    def clear(self) -> None:
        for obj in self.scene.objects:
            self.scene.remove(obj)
        self.info("Scene cleared.")

    def _next_color(self):
        used = {p.color for p in self.scene.plots()}
        free = [c for c in PALETTE if c not in used]
        return free[0] if free else PALETTE[len(self.scene.plots()) % len(PALETTE)]

    # ---- math -----------------------------------------------------------

    def show(self, text: str) -> str | None:
        """Work out the calculus in an expression and print it.

        show diff(x^2*y, x)          → 2*x*y
        show integrate(t^2, t, 0, 3) → 9    (no variables left, so it's evaluated)
        """
        try:
            expr = simplify(expand_calculus(parse(text)))
            result = to_string(expr)
            if not free_vars(expr) and not isinstance(expr, Num):
                result = f"{float(compile_expr(expr, ())()):.10g}"
        except MathError as e:
            self.error(str(e))
            return None
        self.info(f"{text.strip()} = {result}")
        return result

    # ---- simulations ----------------------------------------------------

    def add_simulation(self, name: str) -> None:
        kind = SIMULATIONS[name]
        if self.scene.find(kind) is not None:
            self.info(f"The {_SIMULATION_LABELS[name]} is already running.")
            return
        self.scene.add(kind())
        self.info(f"Added the {_SIMULATION_LABELS[name]}.")

    def remove_simulation(self, name: str) -> None:
        sim = self.scene.find(SIMULATIONS[name])
        if sim is None:
            self.info(f"No {_SIMULATION_LABELS[name]} in the scene.")
            return
        self.scene.remove(sim)
        self.info(f"Removed the {_SIMULATION_LABELS[name]}.")

    def update_simulation(self, name: str, params: dict) -> None:
        kind = SIMULATIONS[name]
        sim = self.scene.find(kind)
        if sim is None:
            self.info(f"No {_SIMULATION_LABELS[name]} in the scene.")
            return
        valid = {f.name for f in fields(kind)}
        unknown = set(params) - valid
        if unknown:
            self.error(f"Unknown parameter(s): {', '.join(sorted(unknown))}")
            return
        self.scene.replace(sim, replace(sim, **params))
        self.info(f"Updated the {_SIMULATION_LABELS[name]}.")

    # ---- view -----------------------------------------------------------

    def set_view(self, mode: str) -> None:
        mode = mode.lower()
        if mode != "3d" and mode not in PLANES:
            self.error(f"Unknown view '{mode}'. Use 3d, xy, xz, or yz.")
            return
        if mode == self.view.mode:
            return
        self._set_view_state(self.view.with_mode(mode))
        self.info("Switched to 3D view." if mode == "3d" else f"Switched to 2D view ({mode.upper()} plane).")

    def set_range(self, axis: str, lo: float, hi: float) -> None:
        if not self.view.is_2d:
            self.error("Ranges only apply in 2D views. Switch with 'view xy' first.")
            return
        if axis not in ("x", "y", "z"):
            self.error(f"Unknown axis '{axis}'. Use x, y, or z.")
            return
        if lo >= hi:
            self.error("The minimum must be less than the maximum.")
            return
        self._set_view_state(self.view.with_range(axis, lo, hi))
        self.info(f"Set {axis} range to [{lo:g}, {hi:g}].")

    def auto_range(self) -> None:
        if self.view.ranges:
            self._set_view_state(self.view.without_ranges())
            self.info("Back to automatic ranges.")

    def _set_view_state(self, view: ViewState) -> None:
        self.view = view
        self.view_changed.emit(view)
