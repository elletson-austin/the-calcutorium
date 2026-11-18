"""Typed commands → App actions. This file only translates text; the work happens in actions.py."""

from typing import Callable

from .actions import SIMULATIONS, App

HELP = """Commands:
  plot <equation>          plot a curve or surface: plot z = sin(x)*cos(y)
  <equation>               anything that isn't a command is plotted: y = x^2
  remove <equation>        remove a plot (or: remove lorenz, remove nbody)
  clear                    remove everything
  show <expression>        work out calculus: show diff(x^2*y, x)
  add lorenz | nbody       start a simulation
  view 3d | xy | xz | yz   switch views
  range <axis> <min> <max> fix an axis range in 2D (range auto to undo)
  help                     this message

Math: + - * / ^, implicit multiplication (2x, xy), |x|, pi, e,
  sin cos tan asin acos atan sinh cosh tanh sec csc cot exp ln log sqrt cbrt abs
  floor ceil sign min max
  diff(f, x)               partial derivative ∂f/∂x  (diff(f) if f has one variable)
  integrate(f, t, a, b)    ∫ from a to b of f dt  (b can be x: an antiderivative)"""


def run_command(app: App, line: str) -> None:
    line = line.strip()
    if not line:
        return
    name, _, rest = line.partition(" ")
    handler = _COMMANDS.get(name.lower())
    if handler is None:
        app.add_plot(line)  # not a command: treat it as an equation
    else:
        handler(app, rest.strip())


def _help(app: App, rest: str) -> None:
    app.info(HELP)


def _plot(app: App, rest: str) -> None:
    if not rest:
        app.error("Usage: plot <equation>")
    else:
        app.add_plot(rest)


def _remove(app: App, rest: str) -> None:
    if rest.lower() in SIMULATIONS:
        app.remove_simulation(rest.lower())
    elif rest:
        app.remove_plot_by_text(rest)
    else:
        app.error("Usage: remove <equation> | lorenz | nbody")


def _clear(app: App, rest: str) -> None:
    app.clear()


def _show(app: App, rest: str) -> None:
    if not rest:
        app.error("Usage: show <expression>")
    else:
        app.show(rest)


def _add(app: App, rest: str) -> None:
    if rest.lower() in SIMULATIONS:
        app.add_simulation(rest.lower())
    else:
        app.error(f"Usage: add {' | '.join(SIMULATIONS)}   (to plot, use: plot <equation>)")


def _view(app: App, rest: str) -> None:
    words = rest.lower().split()
    if words[:1] == ["2d"]:  # "view 2d xy" also works
        words = words[1:] or ["xy"]
    if len(words) != 1:
        app.error("Usage: view 3d | xy | xz | yz")
    else:
        app.set_view(words[0])


def _range(app: App, rest: str) -> None:
    words = rest.lower().split()
    if words == ["auto"]:
        app.auto_range()
        return
    try:
        axis, lo, hi = words[0], float(words[1]), float(words[2])
        if len(words) != 3:
            raise ValueError
    except (IndexError, ValueError):
        app.error("Usage: range <x|y|z> <min> <max>   or   range auto")
        return
    app.set_range(axis, lo, hi)


_COMMANDS: dict[str, Callable[[App, str], None]] = {
    "help": _help,
    "plot": _plot,
    "remove": _remove,
    "clear": _clear,
    "show": _show,
    "add": _add,
    "view": _view,
    "range": _range,
}
