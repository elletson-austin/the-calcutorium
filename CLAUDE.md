# The Calcutorium

Personal math workbench (PySide6 + ModernGL + numpy) that grows with my math degree.
**Read DESIGN.md first**: it has the vision, architecture, and rules. This file is the short version.

## Commands

- Run the app: `uv run main.py`
- Run tests: `uv run pytest`
- Run one test file: `uv run pytest tests/test_calculus.py -q`
- Run the UI tests (opens a window, checks pixels, saves screenshots to tests/output/):
  `uv run pytest -m ui`. Run these after any change to `render/` or `ui/`, and look at the
  screenshots: unit tests can't see a plot that silently fails to draw.

## Architecture (dependencies only point down)

```
ui/        Qt widgets. Calls App actions, listens to events.     (Qt, GL via render/)
app/       App: every user action. commands.py: text → actions.   (no Qt)
scene.py   Immutable scene objects + change events.              (no Qt)
render/    meshes.py: scene → vertex arrays (pure numpy).
           renderer.py, simulations.py: the only GPU code.
mathcore/  Pure functions: parse, diff, simplify, compile, sample. (numpy only)
planes.py, events.py: tiny shared helpers.
```

## Rules

- **mathcore is pure.** Same input → same output. No state, no I/O, no Qt/GL imports.
  Every input is an explicit parameter (e.g. `diff(expr, var)`, never guess the variable).
  Expression nodes are frozen dataclasses; operations return new trees.
- **One path per action.** Buttons and typed commands both call `App` methods in
  `app/actions.py`. Never put action logic in a widget or in `commands.py`.
- **Scene objects are immutable.** Change one with `dataclasses.replace` + `scene.replace(old, new)`.
- **Rendering is read-only.** The viewport reads `app.scene` and `app.view`; it never mutates them.
- **Tests first for math.** Any change to `mathcore/` or `render/meshes.py` gets a test.
  `uv run pytest` must pass before calling a task done.
- Keep things explainable: if a change makes the architecture harder to describe, stop and ask.
- Edit files with the Edit/Write tools, not PowerShell string replacement (it has
  corrupted UTF-8 characters like `×` and `→` before).

## Adding things

- New math function: `mathcore/tokenizer.py` (FUNCTIONS), `compiler.py` (numpy impl),
  `calculus.py` (_DERIVATIVES), plus tests.
- New command: handler in `app/commands.py` that calls an `App` method; add to HELP.
- New kind of plot: follow "Adding a kind of plot" in DESIGN.md (Plot subclass in
  `scene.py`, builder registered in `render/meshes.py::BUILDERS`, `add_<kind>` action,
  command, and panel dropdown entry).
- Never remove `glBindVertexArray(0)` in `ui/viewport.py`; `tests/test_ui.py` checks it.
