# Architecture roadmap

The plan for making The Calcutorium last for years. DESIGN.md says what the
architecture *is*; this file says how it *changes*, in what order, and when each step
counts as done. Todos.md stays the list of features; the
[last section](#where-the-feature-todos-fit) says which feature todos wait on which step.

## How to use this file

- **Do the steps in order.** Each step depends on the ones before it (the table below
  says exactly how). Don't start a step until the previous one's "Done when" list is all
  checked.
- **One sub-step per commit.** Every sub-step (1a, 1b, …) ends with `uv run pytest`
  green, `uv run pytest -m ui` green, and (from Step 0 on) `uv run ruff check .` and
  `uv run pyright` clean. A session can cover one or several sub-steps, but never stop
  halfway through one.
- **No behavior changes unless a sub-step says so.** Refactor sub-steps keep every
  existing test passing *with unchanged assertions*. Only the way tests reach the code
  changes (e.g. `app.scene.plots()` → `app.doc.plots()`). Behavior changes are marked
  **(behavior change)**.
- **At the end of each step:** update DESIGN.md (the architecture diagram, "The life
  of an action", and anything else the step changed), tick the boxes here, and move the
  step's DESIGN.md "Next up" bullet into "History".
- **Decisions are already made.** Each step has a "Decisions" list with the choice
  and the reason. Where a decision is really yours to make, it's marked
  **(your call)** and gives a default; the default happens unless you say otherwise
  when that step starts.

## Order at a glance

| Step | What | Needs | Why this position |
|---|---|---|---|
| 0 | [Guardrails](#step-0-guardrails): lint, types, property tests, registry tests | none | Cheap, and it protects every refactor after it. Property tests must exist *before* Step 4 changes the parser. |
| 1 | [The document is one immutable value](#step-1-the-document-is-one-immutable-value) | 0 | The foundation: stable ids, pure actions, undo. Saving, references between objects, and definitions all build on it. |
| 2 | [Save and open as command scripts](#step-2-save-and-open-as-command-scripts) | 1 | `to_script(doc)` is a pure function of one value, and loading is a pure fold over lines. Both need Step 1. |
| 3 | [Frame plan and cameras as data](#step-3-frame-plan-and-cameras-as-data) | 1 | Fixes the viewport's side effects and the camera carry-over bug *before* Step 4 adds new kinds that must draw through it. |
| 4 | [Definitions and variables beyond x, y, z](#step-4-definitions-and-variables-beyond-x-y-z) | 0, 1, 2, 3 | The big math change. Needs ids (1), must save (2), and new kinds draw through the frame plan (3). Finish before next semester. |
| 5 | [ODE systems in mathcore](#step-5-ode-systems-in-mathcore-simulations-become-scene-objects) | 4 | Systems are written with definitions (`sigma`, `rho`). Start it when the ODE course starts. |

(This puts saving *after* the document refactor, not first as first suggested: written
against one `Document` value, `to_script` is a single pure function. Written before it,
it would have to be rewritten in Step 1.)

---

## Step 0: Guardrails

**Goal:** tools that catch mistakes before you have to, so the later refactors are safe.

### 0a. Linting with ruff

- Add `ruff` to `[dependency-groups] dev` in `pyproject.toml`.
- Configure in `pyproject.toml`:
  ```toml
  [tool.ruff]
  line-length = 120
  target-version = "py314"

  [tool.ruff.lint]
  select = ["E", "F", "W", "I", "B", "UP"]
  ```
- Fix every finding. Where a finding is wrong for this codebase, add a targeted
  `# noqa: <code>` with a reason; don't switch rules off globally.
- **Decision:** lint only, no `ruff format`. The code has hand-aligned comments and
  tables that a formatter would destroy.

### 0b. Type checking with pyright

- Add `pyright` to the dev group. Configure:
  ```toml
  [tool.pyright]
  include = ["the_calcutorium", "tests"]
  pythonVersion = "3.14"
  typeCheckingMode = "standard"
  strict = ["the_calcutorium/mathcore", "the_calcutorium/scene.py",
            "the_calcutorium/app", "the_calcutorium/planes.py",
            "the_calcutorium/render/meshes.py", "the_calcutorium/render/readout.py"]
  ```
- **Decision:** strict for the pure layers (where types are the documentation), standard
  for `ui/` and the GPU code (PySide6 and moderngl stubs are incomplete; strict there is
  noise).
- Fix every error. `# type: ignore[<rule>]` only with a comment saying why.

### 0c. Property-based tests for the math core (hypothesis)

- Add `hypothesis` to the dev group. New file `tests/test_properties.py`.
- Register a profile in `tests/conftest.py` so runs are reproducible (the codebase's
  "same input → same output" rule applied to the tests themselves):
  ```python
  settings.register_profile("default", max_examples=200, deadline=None, derandomize=True)
  settings.load_profile("default")
  ```
- A strategy `exprs(variables=("x", "y"), depth=4)` that builds random `Expr` trees from:
  `Num` (integers −5..5 and halves), `Var`, `Neg`, `BinOp` with `+ - *`, `^` with
  exponent `Num` in 0..4 only, and `Call` with `sin cos exp`. **Decision:** only smooth,
  everywhere-defined building blocks, so the properties below hold everywhere and the
  tests are never flaky. (`/`, `log`, `sqrt` are covered by the existing example tests.)
- Evaluate at points drawn from −2..2 (floats, no nan/inf). Helper:
  `close(a, b)` = `abs(a - b) <= 1e-6 * max(1, abs(a), abs(b))`; skip the example with
  `assume(...)` if either side is non-finite or larger than 1e8.
- The properties:
  1. **Printing round-trips:** `parse(to_string(e))` evaluates the same as `e`. (Not
     structural equality: the printer may parenthesize or regroup.)
  2. **Simplify keeps the value:** `simplify(e)` evaluates the same as `e`.
  3. **Derivatives match finite differences:** `diff(e, "x")` at a point ≈
     `(e(x+h) − e(x−h)) / 2h` with `h = 1e-4`, tolerance `1e-3 * max(1, |value|)`.
  4. **Mixed partials commute (Clairaut):** `diff(diff(e, "x"), "y")` evaluates the
     same as `diff(diff(e, "y"), "x")`.
  5. **diff of a sum:** `diff(a + b, x)` evaluates the same as `diff(a, x) + diff(b, x)`.
- If a property finds a real bug, fix the bug (with a regular example test added to the
  right test file) in its own commit before continuing.

### 0d. Every per-kind table is checked by a test

Today `test_every_plot_kind_has_a_builder` guards `BUILDERS`, but other places that
must list every plot kind have no guard. In particular, `render/readout.py` decides by
an `isinstance` chain in `_plot_line`, so a new kind would be silently skipped.

- In `render/readout.py`, replace the `isinstance` chain with a table:
  ```python
  READERS: dict[type[Plot], Callable[[Plot, str, dict[str, float]], ReadoutLine | None]] = {
      Curve: _curve_line, Surface: _function_line, Contour: _function_line,
  }
  ```
- Add one test per table, all in `tests/test_registries.py`, using a shared helper
  `plot_kinds()` = every concrete subclass of `Plot`, found recursively (not just
  `Plot.__subclasses__()`, which misses grandchildren):
  - `set(BUILDERS) == plot_kinds()`
  - `set(READERS) == plot_kinds()`
  - every kind can be added by a typed command: `{k.kind for k in plot_kinds()}` ⊆
    the kinds `commands.py` knows (`_PLOT_KINDS`)
- **Rule from now on:** any new dict keyed by plot kind gets a line in this test file in
  the same commit.

### 0e. One command for "everything is fine"

- In CLAUDE.md's Commands section, add:
  `uv run ruff check . && uv run pyright && uv run pytest` (all checks) and say that a
  task isn't done until this passes, plus `uv run pytest -m ui` for `render/` or `ui/`
  changes.
- **Decision:** no pre-commit framework (one more tool to understand). Running the
  command is part of finishing a task, as it already is for pytest.

### Done when

- [ ] `ruff check .` and `pyright` report zero problems
- [ ] `tests/test_properties.py` has the five properties, and they pass deterministically
- [ ] `tests/test_registries.py` covers BUILDERS, READERS, and commands
- [ ] CLAUDE.md lists the all-checks command

---

## Step 1: The document is one immutable value

**Goal:** everything the user has made (plots, simulations, view mode, ranges) is one
frozen `Document`. Every action is a pure function `(Document, args) → Result`. The
`App` becomes a thin shell that holds the current document, emits one `changed` event,
and keeps undo history.

### Today

- `scene.py`: `Scene` is a mutable list with three events (`added`, `removed`,
  `replaced`). Plots are `eq=False`, so they're told apart by Python object identity.
- `app/actions.py`: `App` holds `self.scene` and `self.view` separately; view changes
  go through a different event (`view_changed`). Actions mutate and emit messages as
  they go.
- `ui/plot_list.py` keys its rows by plot *object* and re-keys them on `replaced`.
- `ui/viewport.py` keys GPU meshes by plot object and simulations by *type*.
- `ViewState.ranges` is a `dict` inside a frozen dataclass: the field can't be
  reassigned, but the dict itself could be mutated.

### Target design

New top-level module `the_calcutorium/document.py` (no Qt, imports `scene` and
`app/view` types, imported by `app/` and `render/`):

```python
ObjectId = int

@dataclass(frozen=True)
class Entry:
    id: ObjectId
    obj: SceneObject          # a Plot or a LorenzAttractor: unchanged classes

@dataclass(frozen=True)
class Document:
    entries: tuple[Entry, ...] = ()     # display order
    view: ViewState = ViewState()
    next_id: ObjectId = 1

    def get(self, id: ObjectId) -> SceneObject              # KeyError if missing
    def plots(self) -> tuple[Plot, ...]
    def plot_entries(self) -> tuple[Entry, ...]
    def find(self, kind: type) -> Entry | None
    def add(self, obj: SceneObject) -> tuple["Document", ObjectId]
    def replace(self, id: ObjectId, obj: SceneObject) -> "Document"   # keeps id and position
    def remove(self, id: ObjectId) -> "Document"
    def with_view(self, view: ViewState) -> "Document"

@dataclass(frozen=True)
class Changes:
    added: tuple[ObjectId, ...]
    removed: tuple[ObjectId, ...]
    replaced: tuple[ObjectId, ...]
    view_changed: bool

def diff_documents(old: Document, new: Document) -> Changes
```

Pure actions in `app/actions.py` (module-level functions, no class):

```python
Message = tuple[Literal["info", "error"], str]

@dataclass(frozen=True)
class Result:
    doc: Document
    messages: tuple[Message, ...] = ()
    value: object = None      # e.g. the new ObjectId, or show()'s result text

def add_plot(doc: Document, text: str) -> Result
def add_contour(doc: Document, text: str, interval: float | None) -> Result
def set_contour_interval(doc: Document, id: ObjectId, interval: float | None) -> Result
def edit_plot(doc: Document, id: ObjectId, text: str) -> Result
def remove(doc: Document, id: ObjectId) -> Result
def remove_by_text(doc: Document, text: str, kind: str | None) -> Result
def clear(doc: Document) -> Result
def show(doc: Document, text: str) -> Result        # doc unchanged; value = result text
def add_simulation(doc: Document, name: str) -> Result
def remove_simulation(doc: Document, name: str) -> Result
def update_simulation(doc: Document, name: str, params: dict[str, float]) -> Result
def set_view(doc: Document, mode: str) -> Result
def set_range(doc: Document, axis: str, lo: float, hi: float) -> Result
def auto_range(doc: Document) -> Result
```

The shell, in a new file `app/shell.py`:

```python
class App:
    doc: Document
    changed: Event      # (old: Document, new: Document)
    message: Event      # (level, text): unchanged

    def apply(self, action, *args) -> object:
        """Run a pure action, commit its document, emit its messages, return its value."""
    def undo(self) -> None
    def redo(self) -> None

    # One-line wrappers so widgets keep reading naturally:
    def add_plot(self, text: str) -> ObjectId | None:
        return self.apply(actions.add_plot, text)
    ...
```

```
widget/command ──▶ App.apply(action, args) ──▶ action(doc, args) → Result   (pure)
                         │
                         ├─ history.append(old doc); doc = result.doc
                         ├─ changed.emit(old, new)  ──▶ widgets call diff_documents
                         └─ message.emit(...) for each message
```

### Sub-steps

- [ ] **1a. Deeply immutable `ViewState`.** `ranges` becomes
  `tuple[tuple[str, float, float], ...]` sorted by axis name. Add
  `ViewState.range_of(axis) -> tuple[float, float] | None`; update `fixed_ranges` and
  `with_range`. The one assertion that changes shape is `test_range_command`'s
  `app.view.ranges == {"x": (-5.0, 5.0)}`, which becomes
  `app.view.range_of("x") == (-5.0, 5.0)`.
- [ ] **1b. `document.py` with `Entry`, `Document`, `Changes`, `diff_documents`.** Not
  used yet. Tests (`tests/test_document.py`):
  - ids are never reused (add, remove, add → a new id)
  - `replace` keeps the id and the position
  - every operation leaves the old document unchanged
  - `diff_documents` reports each kind of change, and reports nothing for identical docs
- [ ] **1c. Plots become plain values.** Remove `eq=False` from `Plot` and its
  subclasses and from the simulation classes. Identity now comes from `Entry.id`, so two
  equal plots are simply equal. Check that `Equation` still hashes (its `fn` field is
  `compare=False`, so hashing uses text/output/inputs/expr).
- [ ] **1d. Pure actions.** Move every action out of the class into module-level
  functions as listed above. `_next_color(doc)` becomes pure. The duplicate check, the
  error messages, and the message texts stay *exactly* the same. Add `app/shell.py`
  with `App` (wrappers + `apply`, no undo yet). `App` no longer has `scene` or
  `view_changed`. Add `App.doc` and keep `App.view` as a read-only property
  (`self.doc.view`) to keep the diff small. Delete the `Scene` class from `scene.py`
  (it keeps the plot classes).
  - Wrappers that took a plot object take an id: `edit_plot(id, text)`,
    `remove_plot(id)`, `set_contour_interval(id, interval)`.
  - Tests: rename `tests/test_app.py` → `tests/test_actions.py`, rewritten to call the pure
    functions on a `Document`. The assertions stay the same, adapted to ids. Keep a
    small new `tests/test_shell.py` for the event/message behavior
    (`test_scene_events` becomes a `changed`-event test).
- [ ] **1e. Widgets use `changed` and ids.** `PlotList` subscribes to `app.changed`,
  calls `diff_documents`, and keys rows by id. A `replaced` id updates the row in place.
  `PlotRow` stores `self.id` and reads its plot from `app.doc.get(self.id)`. The
  viewport subscribes to `changed` and reacts when `view_changed` is true.
- [ ] **1f. GPU caches keyed by id.** In the viewport, mesh key = `entry.id`, signature
  = `(entry.obj, plane, ranges)`: a changed plot is a different value, so it rebuilds.
  Simulations: key = `entry.id`. This removes "keyed by type" in `_draw_scene`.
- [ ] **1g. Undo and redo (behavior change).** In `App`: `history: list[Document]`,
  `future: list[Document]`, max 200 entries (oldest dropped). Rules:
  - `apply` pushes the old doc only if `result.doc != old` (so `show` or a rejected
    edit doesn't create an undo step), and clears `future`.
  - `undo` pops history onto future and emits `changed`; `redo` does the reverse.
    Nothing to undo → info message "Nothing to undo."
  - Typed commands `undo` and `redo`; shortcuts Ctrl+Z and Ctrl+Y (Ctrl+Shift+Z also
    redoes), as `QAction`s on the main window. When a text box (command bar, plot row)
    has focus, Qt gives it Ctrl+Z first, and that's intended: it undoes typing. App
    undo works whenever focus is anywhere else, e.g. after clicking the viewport.
  - View mode and ranges are in the document, so they are undoable. Camera pan/zoom is
    not in the document (see Step 3), so it is not undoable.
  - Tests: undo/redo round trips, redo cleared by a new action, the 200 limit, no-op
    actions don't create steps.

### Decisions

- **Ids live beside objects (`Entry`), not inside them.** A `Plot` describes math; its
  id is the document's bookkeeping. `dataclasses.replace(plot, …)` never has to worry
  about ids.
- **One `changed(old, new)` event instead of four.** Widgets that need details call
  the pure `diff_documents`. One event is easier to picture, and undo needs nothing
  special: it's just another `changed`.
- **Actions return messages instead of emitting them.** That keeps them pure, and lets
  a whole script load (Step 2) collect its messages.
- **The simulation stays a `LorenzAttractor` object for now.** Step 5 replaces it.

### Done when

- [ ] `grep -rn "eq=False\|class Scene\|view_changed" the_calcutorium` finds nothing
- [ ] `app/actions.py` has no class and no `self`; every function there is
  `(Document, …) → Result`
- [ ] Ctrl+Z / Ctrl+Y and `undo` / `redo` work, with tests
- [ ] DESIGN.md: new diagram row for `document.py`, "The life of an action" rewritten,
  the Key rules updated (no more "scene announces changes")

---

## Step 2: Save and open as command scripts

**Goal:** a saved scene is a text file of the same commands you'd type. Opening one
runs them through the same command code. Files stay readable, diffable, and hand-editable
(homework notebooks in git).

### File format (`.calc`, UTF-8)

```
# The Calcutorium, format 1
plot z = sin(x)*cos(y) color #5c9ffa
contour z = sin(x)*cos(y) every 0.5 color #fa7361
add lorenz
set lorenz rho 20
view xy
range x -5 5
range y -2.5 2.5
```

- One command per line. Blank lines and lines starting with `#` are ignored.
- The first line is exactly `# The Calcutorium, format 1`. Files without it still load
  (hand-written ones); files with a *higher* format number refuse to load with
  "This file was saved by a newer version (format N)."
- `to_script` writes objects in document order, then the view (`view`, then `range`
  lines sorted by axis). Simulations get one `set` line with only the parameters that
  differ from the defaults.
- **Numbers are written with `repr(float)`**, which round-trips exactly: the file
  never loses precision.

### New and changed commands

- `plot <eq> [color #rrggbb]` and `contour <eq> [every <n|auto>] [color #rrggbb]`. The
  color suffix is split off with `rpartition(" color ")`, the same way `every` is today.
  Without it, the next palette color is used, as today. Order when both are present:
  `every` before `color`.
- `set lorenz <param> <value> [<param> <value> …]` → `update_simulation`. Same
  parameter names as the scene classes, same validation as the Simulations panel.
- `save [path]`, `open <path>`, `new`. These are shell commands (they do I/O), handled
  by `App`, not by pure actions. `save` with no path saves to the current file, or
  errors with "Usage: save <path>" if there isn't one. A path without an extension
  gets `.calc`. Relative paths are relative to the current working directory.
- `color` and `every` become reserved words: they can't be used as definition names
  in Step 4.

### Target design

- `app/commands.py` becomes pure: `parse_command(line: str) -> Command`, where
  `Command = DocCommand(action, args) | ShellCommand(name, args)`. Running a
  `DocCommand` is `action(doc, *args) -> Result`. `run_command(app, line)` stays as the
  one entry point the UI calls and just dispatches.
- New `app/script.py`, all pure:
  ```python
  def to_script(doc: Document) -> str
  def load_script(text: str) -> Result   # folds parse_command over the lines, from Document()
  ```
  `load_script` keeps going after a bad line. Each error message is prefixed with
  `line N: `, and the valid lines still load.
- `App.open(path)`: reads the file, `load_script`, then commits the result as **one**
  undo step that replaces the whole document (so opening is undoable). `App.save(path)`
  writes `to_script(self.doc)`. `App.file: Path | None` and `App.saved_doc` track the
  current file and whether there are unsaved changes.

### Sub-steps

- [ ] **2a. Pure command parsing.** `parse_command` + `DocCommand`/`ShellCommand`;
  existing command tests pass unchanged.
- [ ] **2b. `color` suffix and the `set` command** (behavior change), with tests.
- [ ] **2c. `to_script` and `load_script`** with tests:
  - the example file above loads to the expected document
  - **round trip:** for a set of example documents (every plot kind, colors,
    contour intervals, the Lorenz simulation with changed params, each view mode, ranges),
    `load_script(to_script(doc)).doc` equals `doc` ignoring ids and `next_id`
    (compare `doc.plots()`, the simulations, and `doc.view`)
  - a property test (hypothesis): random sequences of valid commands → run them →
    `to_script` → `load_script` → same document (ignoring ids)
  - errors carry line numbers; a newer format refuses to load
- [ ] **2d. Shell commands and the File menu** (behavior change): `save`, `open`,
  `new`; a File menu with New (Ctrl+N), Open… (Ctrl+O), Save (Ctrl+S), Save As…
  (Ctrl+Shift+S) using `QFileDialog` with filter `Calcutorium scenes (*.calc)`. The
  window title becomes `name.calc — The Calcutorium`, with a `*` before the name while
  `doc != saved_doc`. Closing the window, New, or Open with unsaved changes asks
  Save / Discard / Cancel.
- [ ] **2e. Docs.** README gets a "Saving" paragraph with the example file; HELP lists
  the new commands.

### Decisions

- **Not saved:** the camera position and zoom (except as `range` lines), the command
  history, and running simulation state (particles restart from their seed). The file
  records *what you made*, not *where you were looking*.
- **Plot domains are not saved yet**, because they can't be set yet. When the
  "set a plot's domain" todo is done, it must add a command *and* a `to_script` line,
  and extend the round-trip test.
- **Opening replaces the current document** (undoably) and does not merge. Merging two
  scenes is out of scope.

### Done when

- [ ] the round-trip example and property tests pass
- [ ] save → quit → open restores the same plots, colors, intervals, simulations, and view
- [ ] unsaved changes are never lost without a prompt

---

## Step 3: Frame plan and cameras as data

**Goal:** the viewport stops being a mix of drawing, state, and input handling. Cameras
become frozen values changed by pure functions. One pure function decides what goes on
screen, and mesh builders return their own labels. The viewport just executes.

### Today

- `render/camera.py`: `Camera3D` and `Camera2D` are mutable classes; `orbit`, `zoom`,
  `pan`, `move`, `fit` change them in place.
- `ui/viewport.py`:
  - `_on_view_changed` converts one camera into the other. **Bug:** `Camera3D(distance=cam.distance)`
    carries a 2D zoom into 3D, so after `range x -5 5` the 3D view starts far too close.
  - `_build_grid` and `_build_plot` save `_grid_labels` and `_anchors` as a *side effect*
    of building meshes.
  - `_apply_input` changes the camera and calls `app.auto_range()` from inside the
    paint loop.
  - The timer redraws at 60 fps even when nothing changes.

### Target design

`render/camera.py`, all frozen and pure:

```python
@dataclass(frozen=True)
class Orbit3D:
    center: tuple[float, float, float] = (0, 0, 0)
    distance: float = 40.0
    pitch: float = 25.0
    yaw: float = 30.0
    fov: float = 60.0
    projection: Projection = Projection.Perspective

@dataclass(frozen=True)
class Plane2D:
    plane: str
    center: tuple[float, float] = (0, 0)   # (h, v)
    height: float = 50.0                    # visible height in world units

Camera = Orbit3D | Plane2D

def view_matrix(cam: Camera) -> np.ndarray
def projection_matrix(cam: Camera, size: tuple[int, int], ranges=None) -> np.ndarray
def visible_ranges(cam: Plane2D, size) -> tuple[Range, Range]
def fit(cam: Plane2D, h_range: Range, v_range: Range) -> Plane2D
def world_to_screen(cam: Camera, point: Vec3, size) -> tuple[float, float] | None  # None: behind camera
```

New `render/navigation.py`, pure:

```python
@dataclass(frozen=True)
class InputSnapshot:            # what happened since the last frame
    mouse: tuple[float, float]
    drag: tuple[float, float]   # left-button drag delta in pixels
    scroll: float               # wheel steps
    keys: frozenset[str]        # "w" "a" "s" "d" held
    seconds: float              # time since last frame

def navigate(cam: Camera, inp: InputSnapshot, size) -> tuple[Camera, bool]
    """New camera, and whether the user took control of a 2D view (pan/zoom)."""
```

New `render/frame.py`, pure:

```python
@dataclass(frozen=True)
class DrawItem:
    key: Hashable                 # GPU cache key: "grid", "axes", or an ObjectId
    signature: Hashable           # rebuild when this changes
    build: Callable[[], Mesh | None]

def frame_items(doc: Document, cam: Camera, size) -> tuple[DrawItem, ...]
```

Labels travel with meshes. In `render/meshes.py`:

```python
@dataclass(frozen=True)
class Label:
    text: str
    anchor: tuple[float, float, float]   # world position
    color: Color
    align: Literal["above-left", "below", "left"]

@dataclass(frozen=True)
class Mesh:
    program: str
    mode: str
    data: np.ndarray
    labels: tuple[Label, ...] = ()
```

`grid_mesh` returns grid numbers as `Label`s (`GridLabels` is deleted); `curve_mesh`
returns its name label at its last vertex (what `_anchors` did). `Renderer.draw_mesh`
returns the cached `Mesh` so the viewport can collect labels. The overlay draws
`Label`s by projecting each anchor with `world_to_screen`, which works in 3D too.

```
paintGL:
  cam, took_control = navigate(cams[mode], input.snapshot(), size)    # pure
  if took_control and doc.view.ranges: app.auto_range()
  items  = frame_items(app.doc, cam, size)                             # pure
  meshes = [renderer.draw_mesh(i.key, i.signature, i.build) for i in items]
  draw_overlay(labels from meshes, readout(...))
```

### Sub-steps

- [ ] **3a. Frozen cameras.** Rewrite `camera.py` as above; port the camera tests.
  Same numbers, same behavior.
- [ ] **3b. One camera per view mode (fixes the carry-over bug).** The viewport keeps
  `cams: dict[str, Camera]` with a default per mode (`Orbit3D()`, `Plane2D("xy")`, …);
  switching modes just picks the other entry, so it never converts. Switching back to a
  mode restores where you were. Fixed ranges apply with `fit`. Regression test: set
  `range x -5 5` in xy, switch to 3D, assert the 3D camera's distance is the default.
  Remove the `auto_range()` workaround from the UI test fixture if it's no longer
  needed (keep it if the test still needs ranges reset for other reasons, and say so
  in a comment).
- [ ] **3c. `navigate`** in `render/navigation.py`, with tests for orbit, WASD, 2D
  pan, zoom, and `took_control`. The viewport's `_Input` turns into an `InputSnapshot`
  each frame.
- [ ] **3d. Zoom toward the cursor (behavior change).** In 2D, scrolling keeps the
  world point under the mouse fixed on screen. Test: the point under the mouse before
  and after `navigate` is the same, to 1e-9. 3D zoom stays toward the orbit center.
- [ ] **3e. Labels in meshes; `frame_items`.** Add `Label` and `Mesh.labels`, move
  grid and plot labels into the builders, add `frame_items`, make `draw_mesh` return the
  mesh, and delete `_grid_labels`, `_anchors`, `_build_grid`, `_build_plot`. Tests:
  grid labels are major lines only (port the existing test), curve label at the last
  vertex, `frame_items` gives a grid in 2D and axes in 3D, and nothing for a surface
  in 2D.
- [ ] **3f. Redraw only when needed (behavior change).** The timer runs only while a
  simulation exists or a WASD key is held. Otherwise the viewport calls `update()` on
  `app.changed`, on every mouse move/press/release and wheel event (so the hover
  readout keeps following the mouse), and on resize. The UI tests still pass.

### Decisions

- **Cameras are not in the document.** They change every frame while dragging. In the
  document that would flood undo history and dirty the file. Saved, undoable
  view settings are `view` mode and `range`s; the camera is how you're looking at them.
- **The input event plumbing stays in the viewport.** Qt events → `InputSnapshot` is
  inherently Qt code. Everything after the snapshot is pure.

### Done when

- [ ] `grep -n "_grid_labels\|_anchors\|GridLabels" -r the_calcutorium` finds nothing
- [ ] no camera method mutates; `navigate` and `frame_items` are tested
- [ ] the carry-over regression test passes; zoom follows the cursor in 2D
- [ ] the app is idle (no repaints) with no simulation and no input
- [ ] DESIGN.md "Next up": the first pain point (labels as side effects) moves to History

---

## Step 4: Definitions and variables beyond x, y, z

**Goal:** expressions can use named constants and functions you define, constants can be
sliders, and a new kind, parametric curves, uses the parameter `t`. This is what makes
linear algebra, ODEs, and series possible without hacks.

### Today

- `mathcore/equation.py`: `parse_equation` rejects any variable that isn't x, y, z, and
  allows at most 2 inputs.
- `mathcore/tokenizer.py`: letters that aren't a known name (`FUNCTIONS`, `pi`, `e`) are
  split into single-letter variables (`xy` → `x*y`). A multi-letter name like `rho`
  would read as `r*h*o`.
- `compile_expr` raises on unknown variables. Only built-in functions can be called.

### The rules

- **Definitions** are typed like equations whose left side isn't an axis:
  - `a = 2`: a constant (any expression without free variables, e.g. `k = 2*pi`)
  - `f(x) = x^2 + a`, `g(x, y) = x*y`: a function (parameters are names; the
    body may use only its parameters, constants, and other definitions)
- **Which lines are plots:** a left side of `x`, `y`, or `z` (or no `=`) is a plot,
  exactly as today. `f(x) = …` *only defines* f. To draw it, type `y = f(x)`.
- **Names:**
  - Letters only, case-sensitive, at least one letter.
  - Can't be `x y z t`, a built-in function or constant name (`FUNCTIONS`, `pi`, `e`),
    or a reserved command word (`every`, `color`, and every command name in `_COMMANDS`).
  - Multi-letter names are allowed. The tokenizer is given the defined names and
    matches them with the same longest-match rule it already uses for `pi` and `sin`, so
    with `rho` defined, `rho` is one name and `rhox` is `rho*x`.
- **Order doesn't matter:** `b = 2*a` may come before `a = 3`. A cycle is an error that
  names it: "Circular definition: a → b → a".
- **Atomic changes:** a definition change (new value, new body, rename, removal) is
  rejected, with nothing changed, if any existing plot or definition would stop being
  valid. The error names them: "Can't remove a: used by y = a*x, contour z = a*x*y".
- **Changing a definition updates every plot that uses it**, immediately. Changing
  `a` changes the plots, so their meshes rebuild.

### Target design

mathcore, all pure, with every input explicit:

```python
@dataclass(frozen=True)
class Constant:
    name: str
    expr: Expr

@dataclass(frozen=True)
class Function:
    name: str
    params: tuple[str, ...]
    expr: Expr

Definition = Constant | Function
Definitions = tuple[Definition, ...]

def tokenize(text: str, names: frozenset[str] = frozenset(),
             functions: frozenset[str] = frozenset()) -> list[Token]
def parse(text: str, defs: Definitions = ()) -> Expr
def parse_definition(text: str, defs: Definitions) -> Definition
def resolve(expr: Expr, defs: Definitions) -> Expr
    """Substitute constants and inline function calls, recursively, until only axes,
    bound variables and built-ins remain. Raises MathError on cycles, unknown names,
    or wrong argument counts."""
def parse_equation(text: str, defs: Definitions = ()) -> Equation
    """As today, but on resolve(parse(rhs, defs), defs). Equation.text stays what you
    typed; Equation.expr is the resolved expression."""
```

Because `resolve` runs before `compile_expr`, the compiler, `diff`, and `integrate` don't
change: `diff(f(x, y), x)` works because `f` is already inlined.

The document gains `definitions: tuple[Entry, ...]` (ids, like objects) plus the
actions `define(doc, text)`, `edit_definition(doc, id, text)`, and
`remove_definition(doc, id)`. Each re-parses every plot's `equation.text` against the
new definitions (a pure `rebuild_plots(doc) -> Document | MathError`) and rejects the
change if any plot fails.

**Sliders:** a `Constant` whose expression is a plain number has a slider. Its range is
document state, not math, so it lives in `document.py`, not mathcore: each definition
entry's `obj` is a
`DefinitionItem(definition: Definition, slider: Slider | None)`, with
`Slider(lo, hi, step)`. The default is `lo = -10`, `hi = 10`, widened to include the
value; `step = (hi - lo) / 100`. The command
`slider a 0 5 [step 0.1]` sets it. Dragging a slider calls `define` repeatedly. These
changes *coalesce* into one undo step: `App.apply(..., coalesce=("slider", id))`
merges into the previous step if it had the same coalesce key.

**Parametric curves:** a new kind `ParametricCurve(Plot)` with `components:
tuple[Expr, ...]` (2 or 3) and `t_range: Range`.
- Typed as `(cos(t), sin(t), t/4)` or `(cos(t), sin(t))`, with an optional
  ` for t from 0 to 2*pi`.
- **Default `t` range: 0 to 2π.** Most coursework curves (circles, helices,
  cycloids) are 2π-periodic.
- Two components mean (x, y) with z = 0.
- In 2D it's drawn in a plane when its out-of-plane component is identically zero or
  missing. Otherwise it's 3D only, like a surface.
- A new `parse_tuple(text, defs) -> tuple[Expr, ...]` in mathcore splits top-level commas
  inside the outer parentheses.
- It gets a builder (with the same asymptote breaking as curves, via `_curve_segments`),
  a `READERS` entry that returns `None` (a parametric curve has no single value at the
  cursor's x), a command (typing a tuple plots it), and dropdown support. The registry
  tests from Step 0d enforce all of these.

### Sub-steps

- [ ] **4a. Tokenizer and parser take names.** `tokenize`/`parse` accept defined names.
  Tests: `rho` with and without `rho` defined, longest match, user functions parse as
  calls only when defined as functions, and reserved names are rejected. All existing
  parser tests pass unchanged (the defaults are empty).
- [ ] **4b. `Definition`, `parse_definition`, `resolve`.** Tests: constants, functions,
  nesting, order independence, cycles (with the path in the message), wrong arity, and
  diff through a user function (`diff(f(x, y), x)` with `f(x, y) = x^2*y` → `2*x*y`).
  The Step 0c properties gain a variant with random constants substituted.
- [ ] **4c. Definitions in the document** and the three actions, plus `rebuild_plots`,
  atomic rejection with dependents named, and messages. Commands: typing `a = 2` or
  `f(x) = …` defines. `remove a` removes a definition (a plot and a definition can't
  share a name, so `remove` is unambiguous).
- [ ] **4d. Definitions panel** (a new sidebar tab "Definitions"): one row per definition
  (editable text, like plot rows), a slider under each numeric constant, and the
  `slider` command. Slider drags coalesce into one undo step.
- [ ] **4e. Save/load.** `to_script` writes definitions *first* (so plots that use them
  load), then sliders as `slider` lines, then objects. Extend the round-trip tests.
- [ ] **4f. `ParametricCurve`**, following "Adding a kind of plot" in DESIGN.md, plus
  `parse_tuple`. This completes the Todos item "Parametric curves with `t`".

### Decisions

- **Inline, don't evaluate by name.** `resolve` produces an ordinary expression, so the
  compiler, calculus, and contours never learn about definitions. One new concept in one
  place.
- **`f(x) = …` doesn't draw anything.** That keeps "what's a plot" a one-line rule.
- **A function's parameter can't reuse a defined name.** `f(a) = a^2` while `a = 2`
  exists is an error: "Parameter a is already defined". Shadowing would make `resolve`
  harder to picture. (Decided 2026-11-18.) Test it in 4b, including the reverse order:
  defining `a = 2` while `f(a) = a^2` exists is rejected too, naming `f`.
- **Out of scope for this step:** vectors and matrices as values, parametric surfaces,
  implicit equations (`x^2 + y^2 = 1`), and subscripted names (`a1`). Each becomes its own
  todo when a course needs it.

### Done when

- [ ] `a = 2`, `y = a*x`, dragging `a`'s slider moves the line; undo restores it in one step
- [ ] `f(x, y) = x^2*y` then `show diff(f(x, y), x)` prints `2*x*y`
- [ ] `(cos(t), sin(t), t/4)` draws a helix in 3D; `(cos(t), sin(t))` a circle in xy
- [ ] definitions, sliders, and parametric curves survive save → open
- [ ] DESIGN.md: "Equations only allow x, y, z" moves to History; a "Definitions" section
  explains `resolve`

---

## Step 5: ODE systems in mathcore; simulations become scene objects

**Goal:** differential equations are written as text, solved by pure functions, and
shown as solution curves, slope fields, and particle flows. The Lorenz attractor becomes a
script. Start this when the ODE course starts.

### Today

- `render/simulations.py` holds `GpuLorenz`: the equations are hardcoded
  in GLSL compute shaders, and the parameters are dataclasses in `scene.py`, outside the
  `Plot` family.
- DESIGN.md's open question "CPU or GLSL?" leans CPU.

### Target design

mathcore (`mathcore/ode.py`), pure:

```python
@dataclass(frozen=True)
class OdeSystem:
    state: tuple[str, ...]             # e.g. ("x", "y", "z"): 1 to 3 axes
    rates: tuple[Expr, ...]            # resolved right-hand sides, same order
    fn: Callable = field(compare=False, repr=False)   # compiled: (t, *state) -> rates

def parse_system(text: str, defs: Definitions, independent: str = "t") -> OdeSystem
    """"x' = sigma*(y - x), y' = x*(rho - z) - y, z' = x*y - beta*z"
    Primes on the left; the independent variable may appear on the right
    (non-autonomous systems). It's "t" for systems, and "x" for a single equation
    y' = f(x, y) (slope fields, `solve y' = …`), where x is the horizontal axis."""
def rk4_step(system: OdeSystem, t: float, state: np.ndarray, dt: float) -> np.ndarray
    """state has shape (len(system.state), n_points): steps many points at once."""
def solve(system: OdeSystem, t0: float, state0: np.ndarray, t1: float, n: int) -> np.ndarray
```

New plot kinds (each through the full "Adding a kind of plot" checklist):

| Kind | Typed as | Drawn as |
|---|---|---|
| `SolutionCurve` | `solve y' = -y + sin(x) from (0, 1)` (1-D: x is the independent variable) or `solve <system> from (x0, y0[, z0]) for t from 0 to 20` | The path, statically computed with `solve` (2000 steps), drawn like a curve. |
| `SlopeField` | `slopes y' = x - y` | Short segments on a 20×20 grid of the visible range, in the xy plane. 2D only. |
| `PhaseField` | `field x' = y, y' = -x` | Normalized arrows for 2-variable systems, in that plane. |
| `Flow` | `flow <system> [points 20000] [seed 0]` | Animated particles; the Lorenz look. |

**Time is the one impure thing, and it lives in the shell.** A `Flow`'s particle
positions are not in the document. The viewport keeps `flows: dict[ObjectId,
FlowState]` with `FlowState(t, positions)`. Each frame it calls the pure `rk4_step`.
When a `Flow`'s value changes (equations, points, seed), its state restarts from the
seed, the same rule the GPU simulations use today. The points mesh comes from a pure
builder: `points_mesh(positions, color)`.

The Lorenz preset is a script shipped with the app (`the_calcutorium/presets/lorenz.calc`),
opened with `open preset lorenz`:

```
sigma = 10
rho = 28
beta = 8/3
flow x' = sigma*(y - x), y' = x*(rho - z) - y, z' = x*y - beta*z points 20000
```

### Sub-steps

- [ ] **5a. `parse_system`, `rk4_step`, `solve`** with tests: exponential decay
  (y' = −y vs e^(−t), error < 1e-8 at dt = 0.01), harmonic oscillator energy drift
  < 1e-6 over 10 periods, the Lorenz system's first step against a hand-computed value,
  shapes for many points, and purity (input arrays not modified).
- [ ] **5b. `SolutionCurve`** (2D and 3D).
- [ ] **5c. `SlopeField` and `PhaseField`.** Their builders use the visible range,
  like curves in 2D.
- [ ] **5d. `Flow` + the shell's flow state + `points_mesh`**. Performance check: the
  Lorenz preset with 20,000 points runs at ≥ 30 fps on your machine. If it doesn't, lower
  the default point count. Moving the stepping to the GPU is a separate, later decision.
- [ ] **5e. Lorenz preset; delete `LorenzAttractor`, `GpuLorenz`, and the lorenz
  shaders.** The Simulations panel is replaced by the preset and the Definitions
  panel's sliders (`rho` is now a slider). Old `.calc` files with `add lorenz` load via
  a compatibility rule in `load_script`: it becomes the preset's lines.
- [x] **5f. N-body is gone.** It was removed early, on 2026-11-18: it's physics, not
  coursework, and it needed its own GPU compute code. After 5e removes Lorenz's GPU
  code, delete what's left of `render/simulations.py` and `Renderer.draw_simulation`.

### Decisions

- **CPU with numpy first**, as DESIGN.md already leans. Any expression works, and it's
  testable. Revisit only if 5d's performance check fails.
- **ODE state is not saved**; a flow restarts from its seed on load, just as
  simulations do today.

### Done when

- [ ] `render/simulations.py` and every `.comp` shader are deleted
- [ ] the Lorenz attractor runs from `presets/lorenz.calc`, and `rho` is a slider
- [ ] slope fields and phase portraits work for any typed system
- [ ] DESIGN.md: the "Simulations bypass mathcore" bullet and the CPU/GLSL open question
  move to History

---

## Rules that apply from now on

- **Split files before they sprawl.** A module past ~400 lines splits along its natural
  seam, e.g. `render/meshes.py` → `render/meshes/{__init__,grid,curve,surface,contour}.py`,
  with `BUILDERS` staying in `__init__.py`. Layers stay as they are; don't regroup
  folders by feature.
- **Every per-kind table gets a registry test** (Step 0d).
- **Every new piece of document state gets a command and a `to_script` line, and is
  added to the round-trip test** (Step 2). If it isn't in the script, it isn't saved.
- **Out of bounds:** a plugin system, a rewrite in another language, and replacing the
  symbolic core with sympy. Owning `diff` is part of the point.

## Where the feature todos fit

| Todos.md item | Do it after | Why |
|---|---|---|
| Contours: label levels; show the 50-line cap | Step 3 | Labels travel with meshes (`Mesh.labels`). |
| 3D camera frames what's plotted; set a plot's domain | Step 3 | Framing is a pure function of the doc → `Orbit3D`. The domain needs a command + script line (Step 2 rule). |
| Gradient vector field | Step 1 | Refers to its surface by id. |
| Tangent plane at a point | Step 1 | Same: `TangentPlane(surface_id, point)`. |
| Parametric curves with `t` | Part of Step 4 (4f) | |
| Axis labels and ticks in 3D | Step 3 | `world_to_screen` + `Label`s work in 3D. |
| Color surfaces by height | Any time | Independent: a builder + shader change. |
| General ODE systems, slope fields, phase portraits | Step 5 | |
| Linear algebra | Step 4 | Starts by adding vector/matrix values on top of definitions (its own design write-up then). |
| Sequences and series | Step 4 | `n` as a bound variable, like `t`. |
| Hide/show toggle per plot | Step 1 | `Document.hidden: frozenset[ObjectId]`, undoable and saved. |
| Save/load a scene | Is Step 2 | |
