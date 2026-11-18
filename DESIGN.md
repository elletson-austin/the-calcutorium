# The Calcutorium — Design

## Vision

A personal math workbench that grows alongside a math degree. It is not a Desmos clone
and not a product: it's a tool I understand end-to-end and extend one feature at a time
as my courses need it.

**Course roadmap (drives feature priority):**

| When | Course | Features it wants |
|---|---|---|
| Now | Multivariable calculus | Surfaces, level curves/contours, partial derivatives, gradient fields, tangent planes, parametric curves, double integrals |
| Next semester | Linear algebra | Vectors, matrices as transformations, eigenvectors, span/column space visuals |
| Next semester | Differential equations | Slope fields, phase portraits, general ODE systems (Lorenz becomes a preset) |
| Next semester | Analysis | Sequences, series & partial sums, convergence, epsilon-delta visuals |

## Principles

1. **I can picture it.** Every module has one job, and I can explain what's behind the screen.
   If a change makes the architecture harder to explain, it's the wrong change.
2. **Architecture before features.** A clean core makes features cheap to add.
3. **Math is pure functions, all the way down.** See [Pure math core](#pure-math-core).
   The math code knows nothing about Qt or OpenGL and is fully unit-tested.
4. **One path per action.** A GUI button and a typed command do the exact same thing
   through the exact same code.
5. **Dependencies point one way:** `ui → app → scene → math`. Lower layers never import higher ones.
6. **Small and explicit beats clever.** No speculative abstractions; add structure when the
   second use case shows up, not the first.

## Interaction model

Two ways to drive the app, both first-class:

- **Command bar:** a REPL-style input: `plot z = sin(x)*cos(y)`, `view 2d xy`, `range x -5 5`.
- **GUI panels:** sidebar editors, buttons, and parameter sliders.

Both call the same `App` actions (see below). The output log shows the result of every action.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ ui/        Qt widgets. Shows things, forwards user intent.  │
│            main_window, panels, plot_list, command_bar,     │
│            output_log, viewport, overlay                    │
└──────────────┬──────────────────────────────────────────────┘
               │ calls actions                ▲ listens to events
               ▼                              │
┌─────────────────────────────────────────────────────────────┐
│ app/       The "brain". All actions live here. No Qt.       │
│            actions.py   App: add_plot, set_view, show, ...  │
│            commands.py  text → action  ("plot ..." → add)   │
│            view.py      ViewState: 3d/xy/xz/yz + ranges     │
└──────────────┬──────────────────────────────────────────────┘
               │ changes
               ▼
┌─────────────────────────────────────────────────────────────┐
│ scene.py   Pure data: WHAT is on screen, not HOW to draw it.│
│            Scene (objects + added/removed/replaced events)  │
│            Curve, Surface, Contour (all Plots), simulations │
└──────────────┬──────────────────────────────────────────────┘
               │ uses
               ▼
┌─────────────────────────────────────────────────────────────┐
│ mathcore/  Pure functions + numpy. No Qt, no GL. Tested.    │
│            tokenizer, parser, expr, calculus, compiler,     │
│            equation, sampling, contours, ticks              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ render/    Reads the scene, never changes it.               │
│            meshes.py    scene → vertex arrays (pure numpy)  │
│            readout.py   scene + cursor → hover text (pure)  │
│            camera.py    view/projection matrices            │
│            renderer.py, simulations.py  the only GPU code   │
└─────────────────────────────────────────────────────────────┘
   (render/ is used by ui/viewport; it depends on scene + mathcore)

planes.py, events.py: tiny shared helpers (axis names, observer Event).
```

(The math package is `mathcore`, not `math`, so it can never shadow Python's built-in `math`.)

### The life of an action

Typing `z = x^2 + y^2` in the command bar:

1. `ui/command_bar` emits the text; `ui/main_window` passes it to `run_command`.
2. `app/commands` sees it isn't a command word → calls `app.add_plot("z = x^2 + y^2")`.
3. `add_plot` calls `mathcore.parse_equation` → an `Equation` (output z, inputs x, y,
   plus a compiled numpy function).
4. It wraps that in a `Surface` (2 inputs; `scene.graph` picks Curve or Surface) with a
   color and calls `scene.add(plot)`. `contour …` would make a `Contour` instead.
5. The scene emits `added`. `ui/plot_list` adds an editable row; the log prints "Plotted …".
6. Next frame, `ui/viewport` loops over `scene.plots()` and asks the renderer to draw each.
   The renderer has no GPU buffer for this plot yet, so it calls `render/meshes.plot_mesh`,
   which looks up the builder for `Surface` in `BUILDERS`. That calls
   `mathcore.sample_surface` and builds triangles, which the renderer uploads.

Clicking "Add" in the Plots panel enters at step 2 (`app.add_plot`): same code from there on.

### Pure math core

Everything in `mathcore/` is a **pure function**: one input maps to exactly one output, the way
a function does in a math class.

- **Same input → same output, always.** No hidden state, no globals, no randomness, no
  reading the clock, the scene, or the camera.
- **No side effects.** Nothing is mutated, printed, logged, or drawn. A function's only
  effect is its return value.
- **Every input is explicit.** If the result depends on something, it's a parameter.
  `diff(expr, var)` — never "guess which variable you meant".
- **Expressions are immutable values.** AST nodes are frozen dataclasses. Operations build
  *new* trees and never modify the old one, so `diff(f, x)` leaves `f` untouched.
- **Purity goes down the call tree.** A pure function only calls other pure functions.
  If something deep inside needs an impure step, the design is wrong. Pass the data in instead.

The core signatures look like math:

```python
parse(text: str) -> Expr
diff(expr: Expr, var: str) -> Expr            # ∂expr/∂var
simplify(expr: Expr) -> Expr
substitute(expr: Expr, var: str, value: Expr) -> Expr
compile_expr(expr: Expr, inputs: tuple[str, ...]) -> Callable[..., np.ndarray]
parse_equation(text: str) -> Equation          # "z = x*y" → output z, inputs (x, y)
sample_surface(eq: Equation, range1, range2, n) -> np.ndarray   # same args → same grid
```

In expressions: `diff(f, x)` is ∂f/∂x (plain `diff(f)` only when f has one variable),
and `integrate(f, t, a, b)` is ∫ₐᵇ f dt, computed numerically. Put `x` in a bound to get
an antiderivative: `y = integrate(cos(t), t, 0, x)`. Integrals nest, so double integrals work.

Because of this, derivatives compose like they do on paper: `diff(diff(f, x), y)` is
∂²f/∂y∂x, and comparing the two mixed partials is a one-line test.

**Where impurity lives.** State and side effects are allowed only at the edges: `app/`
(changes the scene), `ui/` (Qt), and `render/` (GPU). This is "functional core, imperative
shell": the shell gathers inputs, calls the pure core, and applies the result.

### Key rules

- **Scene objects describe math, not GPU.** A plot holds an `Equation`, a color, and a
  domain. It knows nothing about shaders or vertex layouts; `render/meshes.py` owns that.
- **Each kind of plot is its own class.** `Curve`, `Surface`, `Contour` all extend `Plot`
  and hold only the fields they need (a `Contour` has `levels`). Each checks its equation
  in `__post_init__`, so an invalid plot (a surface with one input) can't be built. The same
  equation can be shown several ways at once: `z = x*y` as a Surface *and* a Contour.
  Settings are fields too (`Contour.interval`), changed by an action that builds a new
  plot (`set_contour_interval`), so the scene events keep the UI in sync automatically.
- **One builder per kind.** `render/meshes.py` maps each class to a function
  `builder(plot, plane, h_range, v_range) -> Mesh | None` in the `BUILDERS` table.

### Adding a kind of plot

1. Math: any new pure function goes in `mathcore/` with tests (contours: `contours.py`).
2. Scene: a `Plot` subclass in `scene.py` with its `kind`, allowed `inputs`, and fields.
3. Render: a builder in `render/meshes.py`, registered in `BUILDERS`
   (a test fails if you forget: `test_every_plot_kind_has_a_builder`).
4. App: an `add_<kind>` action in `app/actions.py`, a command in `app/commands.py`, and an
   entry in `PlotsPanel.ADD_ACTIONS` for the dropdown.
- **Scene objects are immutable.** Editing a plot builds a new `Plot` and calls
  `scene.replace(old, new)`. That's why there's no "dirty" flag: a new object *is* the change.
- **The scene announces changes** through `Event`s (`added`, `removed`, `replaced`).
  Widgets subscribe; nobody passes update callbacks around.
- **Rendering is read-only.** Meshes are a pure function of (scene object, plane, visible
  range). The renderer caches GPU buffers per object and rebuilds only when that
  signature changes, e.g. when you pan a 2D view.
- **View state is data.** `app.view` (a frozen `ViewState`) says 3D vs. which plane and any
  manual ranges. The viewport turns it into a camera.

### Base version feature list

- Curves (`y = x^2`, `x = y^2`, `z = sin(x)`) and surfaces (`z = x*y`, `x*y`), shown in 3D;
  curves also in the matching 2D plane, re-sampled to whatever range is visible.
- Contours (`contour z = x^2 + y^2 every 5`): level curves by marching squares at every
  multiple of an interval you set (the "every" box on the plot's row; 0 = auto, ~10 round-
  numbered levels). Multiples stay put when you pan. More than 50 lines in view and it
  draws every k-th multiple instead. Flat in their 2D plane; in 3D each floats at its height.
- Partial derivatives and integrals inside any expression; `show <expr>` prints the
  worked-out result (`show diff(x^2*y, x)` → `2*x*y`).
- Command bar with history (↑/↓). Anything that isn't a command is plotted. `help` lists it all.
- Views: 3D with z up, like a calculus textbook (orbit, zoom, WASD, Tab =
  perspective/orthographic); 2D xy/xz/yz (pan, zoom, grid with labels, manual ranges).
- 2D hover readout: the cursor's coordinates, each curve's value there (with a dot on the
  curve), and each surface/contour's f at the cursor. `render/readout.py` is pure: the
  cursor is rounded to what one pixel can resolve, and every value is computed at that
  rounded point, so the numbers shown agree with each other.
- Curves break cleanly at undefined points and asymptotes (tan, 1/x); surfaces are lit
  on both sides.
- Lorenz attractor simulation on the GPU.

## History

The first version (before 2026-11-17) was one 900-line `symbolic.py` plus a tangle of UI and
render code. The rewrite fixed, among other things:

- `diff()` picked the alphabetically-first variable, so you couldn't choose ∂/∂x vs ∂/∂y.
- `integrate()` was a cumulative sum that guessed its variable and broke on surfaces.
- `-x^2` parsed as `(-x)^2`.
- Typed commands were dead code: the command bar was never connected.
- The 2D xz and yz views were mirrored relative to their axis labels, and manual ranges
  showed the wrong region (`range x 0 10` displayed 5..15).
- The N-body particle count could be raised past the GPU buffer size.
- The renderer mutated scene objects while drawing.
- 3D used y-up, so `z = f(x, y)` stood up like a wall instead of lying over the xy-plane.
- The last OpenGL vertex array stayed bound when Qt's QPainter drew the 2D labels, so Qt
  overwrote it and that plot silently disappeared. The viewport now unbinds it
  (`glBindVertexArray(0)`) before any Qt drawing. Keep that line!

## Next up

See Todos.md for features. **ROADMAP.md is the step-by-step plan** that resolves the
pain points below (and the open questions), in order, with "done when" checklists.
Known architecture pain points:

- The viewport's mesh-build callbacks also stash label data (`_grid_labels`, `_anchors`) as
  a side effect. Builders should return labels with the mesh; input handling should move
  out of the viewport.
- Equations only allow x, y, z: no parameters (`t` for parametric curves, slider constants).
- The 3D camera doesn't frame what's plotted, and a plot's domain can't be set from the UI.
- Simulations bypass mathcore (equations hardcoded in GLSL). Resolve with general ODE systems.
- The renderer sets lighting uniforms on every program; it redraws at 60 fps even when idle.

## Open questions

- Derivative syntax for the user: the core is always `diff(expr, var)`. Should typed
  input also accept `d/dx f` as shorthand for it?
- ODE systems: evaluate on CPU with numpy (simple, any expression) or compile
  expressions to GLSL (fast, more complex)? Start CPU, revisit if slow.
- ~~Keep N-body at all?~~ Removed 2026-11-18: physics, not coursework, and it needed
  custom GPU compute code. It's in git history if ever wanted.
