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
│            Plot, LorenzAttractor, NBody (immutable)         │
└──────────────┬──────────────────────────────────────────────┘
               │ uses
               ▼
┌─────────────────────────────────────────────────────────────┐
│ mathcore/  Pure functions + numpy. No Qt, no GL. Tested.    │
│            tokenizer, parser, expr, calculus, compiler,     │
│            equation, sampling                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ render/    Reads the scene, never changes it.               │
│            meshes.py    scene → vertex arrays (pure numpy)  │
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
4. It wraps that in a `Plot` with a color and calls `scene.add(plot)`.
5. The scene emits `added`. `ui/plot_list` adds an editable row; the log prints "Plotted …".
6. Next frame, `ui/viewport` loops over `scene.plots()` and asks the renderer to draw each.
   The renderer has no GPU buffer for this plot yet, so it calls `render/meshes.plot_mesh`,
   which calls `mathcore.sample_surface` and builds triangles, then uploads them.

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

- **Scene objects describe math, not GPU.** A `Plot` holds an `Equation` and a color.
  It knows nothing about shaders or vertex layouts; `render/meshes.py` owns that.
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
- Partial derivatives and integrals inside any expression; `show <expr>` prints the
  worked-out result (`show diff(x^2*y, x)` → `2*x*y`).
- Command bar with history (↑/↓). Anything that isn't a command is plotted. `help` lists it all.
- Views: 3D with z up, like a calculus textbook (orbit, zoom, WASD, Tab =
  perspective/orthographic); 2D xy/xz/yz (pan, zoom, grid with labels, manual ranges).
- Curves break cleanly at undefined points and asymptotes (tan, 1/x); surfaces are lit
  on both sides.
- Lorenz attractor and N-body simulations on the GPU.

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

Features, starting with multivariable calculus:
contours/level curves, gradient fields, tangent planes, parametric curves (`t` as a parameter).

## Open questions

- Derivative syntax for the user: the core is always `diff(expr, var)`. Should typed
  input also accept `d/dx f` as shorthand for it?
- ODE systems: evaluate on CPU with numpy (simple, any expression) or compile
  expressions to GLSL (fast, more complex)? Start CPU, revisit if slow.
- Keep N-body at all? It's physics, not coursework.
