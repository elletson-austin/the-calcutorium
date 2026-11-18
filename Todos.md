# Todos

One task per session. Check DESIGN.md before starting.

## Architecture (in order; details and "done when" lists in ROADMAP.md)
- [ ] Step 0: Guardrails (ruff, pyright, property tests, registry tests)
- [ ] Step 1: The document is one immutable value (ids, pure actions, undo/redo)
- [ ] Step 2: Save and open as command scripts (`.calc` files)
- [ ] Step 3: Frame plan and cameras as data (fixes 2D→3D zoom carry-over)
- [ ] Step 4: Definitions, sliders, parametric curves (before next semester)
- [ ] Step 5: ODE systems in mathcore; Lorenz becomes a preset (with the ODE course)

ROADMAP.md's last table says which feature todos below should wait for which step.

## Multivariable calculus (this semester)
- [x] Contour / level curves of a surface: `contour z = x^2 + y^2`
- [x] Contours at a set interval (`every 5`), settable per contour in the plot list
- [ ] Contours: label each level with its value; show when the 50-line cap kicks in
- [ ] 3D camera frames what's plotted; set a plot's domain (`contour z = x^2+y^2` is mostly off-screen now)
- [ ] Gradient vector field: arrows of ∇f on a grid
- [ ] Tangent plane to a surface at a point
- [ ] Parametric curves with `t`: `(cos(t), sin(t), t/4)`
- [ ] Axis labels (x, y, z) and tick numbers in the 3D view
- [ ] Color surfaces by height (colormap) instead of a single color

## Next semester
- [ ] General ODE systems (dx/dt = …), with Lorenz as a preset; slope fields, phase portraits
- [ ] Linear algebra: vectors, matrices as transformations of the grid
- [ ] Sequences and series: partial sums, convergence

## Polish
- [x] 2D hover readout: cursor coordinates, curve values (with a dot), f(x, y) of surfaces/contours
- [ ] Hide/show toggle per plot
- [ ] Save/load a scene to a file
