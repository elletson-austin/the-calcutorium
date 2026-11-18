# The Calcutorium

A personal math workbench: plot curves and surfaces, take partial derivatives and
integrals, and watch differential equations run, all built up feature by feature
alongside a math degree.

```
uv run main.py     # run the app
uv run pytest      # run the tests
```

Type into the bar under the view:

```
z = sin(x)*cos(y)                surface
y = integrate(cos(t), t, 0, x)   antiderivative curve
show diff(x^2*y, x)              → 2*x*y
view xy                          2D view of the xy-plane
help                             everything else
```

See [DESIGN.md](DESIGN.md) for how it's built and why.
