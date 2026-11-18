import pytest

from the_calcutorium.app import App, run_command
from the_calcutorium.scene import Contour, Curve, LorenzAttractor, Plot, Surface


@pytest.fixture
def app():
    app = App()
    app.log = []
    app.message.subscribe(lambda level, text: app.log.append((level, text)))
    return app


def errors(app):
    return [text for level, text in app.log if level == "error"]


def test_add_plot(app):
    plot = app.add_plot("z = x*y")
    assert app.scene.plots() == [plot]
    assert isinstance(plot, Surface)
    assert isinstance(app.add_plot("x^2"), Curve)


def test_invalid_plot_reports_error_and_changes_nothing(app):
    assert app.add_plot("y = (") is None
    assert app.scene.plots() == []
    assert errors(app)


def test_duplicate_plot_is_ignored(app):
    app.add_plot("x^2")
    app.add_plot("x^2")
    assert len(app.scene.plots()) == 1


def test_plots_get_different_colors(app):
    a, b = app.add_plot("x"), app.add_plot("x^2")
    assert a.color != b.color


def test_edit_plot_replaces_it_and_keeps_color(app):
    old = app.add_plot("x^2")
    new = app.edit_plot(old, "x^3")
    assert app.scene.plots() == [new]
    assert new.color == old.color


def test_bad_edit_keeps_the_old_plot(app):
    old = app.add_plot("x^2")
    assert app.edit_plot(old, "x^") is None
    assert app.scene.plots() == [old]


def test_editing_a_curve_into_two_inputs_makes_a_surface(app):
    curve = app.add_plot("x^2")
    assert isinstance(app.edit_plot(curve, "x*y"), Surface)


def test_add_contour(app):
    c = app.add_contour("z = x^2 + y^2")
    assert isinstance(c, Contour) and c.interval is None


def test_set_contour_interval(app):
    c = app.add_contour("x*y")
    c = app.set_contour_interval(c, 5)
    assert app.scene.plots() == [c] and c.interval == 5
    c = app.set_contour_interval(c, None)
    assert c.interval is None


@pytest.mark.parametrize("bad", [0, -1, float("nan")])
def test_contour_interval_must_be_positive(app, bad):
    c = app.add_contour("x*y")
    assert app.set_contour_interval(c, bad) is None
    assert "positive" in errors(app)[-1]
    assert app.scene.plots()[0].interval is None


def test_contour_again_changes_the_interval(app):
    app.add_contour("x*y")
    app.add_contour("x*y", 2)
    assert [p.interval for p in app.scene.plots()] == [2]


def test_contour_needs_two_inputs(app):
    assert app.add_contour("y = x^2") is None
    assert "needs 2" in errors(app)[-1]
    assert app.scene.plots() == []


def test_same_equation_as_surface_and_contour(app):
    app.add_plot("x*y")
    app.add_contour("x*y")
    assert len(app.scene.plots()) == 2
    app.add_contour("x*y")  # but not the same kind twice
    assert len(app.scene.plots()) == 2


def test_editing_a_contour_keeps_it_a_contour(app):
    c = app.add_contour("x*y")
    assert isinstance(app.edit_plot(c, "x^2 - y^2"), Contour)
    assert app.edit_plot(app.scene.plots()[0], "x^2") is None  # 1 input: not a valid contour
    assert isinstance(app.scene.plots()[0], Contour)


def test_invalid_plot_cannot_be_constructed():
    from the_calcutorium.mathcore import MathError, parse_equation
    with pytest.raises(MathError):
        Surface(parse_equation("x^2"), (1, 1, 1))


def test_scene_events(app):
    seen = []
    app.scene.added.subscribe(lambda o: seen.append("added"))
    app.scene.replaced.subscribe(lambda old, new: seen.append("replaced"))
    app.scene.removed.subscribe(lambda o: seen.append("removed"))
    plot = app.add_plot("x")
    plot = app.edit_plot(plot, "2x")
    app.remove_plot(plot)
    assert seen == ["added", "replaced", "removed"]


def test_simulation_lifecycle(app):
    app.add_simulation("lorenz")
    app.add_simulation("lorenz")
    assert len(app.scene.objects) == 1
    app.update_simulation("lorenz", {"rho": 20.0})
    assert app.scene.find(LorenzAttractor).rho == 20.0
    app.remove_simulation("lorenz")
    assert app.scene.objects == ()


def test_unknown_simulation_param(app):
    app.add_simulation("nbody")
    app.update_simulation("nbody", {"warp": 9})
    assert errors(app)


def test_view_and_ranges(app):
    app.set_range("x", -1, 1)
    assert errors(app)  # ranges are 2D-only
    app.set_view("xy")
    app.set_range("x", -1, 1)
    app.set_range("y", 0, 5)
    assert app.view.fixed_ranges == ((-1, 1), (0, 5))
    app.auto_range()
    assert app.view.fixed_ranges is None


def test_view_changed_event(app):
    views = []
    app.view_changed.subscribe(views.append)
    app.set_view("xz")
    assert [v.mode for v in views] == ["xz"]


def test_show(app):
    assert app.show("diff(x^2*y, x)") == "2*x*y"


def test_show_evaluates_when_no_variables_are_left(app):
    assert app.show("integrate(t^2, t, 0, 3)") == "9"
    assert app.show("diff(x^3, x)") == "3*x^2"


# ---- commands --------------------------------------------------------------

def test_plot_command(app):
    run_command(app, "plot z = sin(x)*cos(y)")
    assert [p.text for p in app.scene.plots()] == ["z = sin(x)*cos(y)"]


def test_bare_equation_is_plotted(app):
    run_command(app, "y = x^2")
    assert len(app.scene.plots()) == 1


def test_remove_command(app):
    run_command(app, "x^2")
    run_command(app, "remove x^2")
    assert app.scene.plots() == []


def test_contour_command(app):
    run_command(app, "contour z = x^2 + y^2")
    assert isinstance(app.scene.plots()[0], Contour)


def test_contour_every_command(app):
    run_command(app, "contour z = x^2 + y^2 every 5")
    assert app.scene.plots()[0].interval == 5
    run_command(app, "contour z = x^2 + y^2 every 0.5")
    assert app.scene.plots()[0].interval == 0.5
    run_command(app, "contour z = x^2 + y^2 every auto")
    assert app.scene.plots()[0].interval is None
    run_command(app, "contour z = x^2 + y^2 every lots")
    assert "isn't a number" in errors(app)[-1]


def test_remove_needs_a_kind_when_ambiguous(app):
    run_command(app, "x*y")
    run_command(app, "contour x*y")
    run_command(app, "remove x*y")
    assert "more than one way" in errors(app)[-1]
    run_command(app, "remove contour x*y")
    assert [type(p) for p in app.scene.plots()] == [Surface]


def test_view_commands(app):
    run_command(app, "view xy")
    assert app.view.mode == "xy"
    run_command(app, "view 2d yz")
    assert app.view.mode == "yz"
    run_command(app, "view 3d")
    assert app.view.mode == "3d"


def test_range_command(app):
    run_command(app, "view xy")
    run_command(app, "range x -5 5")
    assert app.view.ranges == {"x": (-5.0, 5.0)}
    run_command(app, "range x five 5")
    assert errors(app)


def test_simulation_commands(app):
    run_command(app, "add lorenz")
    assert isinstance(app.scene.objects[0], LorenzAttractor)
    run_command(app, "remove lorenz")
    assert app.scene.objects == ()


def test_clear_command(app):
    run_command(app, "x")
    run_command(app, "add nbody")
    run_command(app, "clear")
    assert app.scene.objects == ()


def test_help_lists_commands(app):
    run_command(app, "help")
    assert "plot <equation>" in app.log[-1][1]


def test_plot_object_is_immutable(app):
    plot = app.add_plot("x")
    with pytest.raises(AttributeError):
        plot.color = (0, 0, 0)
    assert isinstance(plot, Plot)
