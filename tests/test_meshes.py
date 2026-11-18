import numpy as np
import pytest

from the_calcutorium.mathcore import parse_equation
from the_calcutorium.render.camera import Camera2D
from the_calcutorium.render.meshes import BUILDERS, grid_mesh, plot_mesh
from the_calcutorium.scene import Contour, Plot, graph

RED = (1.0, 0.0, 0.0)


def plot(text: str) -> Plot:
    return graph(parse_equation(text), RED)


def contour(text: str, interval=None) -> Contour:
    return Contour(parse_equation(text), RED, interval=interval)


def test_every_plot_kind_has_a_builder():
    kinds = {cls for cls in Plot.__subclasses__()}
    assert kinds == set(BUILDERS)


def test_grid_labels_are_major_lines_only():
    _, labels = grid_mesh("xy", (-10, 10), (-10, 10))
    assert labels.h == (-10.0, 0.0, 10.0)  # spacing 2, major every 5th line


def test_curve_vertex_layout():
    mesh = plot_mesh(plot("y = 2x"))
    assert mesh.program == "basic" and mesh.mode == "lines"
    assert mesh.data.shape[1] == 6
    np.testing.assert_allclose(mesh.data[:, 1], 2 * mesh.data[:, 0], rtol=1e-5)
    np.testing.assert_allclose(mesh.data[:, 3:], np.tile(RED, (len(mesh.data), 1)))


def test_curve_in_2d_uses_visible_range():
    mesh = plot_mesh(plot("y = x"), "xy", (0, 100), (0, 100))
    assert mesh.data[:, 0].min() == pytest.approx(0)
    assert mesh.data[:, 0].max() == pytest.approx(100)


def test_curve_not_in_plane_is_hidden():
    assert plot_mesh(plot("z = x"), "xy", (-1, 1), (-1, 1)) is None
    assert plot_mesh(plot("z = x"), "xz", (-1, 1), (-1, 1)) is not None


def test_sideways_curve_in_plane():
    # x = y^2 in the xy plane: input y runs along the vertical axis
    mesh = plot_mesh(plot("x = y^2"), "xy", (-5, 5), (-2, 2))
    assert mesh.data[:, 1].min() == pytest.approx(-2)


def test_constant_lies_along_the_plane():
    mesh = plot_mesh(plot("y = 3"), "yz", (-1, 1), (-5, 5))  # yz: horizontal axis is z
    np.testing.assert_allclose(mesh.data[:, 1], 3)
    assert np.ptp(mesh.data[:, 2]) == pytest.approx(2)


def test_undefined_points_break_the_curve():
    mesh = plot_mesh(plot("sqrt(x)"))
    assert (mesh.data[:, 0] >= 0).all()


def test_asymptote_jump_is_not_drawn():
    mesh = plot_mesh(plot("1/x"), "xy", (-10, 10), (-10, 10))
    segments = mesh.data[:, :3].reshape(-1, 2, 3)
    crosses_zero = (segments[:, 0, 0] < 0) & (segments[:, 1, 0] > 0)
    assert not crosses_zero.any()


def test_tan_asymptotes_are_not_drawn():
    mesh = plot_mesh(plot("tan(x)"), "xy", (-30, 30), (-25, 25))
    segments = mesh.data[:, :3].reshape(-1, 2, 3)
    x0, x1 = segments[:, 0, 0], segments[:, 1, 0]
    # an asymptote at (k + 1/2)π lies strictly inside a drawn segment
    k = np.floor(x0 / np.pi - 0.5)
    asymptote = (k + 1.5) * np.pi
    assert not ((x0 < asymptote) & (asymptote < x1)).any()


def test_steep_but_continuous_curve_is_kept():
    mesh = plot_mesh(plot("y = 50x"), "xy", (-10, 10), (-10, 10))
    assert len(mesh.data) == 2 * (500 - 1)  # every segment drawn


def test_surface_mesh():
    mesh = plot_mesh(plot("z = x*y"))
    assert mesh.program == "surface" and mesh.data.shape[1] == 9
    assert len(mesh.data) % 3 == 0
    normals = mesh.data[:, 3:6]
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, rtol=1e-5)


def test_surfaces_are_hidden_in_2d():
    assert plot_mesh(plot("z = x*y"), "xy", (-1, 1), (-1, 1)) is None


def test_contour_in_3d_floats_at_each_level():
    mesh = plot_mesh(contour("z = x^2 + y^2", interval=25))  # values 0..200 over ±10
    z = mesh.data[:, 2]
    assert set(np.round(z, 6)) == {25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0}
    full_circles = z <= 100
    r = np.linalg.norm(mesh.data[full_circles, :2], axis=1)
    np.testing.assert_allclose(r, np.sqrt(z[full_circles]), atol=0.1)


def test_contour_in_its_plane_is_flat_and_uses_visible_range():
    mesh = plot_mesh(contour("z = x^2 + y^2", interval=1), "xy", (-1.2, 1.2), (-1.2, 1.2))
    np.testing.assert_allclose(mesh.data[:, 2], 0.0)
    r = np.linalg.norm(mesh.data[:, :2], axis=1)
    # values reach 2.88 in this range, so levels 1 and 2: circles of radius 1 and √2
    nearest = np.where(np.abs(r - 1) < np.abs(r - np.sqrt(2)), 1.0, np.sqrt(2))
    np.testing.assert_allclose(r, nearest, atol=0.01)  # finer grid in a small range → tight circles
    assert (np.abs(r - 1) < 0.01).sum() > 100


def test_contour_is_hidden_in_other_planes():
    assert plot_mesh(contour("z = x*y"), "xz", (-1, 1), (-1, 1)) is None


def test_contour_auto_levels():
    mesh = plot_mesh(contour("z = x^2 + y^2"))
    assert len(set(np.round(mesh.data[:, 2], 6))) >= 5


@pytest.mark.parametrize("plane", ["xy", "xz", "yz"])
def test_2d_camera_shows_horizontal_axis_left_to_right(plane):
    """The bug in the old camera: xz and yz were mirrored relative to the labels."""
    cam = Camera2D(plane, distance=10)
    view = np.array(cam.view_matrix()).reshape(4, 4).T  # back to row-major
    proj = np.array(cam.projection_matrix(100, 100)).reshape(4, 4).T
    h, v = cam.h_index, cam.v_index

    def screen(point):
        clip = proj @ view @ np.append(point, 1.0)
        return clip[:2] / clip[3]

    right = np.zeros(3)
    right[h] = 1
    up = np.zeros(3)
    up[v] = 1
    assert screen(right)[0] > 0 and abs(screen(right)[1]) < 1e-6
    assert screen(up)[1] > 0 and abs(screen(up)[0]) < 1e-6


def test_2d_camera_manual_ranges_map_to_screen_edges():
    """The other old bug: 'range x 0 10' showed 5..15 because the camera offset was ignored."""
    cam = Camera2D("xy")
    cam.fit((0, 10), (0, 4))
    view = np.array(cam.view_matrix()).reshape(4, 4).T
    proj = np.array(cam.projection_matrix(100, 100, ((0, 10), (0, 4)))).reshape(4, 4).T
    left_bottom = proj @ view @ np.array([0, 0, 0, 1.0])
    right_top = proj @ view @ np.array([10, 4, 0, 1.0])
    np.testing.assert_allclose(left_bottom[:2], [-1, -1], atol=1e-5)
    np.testing.assert_allclose(right_top[:2], [1, 1], atol=1e-5)
