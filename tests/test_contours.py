import numpy as np
import pytest

from the_calcutorium.mathcore import auto_levels, marching_squares, nice_spacing, parse_equation, sample_contours


def grid(f, u_range=(-3, 3), v_range=(-3, 3), n=61):
    u = np.linspace(*u_range, n)
    v = np.linspace(*v_range, n)
    uu, vv = np.meshgrid(u, v)  # rows follow v, columns follow u
    return f(uu, vv), u, v


def test_circle_level_set():
    values, u, v = grid(lambda x, y: x**2 + y**2)
    segments = marching_squares(values, u, v, 4.0)  # x² + y² = 4
    assert len(segments) > 20
    radii = np.linalg.norm(segments.reshape(-1, 2), axis=1)
    np.testing.assert_allclose(radii, 2.0, atol=0.02)


def test_straight_line_level_set():
    values, u, v = grid(lambda x, y: x + y)
    points = marching_squares(values, u, v, 1.0).reshape(-1, 2)
    np.testing.assert_allclose(points.sum(axis=1), 1.0, atol=1e-9)  # linear: interpolation is exact


def test_level_outside_range_gives_nothing():
    values, u, v = grid(lambda x, y: x**2 + y**2)
    assert marching_squares(values, u, v, -1.0).shape == (0, 2, 2)


def test_undefined_cells_are_skipped():
    with np.errstate(invalid="ignore"):
        values, u, v = grid(lambda x, y: np.sqrt(x) + y)  # nan for x < 0
    points = marching_squares(values, u, v, 1.0).reshape(-1, 2)
    assert np.isfinite(points).all()
    assert (points[:, 0] >= -1e-9).all()


def test_saddle_is_resolved_by_cell_center():
    # f = x*y has a saddle at the origin; level 0 is the two axes, which must not be cut
    values, u, v = grid(lambda x, y: x * y + 0.01, n=4)
    segments = marching_squares(values, u, v, 0.0)
    assert len(segments) > 0
    assert np.isfinite(segments).all()


def test_marching_squares_is_pure():
    values, u, v = grid(lambda x, y: np.sin(x) * np.cos(y))
    before = values.copy()
    a = marching_squares(values, u, v, 0.3)
    b = marching_squares(values, u, v, 0.3)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("span, expected", [(20, 2), (100, 10), (7, 0.5), (0.3, 0.02), (50, 5)])
def test_nice_spacing(span, expected):
    assert nice_spacing(span) == pytest.approx(expected)


def test_auto_levels_are_round_numbers_inside_the_range():
    levels = auto_levels(np.array([0.0, 37.0]))
    assert levels == (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0)


def test_auto_levels_ignore_nan_and_handle_constants():
    assert auto_levels(np.array([np.nan, 3.0, 3.0])) == (3.0,)
    assert auto_levels(np.array([np.nan])) == ()


def test_sample_contours_lifts_lines_to_their_level():
    eq = parse_equation("z = x^2 + y^2")
    lines = sample_contours(eq, (-3, 3), (-3, 3), levels=(1.0, 4.0))
    assert [level for level, _ in lines] == [1.0, 4.0]
    for level, segments in lines:
        assert segments.shape[1:] == (2, 3)
        np.testing.assert_allclose(segments[..., 2], level)  # z = c
        r = np.linalg.norm(segments[..., :2], axis=-1)
        np.testing.assert_allclose(r, np.sqrt(level), atol=0.05)


def test_sample_contours_uses_the_equation_axes():
    eq = parse_equation("x = y*z")  # inputs (y, z), output x
    (level, segments), = sample_contours(eq, (-2, 2), (-2, 2), levels=(1.0,))
    np.testing.assert_allclose(segments[..., 0], 1.0)            # x = c
    np.testing.assert_allclose(segments[..., 1] * segments[..., 2], 1.0, atol=0.02)  # y*z = 1
