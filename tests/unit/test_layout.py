"""Unit tests for image-axis layout helpers."""

from __future__ import annotations

import numpy as np
import pytest

from nb_nd_viewer._layout import (
    extract_channel_plane,
    move_display_axes_to_end,
    remove_axes_from_layout,
    validate_layout,
)
from nb_nd_viewer._types import _AxisLayout


class TestValidateLayout:
    """Tests for axis-layout validation."""

    def test_resolves_display_channel_and_slider_axes(self) -> None:
        layout = validate_layout(np.zeros((5, 2, 3, 4)), "ZCYX")

        assert layout == _AxisLayout(
            axis_order="ZCYX",
            channel_axis=1,
            display_axes=(2, 3),
            slider_axes=(0,),
        )

    def test_resolves_reversed_x_y_axes_as_y_x_display(self) -> None:
        layout = validate_layout(np.zeros((5, 2, 4, 3)), "ZCXY")

        assert layout == _AxisLayout(
            axis_order="ZCXY",
            channel_axis=1,
            display_axes=(3, 2),
            slider_axes=(0,),
        )

    def test_resolves_y_x_as_display_axes_when_not_trailing(self) -> None:
        layout = validate_layout(np.zeros((2, 3, 4)), "YXZ")

        assert layout == _AxisLayout(
            axis_order="YXZ",
            channel_axis=None,
            display_axes=(0, 1),
            slider_axes=(2,),
        )

    def test_falls_back_to_last_two_axes_when_x_or_y_is_missing(self) -> None:
        layout = validate_layout(np.zeros((2, 3, 4)), "AZY")

        assert layout == _AxisLayout(
            axis_order="AZY",
            channel_axis=None,
            display_axes=(1, 2),
            slider_axes=(0,),
        )

    def test_rejects_multiple_channel_axes(self) -> None:
        with pytest.raises(ValueError, match="at most one channel axis"):
            validate_layout(np.zeros((2, 3, 4)), "CCY")

    def test_rejects_less_than_two_display_axes(self) -> None:
        with pytest.raises(ValueError, match="at least two non-channel axes"):
            validate_layout(np.zeros((2, 3)), "CY")

    def test_normalizes_axis_order_case(self) -> None:
        layout = validate_layout(np.zeros((2, 3)), "yx")

        assert layout.axis_order == "YX"

    def test_rejects_empty_axes(self) -> None:
        with pytest.raises(ValueError, match="axes must be non-empty"):
            validate_layout(np.zeros((0, 3)), "YX")

    def test_rejects_duplicate_axis_labels(self) -> None:
        with pytest.raises(ValueError, match="duplicated labels: 'Y'"):
            validate_layout(np.zeros((2, 3)), "YY")


class TestRemoveAxesFromLayout:
    """Tests for adjusting layout metadata after slider indexing."""

    def test_removes_slider_axis_and_adjusts_remaining_axes(self) -> None:
        layout = _AxisLayout(
            axis_order="ZCYX",
            channel_axis=1,
            display_axes=(2, 3),
            slider_axes=(0,),
        )

        adjusted = remove_axes_from_layout(layout, (0,))

        assert adjusted == _AxisLayout(
            axis_order="CYX",
            channel_axis=0,
            display_axes=(1, 2),
            slider_axes=(),
        )

    def test_rejects_removing_display_axes(self) -> None:
        layout = _AxisLayout(
            axis_order="ZYX",
            channel_axis=None,
            display_axes=(1, 2),
            slider_axes=(0,),
        )

        with pytest.raises(ValueError, match="display axes were removed"):
            remove_axes_from_layout(layout, (1,))


class TestMoveDisplayAxesToEnd:
    """Tests for image-plane orientation."""

    def test_moves_non_trailing_display_axes_to_render_order(self) -> None:
        layout = _AxisLayout(
            axis_order="YXC",
            channel_axis=2,
            display_axes=(0, 1),
            slider_axes=(),
        )
        plane = np.zeros((2, 3))

        moved = move_display_axes_to_end(plane, layout, removed_axis=2)

        assert moved.shape == (2, 3)


class TestExtractChannelPlane:
    """Tests for channel extraction and orientation."""

    def test_extracts_channel_before_orienting_display_axes(self) -> None:
        layout = _AxisLayout(
            axis_order="YXC",
            channel_axis=2,
            display_axes=(0, 1),
            slider_axes=(),
        )
        plane = np.arange(24).reshape(2, 3, 4)

        channel = extract_channel_plane(plane, layout, 1)

        np.testing.assert_array_equal(channel, plane[:, :, 1])

    def test_orients_non_channel_planes(self) -> None:
        layout = _AxisLayout(
            axis_order="XY",
            channel_axis=None,
            display_axes=(1, 0),
            slider_axes=(),
        )
        plane = np.arange(6).reshape(2, 3)

        channel = extract_channel_plane(plane, layout, 0)

        np.testing.assert_array_equal(channel, np.moveaxis(plane, (1, 0), (-2, -1)))

    def test_orients_reversed_x_y_planes_as_y_x(self) -> None:
        layout = _AxisLayout(
            axis_order="XY",
            channel_axis=None,
            display_axes=(1, 0),
            slider_axes=(),
        )
        plane = np.arange(6).reshape(2, 3)

        channel = extract_channel_plane(plane, layout, 0)

        np.testing.assert_array_equal(channel, plane.T)

    def test_extracts_channel_then_orients_reversed_x_y_axes(self) -> None:
        layout = _AxisLayout(
            axis_order="XCY",
            channel_axis=1,
            display_axes=(2, 0),
            slider_axes=(),
        )
        plane = np.arange(24).reshape(3, 4, 2)

        channel = extract_channel_plane(plane, layout, 1)

        np.testing.assert_array_equal(channel, plane[:, 1, :].T)
