"""Unit tests for the interactive image viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
from ipywidgets import HBox

from nb_nd_viewer.image_viewer import (
    RenderDownsamplingMode,
    _absolute_display_range,
    _AxisLayout,
    _DisplayLimits,
    _downsample_for_render,
    _draw_plane,
    _extract_channel_plane,
    _figure_html,
    _finite_min_max,
    _finite_percentiles,
    _make_axis_sliders,
    _make_channel_controls,
    _move_display_axes_to_end,
    _normalize_for_overlay,
    _remove_axes_from_layout,
    _set_absolute_to_slice_minmax,
    _slider_step,
    _sync_absolute_range,
    _validate_channel_colors,
    _validate_channel_names,
    _validate_layout,
    _validate_max_render_pixels,
    _validate_render_downsampling,
    view_image_layers,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestViewImageLayers:
    """Tests for the public ``view_image_layers`` function."""

    def test_returns_widget_for_2d_image(self) -> None:
        image = np.arange(9).reshape(3, 3)

        widget = view_image_layers(image, "YX")

        assert isinstance(widget, HBox)

    def test_returns_widget_for_channel_stack(self) -> None:
        image = np.arange(24).reshape(2, 3, 4)

        widget = view_image_layers(
            image,
            "CYX",
            channel_names=("DAPI", "Actin"),
            channel_colors=("blue", "green"),
        )

        assert isinstance(widget, HBox)

    def test_rejects_axis_order_with_wrong_length(self) -> None:
        image = np.zeros((2, 3))

        with pytest.raises(ValueError, match="axis_order has length 3"):
            view_image_layers(image, "ZYX")

    def test_rejects_channel_names_without_channel_axis(self) -> None:
        image = np.zeros((2, 3))

        with pytest.raises(ValueError, match="axis_order has no 'C' channel axis"):
            view_image_layers(image, "YX", channel_names=("Channel 0",))

    def test_rejects_too_few_channel_colors(self) -> None:
        image = np.zeros((2, 3, 4))

        with pytest.raises(ValueError, match="expected at least 2"):
            view_image_layers(image, "CYX", channel_colors=("red",))

    def test_rejects_non_positive_max_render_pixels(self) -> None:
        image = np.zeros((2, 3))

        with pytest.raises(ValueError, match="max_render_pixels must be a positive integer"):
            view_image_layers(image, "YX", max_render_pixels=0)

    def test_rejects_unknown_render_downsampling_mode(self) -> None:
        image = np.zeros((2, 3))

        with pytest.raises(ValueError, match="render_downsampling must be"):
            view_image_layers(
                image,
                "YX",
                render_downsampling=cast("RenderDownsamplingMode", "lanczos"),
            )


class TestMakeAxisSliders:
    """Tests for stack-axis slider construction."""

    def test_uses_requested_continuous_update_setting(self) -> None:
        layout = _AxisLayout(
            axis_order="ZYX",
            channel_axis=None,
            display_axes=(1, 2),
            slider_axes=(0,),
        )

        sliders = _make_axis_sliders(
            np.zeros((2, 3, 4)),
            layout,
            continuous_update=True,
        )

        assert sliders[0].continuous_update is True


class TestMakeChannelControls:
    """Tests for channel control construction."""

    def test_uses_requested_continuous_update_setting_for_range_sliders(self) -> None:
        controls = _make_channel_controls(
            0,
            "Image",
            (),
            continuous_update=True,
        )

        assert controls.absolute.continuous_update is True
        assert controls.percentile.continuous_update is True

    def test_defaults_to_absolute_range_without_auto_option(self) -> None:
        controls = _make_channel_controls(
            0,
            "Image",
            (),
            continuous_update=False,
        )

        assert controls.mode.value == "absolute"
        assert tuple(value for _label, value in controls.mode.options) == (
            "absolute",
            "percentile",
        )

    def test_absolute_controls_include_slice_minmax_controls(self) -> None:
        controls = _make_channel_controls(
            0,
            "Image",
            (),
            continuous_update=False,
        )
        controls.mode.value = "absolute"

        assert controls.absolute_slice_minmax.description == "Slice min/max"
        assert controls.absolute_slice_minmax in controls.box.children
        assert controls.absolute_slice_button.description == "Set slice min/max"
        assert controls.absolute_slice_button in controls.box.children

    def test_switches_visible_controls_when_percentile_mode_is_selected(self) -> None:
        controls = _make_channel_controls(
            0,
            "Image",
            (),
            continuous_update=False,
        )

        controls.mode.value = "percentile"

        assert controls.percentile in controls.box.children
        assert controls.absolute not in controls.box.children
        assert controls.absolute_slice_minmax not in controls.box.children


class TestValidateLayout:
    """Tests for axis-layout validation."""

    def test_resolves_display_channel_and_slider_axes(self) -> None:
        layout = _validate_layout(np.zeros((5, 2, 3, 4)), "ZCYX")

        assert layout == _AxisLayout(
            axis_order="ZCYX",
            channel_axis=1,
            display_axes=(2, 3),
            slider_axes=(0,),
        )

    def test_resolves_reversed_x_y_axes_as_y_x_display(self) -> None:
        layout = _validate_layout(np.zeros((5, 2, 4, 3)), "ZCXY")

        assert layout == _AxisLayout(
            axis_order="ZCXY",
            channel_axis=1,
            display_axes=(3, 2),
            slider_axes=(0,),
        )

    def test_resolves_y_x_as_display_axes_when_not_trailing(self) -> None:
        layout = _validate_layout(np.zeros((2, 3, 4)), "YXZ")

        assert layout == _AxisLayout(
            axis_order="YXZ",
            channel_axis=None,
            display_axes=(0, 1),
            slider_axes=(2,),
        )

    def test_falls_back_to_last_two_axes_when_x_or_y_is_missing(self) -> None:
        layout = _validate_layout(np.zeros((2, 3, 4)), "AZY")

        assert layout == _AxisLayout(
            axis_order="AZY",
            channel_axis=None,
            display_axes=(1, 2),
            slider_axes=(0,),
        )

    def test_rejects_multiple_channel_axes(self) -> None:
        with pytest.raises(ValueError, match="at most one channel axis"):
            _validate_layout(np.zeros((2, 3, 4)), "CCY")

    def test_rejects_less_than_two_display_axes(self) -> None:
        with pytest.raises(ValueError, match="at least two non-channel axes"):
            _validate_layout(np.zeros((2, 3)), "CY")

    def test_normalizes_axis_order_case(self) -> None:
        layout = _validate_layout(np.zeros((2, 3)), "yx")

        assert layout.axis_order == "YX"

    def test_rejects_empty_axes(self) -> None:
        with pytest.raises(ValueError, match="axes must be non-empty"):
            _validate_layout(np.zeros((0, 3)), "YX")

    def test_rejects_duplicate_axis_labels(self) -> None:
        with pytest.raises(ValueError, match="duplicated labels: 'Y'"):
            _validate_layout(np.zeros((2, 3)), "YY")


class TestRemoveAxesFromLayout:
    """Tests for adjusting layout metadata after slider indexing."""

    def test_removes_slider_axis_and_adjusts_remaining_axes(self) -> None:
        layout = _AxisLayout(
            axis_order="ZCYX",
            channel_axis=1,
            display_axes=(2, 3),
            slider_axes=(0,),
        )

        adjusted = _remove_axes_from_layout(layout, (0,))

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
            _remove_axes_from_layout(layout, (1,))


class TestValidateChannelNames:
    """Tests for channel display-name validation."""

    def test_generates_default_channel_names(self) -> None:
        names = _validate_channel_names(np.zeros((2, 3, 4)), 0, None)

        assert names == ("Channel 0", "Channel 1")

    def test_rejects_wrong_name_count(self) -> None:
        with pytest.raises(ValueError, match="expected 2"):
            _validate_channel_names(np.zeros((2, 3, 4)), 0, ("only one",))


class TestValidateChannelColors:
    """Tests for channel color validation."""

    def test_returns_rgb_channel_colors(self) -> None:
        colors = _validate_channel_colors(np.zeros((2, 3, 4)), 0, ("red", "green"))

        assert colors[0] == pytest.approx((1.0, 0.0, 0.0))
        assert colors[1] == pytest.approx((0.0, 0.50196078, 0.0))

    def test_rejects_colors_without_channel_axis(self) -> None:
        with pytest.raises(ValueError, match="axis_order has no 'C' channel axis"):
            _validate_channel_colors(np.zeros((3, 4)), None, ("red",))

    def test_rejects_invalid_color_values(self) -> None:
        with pytest.raises(ValueError, match="not-a-color"):
            _validate_channel_colors(np.zeros((1, 3, 4)), 0, ("not-a-color",))


class TestValidateMaxRenderPixels:
    """Tests for render downsampling limit validation."""

    def test_accepts_none_to_disable_downsampling(self) -> None:
        _validate_max_render_pixels(None)

    def test_rejects_non_positive_limit(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            _validate_max_render_pixels(0)


class TestValidateRenderDownsampling:
    """Tests for render downsampling mode validation."""

    @pytest.mark.parametrize(
        "mode",
        ["stride", "nearest", "bilinear", "bicubic", "none"],
    )
    def test_accepts_supported_modes(self, mode: RenderDownsamplingMode) -> None:
        _validate_render_downsampling(mode)

    def test_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="render_downsampling must be"):
            _validate_render_downsampling(cast("RenderDownsamplingMode", "lanczos"))


class TestFiniteMinMax:
    """Tests for finite display-range extraction."""

    def test_ignores_non_finite_values(self) -> None:
        low, high = _finite_min_max(np.array([np.nan, -2.0, np.inf, 4.0]))

        assert (low, high) == (-2.0, 4.0)

    def test_returns_default_range_when_no_finite_values_exist(self) -> None:
        assert _finite_min_max(np.array([np.nan, np.inf])) == (0.0, 1.0)

    def test_expands_constant_range(self) -> None:
        assert _finite_min_max(np.array([2.0, 2.0])) == (2.0, 3.0)


class TestAbsoluteDisplayRange:
    """Tests for absolute display-range extraction."""

    def test_uses_unsigned_integer_dtype_range(self) -> None:
        image = np.array([10, 20], dtype=np.uint8)

        assert _absolute_display_range(image) == (0.0, 255.0)

    def test_uses_signed_integer_dtype_range(self) -> None:
        image = np.array([-10, 20], dtype=np.int16)

        assert _absolute_display_range(image) == (-32768.0, 32767.0)

    def test_uses_finite_data_range_for_float_images(self) -> None:
        image = np.array([np.nan, 1.5, 4.5], dtype=np.float32)

        assert _absolute_display_range(image) == (1.5, 4.5)


class TestSyncAbsoluteRange:
    """Tests for synchronizing absolute display controls."""

    def test_sets_bounds_to_full_dtype_range(self) -> None:
        controls = _make_channel_controls(0, "Image", (), continuous_update=False)

        _sync_absolute_range(
            controls,
            np.array([10, 20], dtype=np.uint8),
            np.array([10, 20], dtype=np.uint8),
        )

        assert controls.absolute.min == 0.0
        assert controls.absolute.max == 255.0
        assert controls.absolute.value == (0.0, 255.0)

    def test_preserves_user_value_when_slice_changes(self) -> None:
        controls = _make_channel_controls(0, "Image", (), continuous_update=False)
        _sync_absolute_range(
            controls,
            np.array([0, 255], dtype=np.uint8),
            np.array([10, 20], dtype=np.uint8),
        )
        controls.absolute.value = (25.0, 200.0)

        _sync_absolute_range(
            controls,
            np.array([0, 255], dtype=np.uint8),
            np.array([40, 80], dtype=np.uint8),
        )

        assert controls.absolute.value == (25.0, 200.0)

    def test_slice_minmax_checkbox_moves_value_to_current_slice(self) -> None:
        controls = _make_channel_controls(0, "Image", (), continuous_update=False)
        controls.absolute_slice_minmax.value = True

        _sync_absolute_range(
            controls,
            np.array([0, 255], dtype=np.uint8),
            np.array([40, 80], dtype=np.uint8),
        )

        assert controls.absolute.min == 0.0
        assert controls.absolute.max == 255.0
        assert controls.absolute.value == (40.0, 80.0)
        assert controls.absolute.disabled is True

    def test_clamps_user_value_to_new_bounds(self) -> None:
        controls = _make_channel_controls(0, "Image", (), continuous_update=False)
        _sync_absolute_range(
            controls,
            np.array([0.0, 10.0]),
            np.array([0.0, 10.0]),
        )
        controls.absolute.value = (2.0, 8.0)

        _sync_absolute_range(
            controls,
            np.array([4.0, 6.0]),
            np.array([4.0, 6.0]),
        )

        assert controls.absolute.value == (4.0, 6.0)


class TestSetAbsoluteToSliceMinmax:
    """Tests for applying current-slice min/max as a one-shot action."""

    def test_sets_absolute_value_to_current_slice_without_enabling_follow_mode(self) -> None:
        controls = _make_channel_controls(0, "Image", (), continuous_update=False)

        _set_absolute_to_slice_minmax(
            controls,
            np.array([0, 255], dtype=np.uint8),
            np.array([40, 80], dtype=np.uint8),
        )

        assert controls.absolute.min == 0.0
        assert controls.absolute.max == 255.0
        assert controls.absolute.value == (40.0, 80.0)
        assert controls.absolute_slice_minmax.value is False

    def test_uses_full_range_when_current_slice_is_constant(self) -> None:
        controls = _make_channel_controls(0, "Image", (), continuous_update=False)

        _set_absolute_to_slice_minmax(
            controls,
            np.array([0.0, 5.0]),
            np.array([10.0, 10.0]),
        )

        assert controls.absolute.value == (0.0, 5.0)


class TestFinitePercentiles:
    """Tests for percentile display-range extraction."""

    def test_returns_percentiles_of_finite_values(self) -> None:
        low, high = _finite_percentiles(np.array([np.nan, 0.0, 10.0, 20.0]), 0.0, 50.0)

        assert (low, high) == (0.0, 10.0)

    def test_expands_equal_percentiles(self) -> None:
        assert _finite_percentiles(np.array([5.0, 5.0]), 1.0, 99.0) == (5.0, 6.0)

    def test_returns_default_range_when_no_finite_values_exist(self) -> None:
        assert _finite_percentiles(np.array([np.nan, np.inf]), 1.0, 99.0) == (0.0, 1.0)


class TestNormalizeForOverlay:
    """Tests for overlay normalization."""

    def test_normalizes_with_explicit_limits(self) -> None:
        normalized = _normalize_for_overlay(
            np.array([-1.0, 0.0, 0.5, 2.0]),
            _DisplayLimits(vmin=0.0, vmax=1.0),
        )

        np.testing.assert_allclose(normalized, np.array([0.0, 0.0, 0.5, 1.0]))
        assert normalized.dtype == np.float32

    def test_returns_zero_when_limits_are_not_ordered(self) -> None:
        normalized = _normalize_for_overlay(
            np.array([1.0, 2.0]),
            _DisplayLimits(vmin=1.0, vmax=1.0),
        )

        np.testing.assert_allclose(normalized, np.array([0.0, 0.0]))

    def test_uses_finite_data_range_when_limits_are_missing(self) -> None:
        normalized = _normalize_for_overlay(
            np.array([np.nan, 1.0, 3.0]),
            _DisplayLimits(vmin=None, vmax=None),
        )

        np.testing.assert_allclose(normalized, np.array([np.nan, 0.0, 1.0]))


class TestDownsampleForRender:
    """Tests for downsampling large rendered planes."""

    def test_returns_original_image_when_limit_is_none(self) -> None:
        image = np.arange(12).reshape(3, 4)

        downsampled = _downsample_for_render(image, None, "stride")

        assert downsampled is image

    def test_returns_original_image_when_under_limit(self) -> None:
        image = np.arange(12).reshape(3, 4)

        downsampled = _downsample_for_render(image, 12, "stride")

        assert downsampled is image

    def test_returns_original_image_when_mode_is_none(self) -> None:
        image = np.arange(100).reshape(10, 10)

        downsampled = _downsample_for_render(image, 25, "none")

        assert downsampled is image

    def test_stride_samples_large_image_toward_limit(self) -> None:
        image = np.arange(100).reshape(10, 10)

        downsampled = _downsample_for_render(image, 25, "stride")

        np.testing.assert_array_equal(downsampled, image[::2, ::2])

    def test_nearest_samples_to_aspect_preserving_shape(self) -> None:
        image = np.arange(100).reshape(10, 10)

        downsampled = _downsample_for_render(image, 25, "nearest")

        assert downsampled.shape == (5, 5)
        np.testing.assert_array_equal(downsampled[[0, -1], [0, -1]], np.array([0, 99]))

    @pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
    def test_interpolating_modes_return_float32_render_plane(
        self,
        mode: RenderDownsamplingMode,
    ) -> None:
        image = np.arange(100).reshape(10, 10)

        downsampled = _downsample_for_render(image, 25, mode)

        assert downsampled.shape == (5, 5)
        assert downsampled.dtype == np.float32


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

        moved = _move_display_axes_to_end(plane, layout, removed_axis=2)

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

        channel = _extract_channel_plane(plane, layout, 1)

        np.testing.assert_array_equal(channel, plane[:, :, 1])

    def test_orients_non_channel_planes(self) -> None:
        layout = _AxisLayout(
            axis_order="XY",
            channel_axis=None,
            display_axes=(1, 0),
            slider_axes=(),
        )
        plane = np.arange(6).reshape(2, 3)

        channel = _extract_channel_plane(plane, layout, 0)

        np.testing.assert_array_equal(channel, np.moveaxis(plane, (1, 0), (-2, -1)))

    def test_orients_reversed_x_y_planes_as_y_x(self) -> None:
        layout = _AxisLayout(
            axis_order="XY",
            channel_axis=None,
            display_axes=(1, 0),
            slider_axes=(),
        )
        plane = np.arange(6).reshape(2, 3)

        channel = _extract_channel_plane(plane, layout, 0)

        np.testing.assert_array_equal(channel, plane.T)

    def test_extracts_channel_then_orients_reversed_x_y_axes(self) -> None:
        layout = _AxisLayout(
            axis_order="XCY",
            channel_axis=1,
            display_axes=(2, 0),
            slider_axes=(),
        )
        plane = np.arange(24).reshape(3, 4, 2)

        channel = _extract_channel_plane(plane, layout, 1)

        np.testing.assert_array_equal(channel, plane[:, 1, :].T)


class TestDrawPlane:
    """Tests for Matplotlib rendering dispatch."""

    def test_draws_non_channel_plane_with_requested_limits(self) -> None:
        layout = _AxisLayout(
            axis_order="YX",
            channel_axis=None,
            display_axes=(0, 1),
            slider_axes=(),
        )

        fig = _draw_plane(
            plane=np.arange(4).reshape(2, 2),
            layout=layout,
            mode="single",
            selected_channels=(0,),
            channel_names=(),
            channel_colors={},
            limits={0: _DisplayLimits(vmin=0.0, vmax=3.0)},
            figsize=(2.0, 2.0),
            cmap="gray",
            max_render_pixels=None,
            render_downsampling="stride",
        )

        try:
            assert len(fig.axes) == 1
            assert fig.axes[0].images[0].get_clim() == (0.0, 3.0)
        finally:
            plt.close(fig)

    def test_draws_reversed_x_y_plane_as_y_x(self) -> None:
        layout = _AxisLayout(
            axis_order="XY",
            channel_axis=None,
            display_axes=(1, 0),
            slider_axes=(),
        )

        fig = _draw_plane(
            plane=np.arange(6).reshape(2, 3),
            layout=layout,
            mode="single",
            selected_channels=(0,),
            channel_names=(),
            channel_colors={},
            limits={0: _DisplayLimits(vmin=None, vmax=None)},
            figsize=(2.0, 2.0),
            cmap="gray",
            max_render_pixels=None,
            render_downsampling="stride",
        )

        try:
            np.testing.assert_array_equal(
                fig.axes[0].images[0].get_array(), np.arange(6).reshape(2, 3).T
            )
        finally:
            plt.close(fig)

    def test_draws_single_channel_plane_with_title(self) -> None:
        layout = _AxisLayout(
            axis_order="CYX",
            channel_axis=0,
            display_axes=(1, 2),
            slider_axes=(),
        )

        fig = _draw_plane(
            plane=np.arange(8).reshape(2, 2, 2),
            layout=layout,
            mode="single",
            selected_channels=(1,),
            channel_names=("A", "B"),
            channel_colors={1: (0.0, 1.0, 0.0)},
            limits={1: _DisplayLimits(vmin=0.0, vmax=7.0)},
            figsize=(2.0, 2.0),
            cmap="gray",
            max_render_pixels=None,
            render_downsampling="stride",
        )

        try:
            assert fig.axes[0].get_title() == "B"
            np.testing.assert_array_equal(
                fig.axes[0].images[0].get_array(),
                np.array([[4, 5], [6, 7]]),
            )
        finally:
            plt.close(fig)

    def test_draws_overlay_plane_as_rgb_image(self) -> None:
        layout = _AxisLayout(
            axis_order="CYX",
            channel_axis=0,
            display_axes=(1, 2),
            slider_axes=(),
        )

        fig = _draw_plane(
            plane=np.ones((2, 2, 2)),
            layout=layout,
            mode="overlay",
            selected_channels=(0, 1),
            channel_names=("Red", "Green"),
            channel_colors={0: (1.0, 0.0, 0.0), 1: (0.0, 1.0, 0.0)},
            limits={
                0: _DisplayLimits(vmin=0.0, vmax=1.0),
                1: _DisplayLimits(vmin=0.0, vmax=1.0),
            },
            figsize=(2.0, 2.0),
            cmap="gray",
            max_render_pixels=None,
            render_downsampling="stride",
        )

        try:
            assert fig.axes[0].get_title() == "Red, Green"
            image = fig.axes[0].images[0].get_array()
            assert image is not None
            assert image.shape == (2, 2, 3)
        finally:
            plt.close(fig)

    def test_draws_side_by_side_plane_with_one_axis_per_channel(self) -> None:
        layout = _AxisLayout(
            axis_order="CYX",
            channel_axis=0,
            display_axes=(1, 2),
            slider_axes=(),
        )

        fig = _draw_plane(
            plane=np.ones((2, 2, 2)),
            layout=layout,
            mode="side-by-side",
            selected_channels=(0, 1),
            channel_names=("A", "B"),
            channel_colors={0: (1.0, 0.0, 0.0), 1: (0.0, 1.0, 0.0)},
            limits={
                0: _DisplayLimits(vmin=0.0, vmax=1.0),
                1: _DisplayLimits(vmin=0.0, vmax=1.0),
            },
            figsize=(2.0, 2.0),
            cmap="gray",
            max_render_pixels=None,
            render_downsampling="stride",
        )

        try:
            assert [axis.get_title() for axis in fig.axes] == ["A", "B"]
        finally:
            plt.close(fig)


class TestFigureHtml:
    """Tests for serializing figures into notebook HTML."""

    def test_returns_embedded_png_image_html(self) -> None:
        fig = _draw_plane(
            plane=np.arange(4).reshape(2, 2),
            layout=_AxisLayout("YX", None, (0, 1), ()),
            mode="single",
            selected_channels=(0,),
            channel_names=(),
            channel_colors={},
            limits={0: _DisplayLimits(vmin=None, vmax=None)},
            figsize=(1.0, 1.0),
            cmap="gray",
            max_render_pixels=None,
            render_downsampling="stride",
        )

        try:
            html = _figure_html(fig)
        finally:
            plt.close(fig)

        assert html.startswith('<img alt="Rendered image layer" src="data:image/png;base64,')
        assert 'style="display:block; max-width:100%; height:auto; overflow:hidden;"' in html

    def test_saves_without_tight_bbox_options(self, mocker: MockerFixture) -> None:
        fig = plt.figure()
        savefig = mocker.patch.object(
            fig,
            "savefig",
            side_effect=lambda buffer, **_kwargs: buffer.write(b"png"),
        )

        try:
            _figure_html(fig)
        finally:
            plt.close(fig)

        kwargs = savefig.call_args.kwargs
        assert kwargs["format"] == "png"
        assert kwargs["dpi"] == 120
        assert "bbox_inches" not in kwargs
        assert "pad_inches" not in kwargs


class TestSliderStep:
    """Tests for display-range slider step sizing."""

    def test_returns_fraction_of_positive_span(self) -> None:
        assert _slider_step(0.0, 10.0) == 0.02

    def test_returns_default_for_non_positive_span(self) -> None:
        assert _slider_step(1.0, 1.0) == 1.0
