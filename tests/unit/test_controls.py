"""Unit tests for ipywidgets control helpers."""

from __future__ import annotations

import numpy as np

from nb_nd_viewer._controls import (
    make_axis_sliders,
    make_channel_controls,
    set_absolute_to_slice_minmax,
    sync_absolute_range,
)
from nb_nd_viewer._types import _AxisLayout


class TestMakeAxisSliders:
    """Tests for stack-axis slider construction."""

    def test_uses_requested_continuous_update_setting(self) -> None:
        layout = _AxisLayout(
            axis_order="ZYX",
            channel_axis=None,
            display_axes=(1, 2),
            slider_axes=(0,),
        )

        sliders = make_axis_sliders(
            np.zeros((2, 3, 4)),
            layout,
            continuous_update=True,
        )

        assert sliders[0].continuous_update is True


class TestMakeChannelControls:
    """Tests for channel control construction."""

    def test_uses_requested_continuous_update_setting_for_range_sliders(self) -> None:
        controls = make_channel_controls(
            0,
            "Image",
            (),
            continuous_update=True,
        )

        assert controls.absolute.continuous_update is True
        assert controls.percentile.continuous_update is True

    def test_defaults_to_absolute_range_without_auto_option(self) -> None:
        controls = make_channel_controls(
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
        controls = make_channel_controls(
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
        controls = make_channel_controls(
            0,
            "Image",
            (),
            continuous_update=False,
        )

        controls.mode.value = "percentile"

        assert controls.percentile in controls.box.children
        assert controls.absolute not in controls.box.children
        assert controls.absolute_slice_minmax not in controls.box.children


class TestSyncAbsoluteRange:
    """Tests for synchronizing absolute display controls."""

    def test_sets_bounds_to_full_dtype_range(self) -> None:
        controls = make_channel_controls(0, "Image", (), continuous_update=False)

        sync_absolute_range(
            controls,
            np.array([10, 20], dtype=np.uint8),
            np.array([10, 20], dtype=np.uint8),
        )

        assert controls.absolute.min == 0.0
        assert controls.absolute.max == 255.0
        assert controls.absolute.value == (0.0, 255.0)

    def test_preserves_user_value_when_slice_changes(self) -> None:
        controls = make_channel_controls(0, "Image", (), continuous_update=False)
        sync_absolute_range(
            controls,
            np.array([0, 255], dtype=np.uint8),
            np.array([10, 20], dtype=np.uint8),
        )
        controls.absolute.value = (25.0, 200.0)

        sync_absolute_range(
            controls,
            np.array([0, 255], dtype=np.uint8),
            np.array([40, 80], dtype=np.uint8),
        )

        assert controls.absolute.value == (25.0, 200.0)

    def test_slice_minmax_checkbox_moves_value_to_current_slice(self) -> None:
        controls = make_channel_controls(0, "Image", (), continuous_update=False)
        controls.absolute_slice_minmax.value = True

        sync_absolute_range(
            controls,
            np.array([0, 255], dtype=np.uint8),
            np.array([40, 80], dtype=np.uint8),
        )

        assert controls.absolute.min == 0.0
        assert controls.absolute.max == 255.0
        assert controls.absolute.value == (40.0, 80.0)
        assert controls.absolute.disabled is True

    def test_clamps_user_value_to_new_bounds(self) -> None:
        controls = make_channel_controls(0, "Image", (), continuous_update=False)
        sync_absolute_range(
            controls,
            np.array([0.0, 10.0]),
            np.array([0.0, 10.0]),
        )
        controls.absolute.value = (2.0, 8.0)

        sync_absolute_range(
            controls,
            np.array([4.0, 6.0]),
            np.array([4.0, 6.0]),
        )

        assert controls.absolute.value == (4.0, 6.0)


class TestSetAbsoluteToSliceMinmax:
    """Tests for applying current-slice min/max as a one-shot action."""

    def test_sets_absolute_value_to_current_slice_without_enabling_follow_mode(self) -> None:
        controls = make_channel_controls(0, "Image", (), continuous_update=False)

        set_absolute_to_slice_minmax(
            controls,
            np.array([0, 255], dtype=np.uint8),
            np.array([40, 80], dtype=np.uint8),
        )

        assert controls.absolute.min == 0.0
        assert controls.absolute.max == 255.0
        assert controls.absolute.value == (40.0, 80.0)
        assert controls.absolute_slice_minmax.value is False

    def test_uses_full_range_when_current_slice_is_constant(self) -> None:
        controls = make_channel_controls(0, "Image", (), continuous_update=False)

        set_absolute_to_slice_minmax(
            controls,
            np.array([0.0, 5.0]),
            np.array([10.0, 10.0]),
        )

        assert controls.absolute.value == (0.0, 5.0)
