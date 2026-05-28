"""ipywidgets control construction and synchronization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import ipywidgets as widgets
import matplotlib.colors as mcolors
import numpy as np

from nb_nd_viewer._ranges import absolute_display_range, clamp, finite_min_max, slider_step
from nb_nd_viewer._types import _AxisLayout, _ChannelControls

if TYPE_CHECKING:
    from nb_nd_viewer._types import ImageArray

WidgetT = TypeVar("WidgetT", bound=widgets.Widget)
_CONTROL_WIDTH = "320px"
_CONTROL_DESCRIPTION_WIDTH = "70px"


def make_axis_sliders(
    array: ImageArray,
    layout: _AxisLayout,
    *,
    continuous_update: bool,
) -> dict[int, widgets.IntSlider]:
    """Return one slider for each non-displayed stack axis."""
    return {
        axis: _with_tooltip(
            widgets.IntSlider(
                value=0,
                min=0,
                max=array.shape[axis] - 1,
                step=1,
                description=layout.axis_order[axis],
                continuous_update=continuous_update,
                layout=_control_layout(),
                style=_description_style(),
            ),
            f"Select the {layout.axis_order[axis]} layer shown in the image plane.",
        )
        for axis in layout.slider_axes
    }


def make_channel_selector(channel_names: tuple[str, ...]) -> widgets.SelectMultiple:
    """Return the channel picker used by all channel display modes."""
    return _with_tooltip(
        widgets.SelectMultiple(
            options=[(name, index) for index, name in enumerate(channel_names)],
            value=(0,),
            description="Channels",
            rows=min(max(len(channel_names), 1), 8),
            layout=_control_layout(),
            style=_description_style(),
        ),
        "Select every channel that should be visible in overlay or side-by-side mode.",
    )


def make_display_mode() -> widgets.ToggleButtons:
    """Return the channel display mode selector."""
    return _with_tooltip(
        widgets.ToggleButtons(
            options=[
                ("Single", "single"),
                ("Overlay", "overlay"),
                ("Side-by-side", "side-by-side"),
            ],
            value="single",
            description="Mode",
            tooltips=[
                "Show the first selected channel.",
                "Blend all selected channels into one RGB image.",
                "Show all selected channels in separate panels.",
            ],
            layout=_control_layout(),
            style=_description_style(),
        ),
        "Choose how selected channels are rendered.",
    )


def make_channel_controls(
    index: int,
    name: str,
    channel_colors: tuple[tuple[float, float, float], ...],
    *,
    continuous_update: bool,
) -> _ChannelControls:
    """Return color and intensity controls for one rendered channel."""
    color = None
    if channel_colors:
        color = _with_tooltip(
            widgets.ColorPicker(
                concise=False,
                description="Color",
                value=mcolors.to_hex(channel_colors[index]),
                layout=_control_layout(),
                style=_description_style(),
            ),
            f"Display color for {name}.",
        )
    mode = _with_tooltip(
        widgets.ToggleButtons(
            options=[
                ("Absolute", "absolute"),
                ("Percentile", "percentile"),
            ],
            value="absolute",
            description="Range",
            tooltips=[
                "Use absolute display thresholds for this channel.",
                "Use percentile display thresholds for this channel.",
            ],
            layout=_control_layout(),
            style=_description_style(),
        ),
        f"Choose display range mode for {name}.",
    )
    absolute = _with_tooltip(
        widgets.FloatRangeSlider(
            value=(0.0, 1.0),
            min=0.0,
            max=1.0,
            step=0.01,
            description="Min/max",
            continuous_update=continuous_update,
            readout=True,
            readout_format=".3g",
            layout=_control_layout(),
            style=_description_style(),
        ),
        f"Absolute display thresholds for {name}; source image data is unchanged.",
    )
    absolute_slice_minmax = _with_tooltip(
        widgets.Checkbox(
            value=False,
            description="Slice min/max",
            indent=False,
            layout=_control_layout(),
        ),
        f"Move {name}'s absolute thresholds to the current slice min/max.",
    )
    absolute_slice_button = _with_tooltip(
        widgets.Button(
            description="Set slice min/max",
            button_style="",
            layout=_control_layout(),
        ),
        f"Set {name}'s absolute thresholds to the current slice min/max.",
    )
    percentile = _with_tooltip(
        widgets.FloatRangeSlider(
            value=(1.0, 99.0),
            min=0.0,
            max=100.0,
            step=0.5,
            description="Percent",
            continuous_update=continuous_update,
            readout=True,
            readout_format=".1f",
            layout=_control_layout(),
            style=_description_style(),
        ),
        f"Percentile display thresholds for {name}; source image data is unchanged.",
    )
    header = widgets.HTML(value=f'<strong title="Display settings for {name}.">{name}</strong>')
    box = widgets.VBox([], layout=widgets.Layout(grid_gap="6px"))
    controls = _ChannelControls(
        color=color,
        mode=mode,
        absolute=absolute,
        absolute_slice_minmax=absolute_slice_minmax,
        absolute_slice_button=absolute_slice_button,
        percentile=percentile,
        box=box,
    )
    _sync_channel_box(controls, header)

    def on_mode_change(change: dict[str, object]) -> None:
        if change["name"] == "value":
            _sync_channel_box(controls, header)

    mode.observe(on_mode_change, names="value")
    return controls


def sync_visible_settings(
    settings_box: widgets.VBox,
    settings: dict[int, _ChannelControls],
    selected_channels: tuple[int, ...],
) -> None:
    """Show settings only for channels currently selected for rendering."""
    settings_box.children = tuple(
        settings[channel_index].box for channel_index in selected_channels
    )


def _sync_channel_box(controls: _ChannelControls, header: widgets.HTML) -> None:
    """Show only the controls relevant to one channel's active range mode."""
    children: list[widgets.Widget] = [header]
    if controls.color is not None:
        children.append(controls.color)
    children.append(controls.mode)
    if controls.mode.value == "absolute":
        children.append(controls.absolute_slice_minmax)
        children.append(controls.absolute_slice_button)
        children.append(controls.absolute)
    elif controls.mode.value == "percentile":
        children.append(controls.percentile)
    controls.absolute.disabled = bool(controls.absolute_slice_minmax.value)
    controls.box.children = tuple(children)


def sync_absolute_range(
    controls: _ChannelControls,
    full_image: ImageArray,
    current_slice: ImageArray,
) -> None:
    """Update absolute slider bounds while preserving user-selected values."""
    low, high = absolute_display_range(full_image)
    old_low = float(controls.absolute.min)
    old_high = float(controls.absolute.max)
    value_low, value_high = sorted(controls.absolute.value)

    if controls.absolute_slice_minmax.value:
        value_low, value_high = _slice_minmax_within_range(current_slice, low, high)
    elif np.isclose(value_low, old_low) and np.isclose(value_high, old_high):
        value_low, value_high = low, high
    else:
        value_low = clamp(float(value_low), low, high)
        value_high = clamp(float(value_high), low, high)
    if value_low == value_high:
        value_low, value_high = low, high

    _apply_absolute_slider_range(
        controls,
        bounds=(low, high),
        value=(value_low, value_high),
        disabled=bool(controls.absolute_slice_minmax.value),
    )


def set_absolute_to_slice_minmax(
    controls: _ChannelControls,
    full_image: ImageArray,
    current_slice: ImageArray,
) -> None:
    """Move the absolute slider value to the current slice min/max once."""
    low, high = absolute_display_range(full_image)
    value_low, value_high = _slice_minmax_within_range(current_slice, low, high)
    _apply_absolute_slider_range(
        controls,
        bounds=(low, high),
        value=(value_low, value_high),
        disabled=bool(controls.absolute.disabled),
    )


def _apply_absolute_slider_range(
    controls: _ChannelControls,
    *,
    bounds: tuple[float, float],
    value: tuple[float, float],
    disabled: bool,
) -> None:
    """Apply bounds, step, value, and enabled state to the absolute slider."""
    low, high = bounds
    with controls.absolute.hold_trait_notifications():
        controls.absolute.min = low
        controls.absolute.max = high
        controls.absolute.step = slider_step(low, high)
        controls.absolute.value = value
        controls.absolute.disabled = disabled


def _slice_minmax_within_range(
    current_slice: ImageArray,
    low: float,
    high: float,
) -> tuple[float, float]:
    """Return current-slice finite min/max clamped to absolute slider bounds."""
    value_low, value_high = finite_min_max(current_slice)
    value_low = clamp(float(value_low), low, high)
    value_high = clamp(float(value_high), low, high)
    if value_low == value_high:
        return low, high
    return value_low, value_high


def _with_tooltip(widget: WidgetT, tooltip: str) -> WidgetT:
    """Attach a hover tooltip to an ipywidgets control."""
    widget.set_trait("tooltip", tooltip)
    return widget


def _control_layout() -> widgets.Layout:
    """Return the standard layout for primary controls."""
    return widgets.Layout(width=_CONTROL_WIDTH)


def _description_style() -> dict[str, str]:
    """Return the standard description-width style for labeled controls."""
    return {"description_width": _CONTROL_DESCRIPTION_WIDTH}


def control_section(title: str, children: list[widgets.Widget]) -> widgets.VBox:
    """Return a compact titled group of controls."""
    header = widgets.HTML(value=f'<strong title="{title} controls.">{title}</strong>')
    body = widgets.VBox(children, layout=widgets.Layout(grid_gap="6px"))
    return widgets.VBox([header, body], layout=widgets.Layout(grid_gap="4px"))
