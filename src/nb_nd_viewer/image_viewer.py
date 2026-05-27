"""Interactive Matplotlib viewers for exploratory image arrays."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import ipywidgets as widgets
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, NDArray

DisplayMode = Literal["single", "overlay", "side-by-side"]
RenderDownsamplingMode = Literal["stride", "nearest", "bilinear", "bicubic", "none"]
WidgetT = TypeVar("WidgetT", bound=widgets.Widget)
_DISPLAY_NDIMS = 2
_DEFAULT_DPI = 120
_CONTROL_WIDTH = "320px"
_CONTROL_DESCRIPTION_WIDTH = "70px"
_DEFAULT_CHANNEL_COLORS = (
    "red",
    "lime",
    "blue",
    "magenta",
    "cyan",
    "yellow",
    "orange",
    "white",
)


@dataclass(frozen=True)
class _AxisLayout:
    """Resolved axis roles for an image array.

    ``display_axes`` stores the render row axis followed by the render column
    axis.
    """

    axis_order: str
    channel_axis: int | None
    display_axes: tuple[int, int]
    slider_axes: tuple[int, ...]


@dataclass(frozen=True)
class _DisplayLimits:
    """Resolved display limits for one rendered image plane."""

    vmin: float | None
    vmax: float | None


@dataclass(frozen=True)
class _ChannelControls:
    """Controls that affect one channel's rendered color and intensity."""

    color: widgets.ColorPicker | None
    mode: widgets.ToggleButtons
    absolute: widgets.FloatRangeSlider
    absolute_slice_minmax: widgets.Checkbox
    absolute_slice_button: widgets.Button
    percentile: widgets.FloatRangeSlider
    box: widgets.VBox


def view_image_layers(  # noqa: C901, PLR0913, PLR0915
    image: ArrayLike,
    axis_order: str,
    channel_names: Sequence[str] | None = None,
    channel_colors: Sequence[str] | None = None,
    figsize: tuple[float, float] = (6, 6),
    cmap: str = "gray",
    *,
    continuous_update: bool = False,
    max_render_pixels: int | None = 1_000_000,
    render_downsampling: RenderDownsamplingMode = "stride",
) -> widgets.Widget:
    """Display an interactive layer viewer for 2D, stack, and channel images.

    ``axis_order`` must contain one character per image dimension. Use ``C`` for
    the optional channel axis. When both ``Y`` and ``X`` are present, they are
    displayed as the image plane with ``Y`` as rows and ``X`` as columns,
    regardless of their position in ``axis_order``. Other non-channel axes
    become layer sliders. If either ``Y`` or ``X`` is missing, the last two
    non-channel axes are displayed in their listed order. For example,
    ``axis_order="ZCYX"`` and ``axis_order="ZCXY"`` both create a ``Z`` slider,
    treat ``C`` as the channel axis, and display ``Y``/``X``.

    Channel images can be shown as one channel, an RGB overlay, or side-by-side
    panels. Overlay and side-by-side views render every selected channel, so
    select all channels that should be visible. Each selected channel gets its
    own color and intensity controls. Intensity thresholds are display contrast
    limits only: they clip the rendered view and never mask or modify the source
    image data.

    Parameters
    ----------
    image
        Image-like array to inspect.
    axis_order
        Axis labels matching ``image.ndim``. ``C`` marks the optional channel
        axis. ``Y`` and ``X`` are displayed as rows and columns when both are
        present; otherwise the last two non-channel axes are displayed.
    channel_names
        Optional display names for the channel axis.
    channel_colors
        Optional Matplotlib-compatible colors used for channel rendering.
    figsize
        Base Matplotlib figure size before responsive notebook scaling.
    cmap
        Colormap for non-channel images.
    continuous_update
        If ``True``, redraw while layer and display-range sliders are dragged.
        If ``False``, redraw when a slider drag is released.
    max_render_pixels
        Maximum number of pixels sent to Matplotlib for each rendered 2D plane.
        Larger planes are downsampled before rendering. Use ``None`` to render
        every source pixel.
    render_downsampling
        Downsampling mode for planes larger than ``max_render_pixels``.
        ``"stride"`` samples every Nth row and column; ``"nearest"``,
        ``"bilinear"``, and ``"bicubic"`` resample to a smaller render plane;
        ``"none"`` disables render downsampling.

    Returns
    -------
    ipywidgets.Widget
        A widget containing controls and a responsive image display.
    """
    array = np.asarray(image)
    layout = _validate_layout(array, axis_order)
    _validate_max_render_pixels(max_render_pixels)
    _validate_render_downsampling(render_downsampling)
    names = _validate_channel_names(array, layout.channel_axis, channel_names)
    colors = _validate_channel_colors(array, layout.channel_axis, channel_colors)

    axis_sliders = _make_axis_sliders(array, layout, continuous_update=continuous_update)
    has_channels = layout.channel_axis is not None
    channel_selector = _make_channel_selector(names) if has_channels else None
    display_mode = _make_display_mode() if has_channels else None

    setting_names = names if has_channels else ("Image",)
    setting_colors = colors if has_channels else ()
    settings = {
        index: _make_channel_controls(
            index,
            setting_names[index],
            setting_colors,
            continuous_update=continuous_update,
        )
        for index in range(len(setting_names))
    }
    settings_box = widgets.VBox([], layout=widgets.Layout(grid_gap="8px"))
    image_html = widgets.HTML(
        layout=widgets.Layout(
            flex="1 1 420px",
            max_width="100%",
            min_width="280px",
            overflow="hidden",
        ),
    )
    updating_controls = False

    def current_plane() -> tuple[NDArray[Any], _AxisLayout]:
        indexer: list[int | slice] = [slice(None)] * array.ndim
        for axis, slider in axis_sliders.items():
            indexer[axis] = slider.value
        return np.asarray(array[tuple(indexer)]), _remove_axes_from_layout(
            layout,
            tuple(axis_sliders),
        )

    def selected_channels() -> tuple[int, ...]:
        if channel_selector is None:
            return (0,)
        return tuple(int(index) for index in channel_selector.value) or (0,)

    def current_display_mode() -> DisplayMode:
        if display_mode is None:
            return "single"
        return display_mode.value

    def channel_plane(channel_index: int) -> NDArray[Any]:
        plane, plane_layout = current_plane()
        if not has_channels:
            return _move_display_axes_to_end(plane, plane_layout, None)
        return _extract_channel_plane(plane, plane_layout, channel_index)

    def channel_volume(channel_index: int) -> NDArray[Any]:
        if not has_channels:
            return array
        channel_axis = cast("int", layout.channel_axis)
        indexer: list[int | slice] = [slice(None)] * array.ndim
        indexer[channel_axis] = channel_index
        return np.asarray(array[tuple(indexer)])

    def display_limits(channel_index: int) -> _DisplayLimits:
        controls = settings[channel_index]
        mode = controls.mode.value
        if mode == "absolute":
            low, high = sorted(controls.absolute.value)
            return _DisplayLimits(vmin=float(low), vmax=float(high))

        low_pct, high_pct = sorted(controls.percentile.value)
        low, high = _finite_percentiles(channel_plane(channel_index), low_pct, high_pct)
        return _DisplayLimits(vmin=low, vmax=high)

    def redraw() -> None:
        nonlocal updating_controls
        if updating_controls:
            return
        updating_controls = True
        try:
            _sync_visible_settings(settings_box, settings, selected_channels())
            for channel_index in selected_channels():
                _sync_absolute_range(
                    settings[channel_index],
                    channel_volume(channel_index),
                    channel_plane(channel_index),
                )
        finally:
            updating_controls = False

        plane, plane_layout = current_plane()
        channels = selected_channels()
        limits = {channel_index: display_limits(channel_index) for channel_index in channels}
        channel_rgb: dict[int, tuple[float, float, float]] = {}
        for channel_index in channels:
            color_picker = settings[channel_index].color
            if color_picker is not None:
                channel_rgb[channel_index] = mcolors.to_rgb(color_picker.value)
        fig = _draw_plane(
            plane=plane,
            layout=plane_layout,
            mode=current_display_mode(),
            selected_channels=channels,
            channel_names=names,
            channel_colors=channel_rgb,
            limits=limits,
            figsize=figsize,
            cmap=cmap,
            max_render_pixels=max_render_pixels,
            render_downsampling=render_downsampling,
        )
        image_html.value = _figure_html(fig)
        plt.close(fig)

    def on_control_change(change: dict[str, Any]) -> None:
        if change["name"] == "value":
            redraw()

    watched_controls: list[widgets.Widget] = [*axis_sliders.values()]
    if channel_selector is not None:
        watched_controls.append(channel_selector)
    if display_mode is not None:
        watched_controls.append(display_mode)
    for controls in settings.values():
        watched_controls.extend(
            control
            for control in (
                controls.color,
                controls.mode,
                controls.absolute,
                controls.absolute_slice_minmax,
                controls.percentile,
            )
            if control is not None
        )
    for control in watched_controls:
        control.observe(on_control_change, names="value")

    def on_slice_minmax_click(channel_index: int) -> None:
        _set_absolute_to_slice_minmax(
            settings[channel_index],
            channel_volume(channel_index),
            channel_plane(channel_index),
        )
        redraw()

    for channel_index, controls in settings.items():
        controls.absolute_slice_button.on_click(
            lambda _button, channel_index=channel_index: on_slice_minmax_click(channel_index),
        )

    control_sections: list[widgets.Widget] = []
    if axis_sliders:
        control_sections.append(_control_section("Layers", [*axis_sliders.values()]))
    if channel_selector is not None and display_mode is not None:
        control_sections.append(_control_section("Channels", [display_mode, channel_selector]))
    control_sections.append(_control_section("Display settings", [settings_box]))

    control_panel = widgets.VBox(
        control_sections,
        layout=widgets.Layout(
            grid_gap="12px",
            padding="10px",
            width="360px",
            flex="0 0 360px",
        ),
    )
    widget = widgets.HBox(
        [control_panel, image_html],
        layout=widgets.Layout(
            align_items="flex-start",
            grid_gap="12px",
            flex_flow="row nowrap",
            max_width="100%",
            overflow="hidden",
            width="100%",
        ),
    )
    redraw()
    return widget


def _make_axis_sliders(
    array: NDArray[Any],
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


def _make_channel_selector(channel_names: tuple[str, ...]) -> widgets.SelectMultiple:
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


def _make_display_mode() -> widgets.ToggleButtons:
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


def _make_channel_controls(
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

    def on_mode_change(change: dict[str, Any]) -> None:
        if change["name"] == "value":
            _sync_channel_box(controls, header)

    mode.observe(on_mode_change, names="value")
    return controls


def _sync_visible_settings(
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


def _sync_absolute_range(
    controls: _ChannelControls,
    full_image: NDArray[Any],
    current_slice: NDArray[Any],
) -> None:
    """Update absolute slider bounds while preserving user-selected values."""
    low, high = _absolute_display_range(full_image)
    old_low = float(controls.absolute.min)
    old_high = float(controls.absolute.max)
    value_low, value_high = sorted(controls.absolute.value)

    if controls.absolute_slice_minmax.value:
        value_low, value_high = _slice_minmax_within_range(current_slice, low, high)
    elif np.isclose(value_low, old_low) and np.isclose(value_high, old_high):
        value_low, value_high = low, high
    else:
        value_low = _clamp(float(value_low), low, high)
        value_high = _clamp(float(value_high), low, high)
    if value_low == value_high:
        value_low, value_high = low, high

    _apply_absolute_slider_range(
        controls,
        bounds=(low, high),
        value=(value_low, value_high),
        disabled=bool(controls.absolute_slice_minmax.value),
    )


def _set_absolute_to_slice_minmax(
    controls: _ChannelControls,
    full_image: NDArray[Any],
    current_slice: NDArray[Any],
) -> None:
    """Move the absolute slider value to the current slice min/max once."""
    low, high = _absolute_display_range(full_image)
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
        controls.absolute.step = _slider_step(low, high)
        controls.absolute.value = value
        controls.absolute.disabled = disabled


def _slice_minmax_within_range(
    current_slice: NDArray[Any],
    low: float,
    high: float,
) -> tuple[float, float]:
    """Return current-slice finite min/max clamped to absolute slider bounds."""
    value_low, value_high = _finite_min_max(current_slice)
    value_low = _clamp(float(value_low), low, high)
    value_high = _clamp(float(value_high), low, high)
    if value_low == value_high:
        return low, high
    return value_low, value_high


def _validate_layout(array: NDArray[Any], axis_order: str) -> _AxisLayout:
    """Validate axis metadata and resolve display and slider axes."""
    normalized = axis_order.upper()
    if len(normalized) != array.ndim:
        msg = f"axis_order has length {len(normalized)}, but image has {array.ndim} dimensions."
        raise ValueError(msg)
    if any(size == 0 for size in array.shape):
        msg = "image axes must be non-empty."
        raise ValueError(msg)
    if normalized.count("C") > 1:
        msg = "axis_order may contain at most one channel axis marked with 'C'."
        raise ValueError(msg)
    duplicate_labels = sorted({label for label in normalized if normalized.count(label) > 1})
    if duplicate_labels:
        labels = "', '".join(duplicate_labels)
        msg = f"axis_order labels must be unique; duplicated labels: '{labels}'."
        raise ValueError(msg)

    channel_axis = normalized.find("C")
    channel_axis_or_none = None if channel_axis == -1 else channel_axis
    non_channel_axes = tuple(axis for axis in range(array.ndim) if axis != channel_axis_or_none)
    if len(non_channel_axes) < _DISPLAY_NDIMS:
        msg = "image needs at least two non-channel axes to display a 2D plane."
        raise ValueError(msg)

    if "Y" in normalized and "X" in normalized:
        y_axis = normalized.index("Y")
        x_axis = normalized.index("X")
        display_axes = (y_axis, x_axis)
        slider_axes = tuple(axis for axis in non_channel_axes if axis not in {y_axis, x_axis})
    else:
        display_axes = (non_channel_axes[-2], non_channel_axes[-1])
        slider_axes = non_channel_axes[:-2]
    return _AxisLayout(
        axis_order=normalized,
        channel_axis=channel_axis_or_none,
        display_axes=display_axes,
        slider_axes=slider_axes,
    )


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


def _control_section(title: str, children: list[widgets.Widget]) -> widgets.VBox:
    """Return a compact titled group of controls."""
    header = widgets.HTML(value=f'<strong title="{title} controls.">{title}</strong>')
    body = widgets.VBox(children, layout=widgets.Layout(grid_gap="6px"))
    return widgets.VBox([header, body], layout=widgets.Layout(grid_gap="4px"))


def _remove_axes_from_layout(layout: _AxisLayout, removed_axes: tuple[int, ...]) -> _AxisLayout:
    """Return a layout adjusted after integer indexing removes axes."""
    if any(axis in removed_axes for axis in layout.display_axes):
        msg = "display axes were removed while preparing the current plane."
        raise ValueError(msg)

    def adjust_axis(axis: int | None) -> int | None:
        if axis is None:
            return None
        return axis - sum(removed_axis < axis for removed_axis in removed_axes)

    adjusted_display_axes = tuple(
        adjusted_axis
        for axis in layout.display_axes
        if (adjusted_axis := adjust_axis(axis)) is not None
    )
    if len(adjusted_display_axes) != _DISPLAY_NDIMS:
        msg = "display axes were removed while preparing the current plane."
        raise ValueError(msg)

    remaining_axis_order = "".join(
        name for axis, name in enumerate(layout.axis_order) if axis not in removed_axes
    )
    return _AxisLayout(
        axis_order=remaining_axis_order,
        channel_axis=adjust_axis(layout.channel_axis),
        display_axes=(adjusted_display_axes[0], adjusted_display_axes[1]),
        slider_axes=(),
    )


def _validate_channel_names(
    array: NDArray[Any],
    channel_axis: int | None,
    channel_names: Sequence[str] | None,
) -> tuple[str, ...]:
    """Return display names matching the channel axis."""
    if channel_axis is None:
        if channel_names is not None:
            msg = "channel_names were provided, but axis_order has no 'C' channel axis."
            raise ValueError(msg)
        return ()

    channel_count = array.shape[channel_axis]
    if channel_names is None:
        return tuple(f"Channel {index}" for index in range(channel_count))
    if len(channel_names) != channel_count:
        msg = f"channel_names has length {len(channel_names)}, expected {channel_count}."
        raise ValueError(msg)
    return tuple(str(name) for name in channel_names)


def _validate_channel_colors(
    array: NDArray[Any],
    channel_axis: int | None,
    channel_colors: Sequence[str] | None,
) -> tuple[tuple[float, float, float], ...]:
    """Return RGB colors matching the channel axis."""
    if channel_axis is None:
        if channel_colors is not None:
            msg = "channel_colors were provided, but axis_order has no 'C' channel axis."
            raise ValueError(msg)
        return ()

    channel_count = array.shape[channel_axis]
    raw_colors = channel_colors or _DEFAULT_CHANNEL_COLORS
    if len(raw_colors) < channel_count:
        msg = f"channel_colors has length {len(raw_colors)}, expected at least {channel_count}."
        raise ValueError(msg)
    return tuple(mcolors.to_rgb(color) for color in raw_colors[:channel_count])


def _validate_max_render_pixels(max_render_pixels: int | None) -> None:
    """Validate the optional render downsampling limit."""
    if max_render_pixels is None:
        return
    if max_render_pixels < 1:
        msg = "max_render_pixels must be a positive integer or None."
        raise ValueError(msg)


def _validate_render_downsampling(render_downsampling: RenderDownsamplingMode) -> None:
    """Validate the render downsampling mode."""
    if render_downsampling in {"stride", "nearest", "bilinear", "bicubic", "none"}:
        return
    msg = "render_downsampling must be 'stride', 'nearest', 'bilinear', 'bicubic', or 'none'."
    raise ValueError(msg)


def _draw_plane(  # noqa: PLR0913
    *,
    plane: NDArray[Any],
    layout: _AxisLayout,
    mode: DisplayMode,
    selected_channels: tuple[int, ...],
    channel_names: tuple[str, ...],
    channel_colors: dict[int, tuple[float, float, float]],
    limits: dict[int, _DisplayLimits],
    figsize: tuple[float, float],
    cmap: str,
    max_render_pixels: int | None,
    render_downsampling: RenderDownsamplingMode,
) -> Figure:
    """Draw the selected plane and return the Matplotlib figure."""
    if layout.channel_axis is None:
        image_2d = _move_display_axes_to_end(plane, layout, None)
        fig, ax = plt.subplots(figsize=figsize)
        _imshow(
            ax,
            image_2d,
            cmap,
            limits[0],
            max_render_pixels=max_render_pixels,
            render_downsampling=render_downsampling,
        )
        _clean_axis(ax)
        _tight_figure(fig)
        return fig

    if mode == "overlay":
        return _draw_overlay(
            plane=plane,
            layout=layout,
            selected_channels=selected_channels,
            channel_names=channel_names,
            channel_colors=channel_colors,
            limits=limits,
            figsize=figsize,
            max_render_pixels=max_render_pixels,
            render_downsampling=render_downsampling,
        )
    if mode == "side-by-side":
        return _draw_side_by_side(
            plane=plane,
            layout=layout,
            selected_channels=selected_channels,
            channel_names=channel_names,
            channel_colors=channel_colors,
            limits=limits,
            figsize=figsize,
            max_render_pixels=max_render_pixels,
            render_downsampling=render_downsampling,
        )

    channel_index = selected_channels[0]
    fig, ax = plt.subplots(figsize=figsize)
    _imshow(
        ax,
        _extract_channel_plane(plane, layout, channel_index),
        _channel_cmap(channel_colors[channel_index]),
        limits[channel_index],
        max_render_pixels=max_render_pixels,
        render_downsampling=render_downsampling,
    )
    ax.set_title(channel_names[channel_index])
    _clean_axis(ax)
    _tight_figure(fig)
    return fig


def _draw_overlay(  # noqa: PLR0913
    *,
    plane: NDArray[Any],
    layout: _AxisLayout,
    selected_channels: tuple[int, ...],
    channel_names: tuple[str, ...],
    channel_colors: dict[int, tuple[float, float, float]],
    limits: dict[int, _DisplayLimits],
    figsize: tuple[float, float],
    max_render_pixels: int | None,
    render_downsampling: RenderDownsamplingMode,
) -> Figure:
    """Draw selected channels as a blended RGB overlay."""
    first = _extract_channel_plane(plane, layout, selected_channels[0])
    first = _downsample_for_render(first, max_render_pixels, render_downsampling)
    rgb = np.zeros((*first.shape, 3), dtype=np.float32)
    for channel_index in selected_channels:
        image = _extract_channel_plane(plane, layout, channel_index)
        channel = _normalize_for_overlay(
            _downsample_for_render(image, max_render_pixels, render_downsampling),
            limits[channel_index],
        )
        color = np.asarray(channel_colors[channel_index], dtype=np.float32)
        rgb += channel[..., np.newaxis] * color

    fig, ax = plt.subplots(figsize=figsize)
    np.clip(rgb, 0.0, 1.0, out=rgb)
    ax.imshow(rgb)
    ax.set_title(", ".join(channel_names[index] for index in selected_channels))
    _clean_axis(ax)
    _tight_figure(fig)
    return fig


def _draw_side_by_side(  # noqa: PLR0913
    *,
    plane: NDArray[Any],
    layout: _AxisLayout,
    selected_channels: tuple[int, ...],
    channel_names: tuple[str, ...],
    channel_colors: dict[int, tuple[float, float, float]],
    limits: dict[int, _DisplayLimits],
    figsize: tuple[float, float],
    max_render_pixels: int | None,
    render_downsampling: RenderDownsamplingMode,
) -> Figure:
    """Draw selected channels in separate panels with per-channel limits."""
    width, height = figsize
    fig, axes = plt.subplots(
        1,
        len(selected_channels),
        figsize=(max(width, 3.0 * len(selected_channels)), height),
        squeeze=False,
    )
    for ax, channel_index in zip(axes[0], selected_channels, strict=True):
        _imshow(
            ax,
            _extract_channel_plane(plane, layout, channel_index),
            _channel_cmap(channel_colors[channel_index]),
            limits[channel_index],
            max_render_pixels=max_render_pixels,
            render_downsampling=render_downsampling,
        )
        ax.set_title(channel_names[channel_index])
        _clean_axis(ax)
    _tight_figure(fig)
    return fig


def _extract_channel_plane(
    plane: NDArray[Any],
    layout: _AxisLayout,
    channel_index: int,
) -> NDArray[Any]:
    """Extract one channel and move display axes to render rows/columns."""
    if layout.channel_axis is None:
        return _move_display_axes_to_end(plane, layout, None)
    indexer: list[int | slice] = [slice(None)] * plane.ndim
    indexer[layout.channel_axis] = channel_index
    channel = np.asarray(plane[tuple(indexer)])
    return _move_display_axes_to_end(channel, layout, layout.channel_axis)


def _move_display_axes_to_end(
    plane: NDArray[Any],
    layout: _AxisLayout,
    removed_axis: int | None,
) -> NDArray[Any]:
    """Move resolved display axes to the final row and column dimensions."""
    display_axes = tuple(
        axis - (1 if removed_axis is not None and axis > removed_axis else 0)
        for axis in layout.display_axes
    )
    return np.moveaxis(plane, display_axes, (-2, -1))


def _imshow(  # noqa: PLR0913
    ax: Axes,
    image: NDArray[Any],
    cmap: str | mcolors.Colormap,
    limits: _DisplayLimits,
    *,
    max_render_pixels: int | None,
    render_downsampling: RenderDownsamplingMode,
) -> None:
    """Render a 2D image with optional explicit display limits."""
    image = _downsample_for_render(image, max_render_pixels, render_downsampling)
    if limits.vmin is None or limits.vmax is None:
        ax.imshow(image, cmap=cmap)
        return
    ax.imshow(image, cmap=cmap, vmin=limits.vmin, vmax=limits.vmax)


def _clean_axis(ax: Axes) -> None:
    """Remove all axis chrome so only image data remains."""
    ax.set_axis_off()
    ax.set_frame_on(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _tight_figure(fig: Figure) -> None:
    """Minimize Matplotlib padding around image axes."""
    fig.patch.set_alpha(0.0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.94, wspace=0.02, hspace=0.02)


def _figure_html(fig: Figure) -> str:
    """Return responsive HTML for a Matplotlib figure."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=_DEFAULT_DPI)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        '<img alt="Rendered image layer" '
        f'src="data:image/png;base64,{encoded}" '
        'style="display:block; max-width:100%; height:auto; overflow:hidden;" />'
    )


def _channel_cmap(color: tuple[float, float, float]) -> mcolors.Colormap:
    """Return a black-to-channel-color colormap."""
    return mcolors.LinearSegmentedColormap.from_list("channel", [(0.0, 0.0, 0.0), color])


def _downsample_for_render(
    image: NDArray[Any],
    max_render_pixels: int | None,
    render_downsampling: RenderDownsamplingMode,
) -> NDArray[Any]:
    """Downsample a 2D image toward the render pixel limit."""
    if (
        render_downsampling == "none"
        or max_render_pixels is None
        or image.size <= max_render_pixels
    ):
        return image

    if render_downsampling == "nearest":
        return _resize_nearest(image, _render_shape(image.shape, max_render_pixels))
    if render_downsampling == "bilinear":
        return _resize_bilinear(image, _render_shape(image.shape, max_render_pixels))
    if render_downsampling == "bicubic":
        return _resize_bicubic(image, _render_shape(image.shape, max_render_pixels))

    stride = int(np.ceil(np.sqrt(image.size / max_render_pixels)))
    return image[::stride, ::stride]


def _render_shape(shape: tuple[int, ...], max_render_pixels: int) -> tuple[int, int]:
    """Return an aspect-preserving render shape within the pixel limit."""
    height, width = shape
    scale = float(np.sqrt((height * width) / max_render_pixels))
    render_height = max(1, int(np.floor(height / scale)))
    render_width = max(1, int(np.floor(width / scale)))

    while render_height * render_width > max_render_pixels:
        if render_height >= render_width and render_height > 1:
            render_height -= 1
        elif render_width > 1:
            render_width -= 1
        else:
            break
    return render_height, render_width


def _resize_nearest(image: NDArray[Any], shape: tuple[int, int]) -> NDArray[Any]:
    """Resize a 2D image by nearest-neighbor sampling."""
    row_indices = np.rint(np.linspace(0, image.shape[0] - 1, shape[0])).astype(np.intp)
    column_indices = np.rint(np.linspace(0, image.shape[1] - 1, shape[1])).astype(np.intp)
    return image[np.ix_(row_indices, column_indices)]


def _resize_bilinear(image: NDArray[Any], shape: tuple[int, int]) -> NDArray[np.float32]:
    """Resize a 2D image with separable linear interpolation."""
    values = np.asarray(image, dtype=np.float32)
    row_positions = np.linspace(0, values.shape[0] - 1, shape[0], dtype=np.float32)
    column_positions = np.linspace(0, values.shape[1] - 1, shape[1], dtype=np.float32)
    row_resampled = _linear_sample_axis(values, row_positions, axis=0)
    return _linear_sample_axis(row_resampled, column_positions, axis=1)


def _resize_bicubic(image: NDArray[Any], shape: tuple[int, int]) -> NDArray[np.float32]:
    """Resize a 2D image with separable cubic interpolation."""
    values = np.asarray(image, dtype=np.float32)
    row_positions = np.linspace(0, values.shape[0] - 1, shape[0], dtype=np.float32)
    column_positions = np.linspace(0, values.shape[1] - 1, shape[1], dtype=np.float32)
    row_resampled = _cubic_sample_axis(values, row_positions, axis=0)
    return _cubic_sample_axis(row_resampled, column_positions, axis=1)


def _linear_sample_axis(
    values: NDArray[np.float32],
    positions: NDArray[np.float32],
    *,
    axis: Literal[0, 1],
) -> NDArray[np.float32]:
    """Sample one axis with linear interpolation."""
    moved = np.moveaxis(values, axis, 0)
    lower = np.floor(positions).astype(np.intp)
    upper = np.clip(lower + 1, 0, moved.shape[0] - 1)
    weight = (positions - lower).astype(np.float32)
    sampled = (1.0 - weight[:, np.newaxis]) * moved[lower] + weight[:, np.newaxis] * moved[upper]
    return np.moveaxis(sampled.astype(np.float32, copy=False), 0, axis)


def _cubic_sample_axis(
    values: NDArray[np.float32],
    positions: NDArray[np.float32],
    *,
    axis: Literal[0, 1],
) -> NDArray[np.float32]:
    """Sample one axis with Catmull-Rom cubic interpolation."""
    moved = np.moveaxis(values, axis, 0)
    base = np.floor(positions).astype(np.intp)
    fraction = (positions - base).astype(np.float32)
    indices = (
        np.clip(base - 1, 0, moved.shape[0] - 1),
        np.clip(base, 0, moved.shape[0] - 1),
        np.clip(base + 1, 0, moved.shape[0] - 1),
        np.clip(base + 2, 0, moved.shape[0] - 1),
    )
    p0, p1, p2, p3 = (moved[index] for index in indices)
    t = fraction[:, np.newaxis]
    sampled = 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t**2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t**3
    )
    return np.moveaxis(sampled.astype(np.float32, copy=False), 0, axis)


def _normalize_for_overlay(image: NDArray[Any], limits: _DisplayLimits) -> NDArray[np.float32]:
    """Normalize one channel plane to 0-1 for RGB overlay rendering."""
    values = np.asarray(image, dtype=np.float32)
    if limits.vmin is None or limits.vmax is None:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            low, high = 0.0, 1.0
        else:
            low = float(np.min(finite))
            high = float(np.max(finite))
            if low == high:
                high = low + 1.0
    else:
        low, high = limits.vmin, limits.vmax
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _finite_min_max(values: NDArray[Any]) -> tuple[float, float]:
    """Return finite min and max values suitable for display controls."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    low = float(np.min(finite))
    high = float(np.max(finite))
    if low == high:
        high = low + 1.0
    return low, high


def _absolute_display_range(values: NDArray[Any]) -> tuple[float, float]:
    """Return stable absolute display bounds for an image array."""
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.bool_):
        return 0.0, 1.0
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        return float(info.min), float(info.max)
    return _finite_min_max(array)


def _finite_percentiles(values: NDArray[Any], low: float, high: float) -> tuple[float, float]:
    """Return finite percentile limits."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    low_value, high_value = np.percentile(finite, [low, high])
    if low_value == high_value:
        high_value = low_value + 1.0
    return float(low_value), float(high_value)


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp a value into a closed interval."""
    return min(max(value, low), high)


def _slider_step(low: float, high: float) -> float:
    """Return a practical absolute slider step for the current data range."""
    span = high - low
    if span <= 0:
        return 1.0
    return max(span / 500.0, np.finfo(float).eps)
