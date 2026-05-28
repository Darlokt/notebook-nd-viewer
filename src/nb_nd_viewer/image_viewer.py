"""Interactive Matplotlib viewers for exploratory image arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import ipywidgets as widgets
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from nb_nd_viewer import _channels, _controls, _layout, _ranges, _rendering
from nb_nd_viewer import _types as _viewer_types
from nb_nd_viewer._types import _DisplayLimits

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

__all__ = ["view_image_layers"]


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
    render_downsampling: _viewer_types.RenderDownsamplingMode = "stride",
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
    if not (
        np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        msg = "image must have a bool, integer, or floating dtype."
        raise ValueError(msg)
    typed_array = cast("_viewer_types.ImageArray", array)
    resolved_layout = _layout.validate_layout(typed_array, axis_order)
    _downsampling_mode = render_downsampling
    _rendering.validate_max_render_pixels(max_render_pixels)
    _rendering.validate_render_downsampling(_downsampling_mode)
    names = _channels.validate_channel_names(
        typed_array, resolved_layout.channel_axis, channel_names
    )
    colors = _channels.validate_channel_colors(
        typed_array, resolved_layout.channel_axis, channel_colors
    )

    axis_sliders = _controls.make_axis_sliders(
        typed_array,
        resolved_layout,
        continuous_update=continuous_update,
    )
    has_channels = resolved_layout.channel_axis is not None
    channel_selector = _controls.make_channel_selector(names) if has_channels else None
    display_mode = _controls.make_display_mode() if has_channels else None

    setting_names = names if has_channels else ("Image",)
    setting_colors = colors if has_channels else ()
    settings = {
        index: _controls.make_channel_controls(
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

    def current_plane() -> tuple[_viewer_types.ImageArray, _viewer_types._AxisLayout]:
        indexer: list[int | slice] = [slice(None)] * typed_array.ndim
        for axis, slider in axis_sliders.items():
            indexer[axis] = slider.value
        return np.asarray(typed_array[tuple(indexer)]), _layout.remove_axes_from_layout(
            resolved_layout,
            tuple(axis_sliders),
        )

    def selected_channels() -> tuple[int, ...]:
        if channel_selector is None:
            return (0,)
        return tuple(int(index) for index in channel_selector.value) or (0,)

    def current_display_mode() -> _viewer_types.DisplayMode:
        if display_mode is None:
            return "single"
        return display_mode.value

    def channel_plane(channel_index: int) -> _viewer_types.ImageArray:
        plane, plane_layout = current_plane()
        if not has_channels:
            return _layout.move_display_axes_to_end(plane, plane_layout, None)
        return _layout.extract_channel_plane(plane, plane_layout, channel_index)

    def channel_volume(channel_index: int) -> _viewer_types.ImageArray:
        if not has_channels:
            return typed_array
        channel_axis = cast("int", resolved_layout.channel_axis)
        indexer: list[int | slice] = [slice(None)] * typed_array.ndim
        indexer[channel_axis] = channel_index
        return np.asarray(typed_array[tuple(indexer)])

    def display_limits(channel_index: int) -> _viewer_types._DisplayLimits:
        controls = settings[channel_index]
        mode = controls.mode.value
        if mode == "absolute":
            low, high = sorted(controls.absolute.value)
            return _DisplayLimits(vmin=float(low), vmax=float(high))

        low_pct, high_pct = sorted(controls.percentile.value)
        low, high = _ranges.finite_percentiles(channel_plane(channel_index), low_pct, high_pct)
        return _DisplayLimits(vmin=low, vmax=high)

    def redraw() -> None:
        nonlocal updating_controls
        if updating_controls:
            return
        updating_controls = True
        try:
            _controls.sync_visible_settings(settings_box, settings, selected_channels())
            for channel_index in selected_channels():
                _controls.sync_absolute_range(
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
        fig = _rendering.draw_plane(
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
            render_downsampling=_downsampling_mode,
        )
        image_html.value = _rendering.figure_html(fig)
        plt.close(fig)

    def on_control_change(change: dict[str, object]) -> None:
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
        _controls.set_absolute_to_slice_minmax(
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
        control_sections.append(_controls.control_section("Layers", [*axis_sliders.values()]))
    if channel_selector is not None and display_mode is not None:
        control_sections.append(
            _controls.control_section("Channels", [display_mode, channel_selector])
        )
    control_sections.append(_controls.control_section("Display settings", [settings_box]))

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
