"""Interactive Matplotlib viewer for exploratory image arrays with labels."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import ipywidgets as widgets
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from nb_nd_viewer import _channels, _controls, _labels, _layout, _ranges, _rendering
from nb_nd_viewer import _types as _viewer_types
from nb_nd_viewer._types import _DisplayLimits

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

__all__ = ["view_labeled_image"]


def view_labeled_image(  # noqa: C901, PLR0912, PLR0913, PLR0915
    image: ArrayLike,
    axis_order: str,
    labels: ArrayLike | None = None,
    label_axis_order: str | None = None,
    channel_names: Sequence[str] | None = None,
    label_names: Sequence[str] | None = None,
    label_colors: Sequence[str] | None = None,
    label_opacities: float | Sequence[float] = 0.5,
    label_kinds: _viewer_types.LabelKindSetting | Sequence[_viewer_types.LabelKindSetting] = "auto",
    figsize: tuple[float, float] = (6, 6),
    cmap: str = "gray",
    integer_label_cmap: str = "tab20",
    *,
    continuous_update: bool = False,
    max_render_pixels: int | None = 1_000_000,
    render_downsampling: _viewer_types.RenderDownsamplingMode = "stride",
) -> widgets.Widget:
    """Display an interactive image viewer with optional label overlays.

    This viewer keeps the stack and image-channel browsing model from
    :func:`nb_nd_viewer.image_viewer.view_image_layers`, but image channels
    render as white intensity layers and labels have independent overlay
    controls. ``labels`` may omit image stack axes; omitted axes are broadcast
    across every image slice. Label arrays use ``C`` in ``label_axis_order`` as
    an optional label-channel axis for multiple masks or label images.

    Parameters
    ----------
    image
        Image-like array to inspect.
    axis_order
        Axis labels matching ``image.ndim``. ``C`` marks the optional image
        channel axis.
    labels
        Optional bool or integer label array to overlay.
    label_axis_order
        Axis labels matching ``labels.ndim``. Required when ``labels`` is
        provided. ``C`` marks the optional label channel axis.
    channel_names
        Optional display names for the image channel axis.
    label_names
        Optional display names for expanded label entries.
    label_colors
        Optional Matplotlib-compatible colors for binary label rendering.
    label_opacities
        One opacity for all labels or one value per expanded label entry.
    label_kinds
        ``"auto"``, ``"binary"``, or ``"integer"`` for all labels, or one
        setting per expanded label entry.
    figsize
        Base Matplotlib figure size before responsive notebook scaling.
    cmap
        Colormap for non-channel images.
    integer_label_cmap
        Matplotlib colormap used for integer label IDs.
    continuous_update
        If ``True``, redraw while sliders are dragged.
    max_render_pixels
        Maximum number of pixels rendered per 2D plane. Label overlays use the
        same budget but always downsample with nearest-neighbor sampling.
    render_downsampling
        Downsampling mode for image planes larger than ``max_render_pixels``.

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
    resolved_labels = None
    if labels is not None:
        resolved_labels = _labels.validate_labels(
            labels,
            label_axis_order,
            typed_array,
            resolved_layout,
            label_names,
            label_colors,
            label_opacities,
            label_kinds,
        )

    axis_sliders = _controls.make_axis_sliders(
        typed_array,
        resolved_layout,
        continuous_update=continuous_update,
    )
    has_channels = resolved_layout.channel_axis is not None
    channel_selector = _controls.make_channel_selector(names) if has_channels else None
    display_mode = _controls.make_labeled_image_display_mode() if has_channels else None

    setting_names = names if has_channels else ("Image",)
    image_settings = {
        index: _controls.make_channel_controls(
            index,
            setting_names[index],
            (),
            continuous_update=continuous_update,
        )
        for index in range(len(setting_names))
    }
    image_settings_box = widgets.VBox([], layout=widgets.Layout(grid_gap="8px"))

    label_selector = None
    label_settings: dict[int, _viewer_types._LabelControls] = {}
    label_settings_box = widgets.VBox([], layout=widgets.Layout(grid_gap="8px"))
    if resolved_labels is not None:
        label_selector = _controls.make_label_selector(
            tuple(entry.name for entry in resolved_labels.entries)
        )
        label_settings = {
            entry.index: _controls.make_label_controls(
                entry.name,
                entry.kind,
                entry.color,
                entry.opacity,
                continuous_update=continuous_update,
            )
            for entry in resolved_labels.entries
        }

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

    def image_slider_values() -> dict[int, int]:
        return {axis: int(slider.value) for axis, slider in axis_sliders.items()}

    def selected_channels() -> tuple[int, ...]:
        if channel_selector is None:
            return (0,)
        return tuple(int(index) for index in channel_selector.value) or (0,)

    def selected_labels() -> tuple[int, ...]:
        if label_selector is None:
            return ()
        return tuple(int(index) for index in label_selector.value)

    def current_display_mode() -> _viewer_types.LabeledImageDisplayMode:
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
        controls = image_settings[channel_index]
        mode = controls.mode.value
        if mode == "absolute":
            low, high = sorted(controls.absolute.value)
            return _DisplayLimits(vmin=float(low), vmax=float(high))

        low_pct, high_pct = sorted(controls.percentile.value)
        low, high = _ranges.finite_percentiles(channel_plane(channel_index), low_pct, high_pct)
        return _DisplayLimits(vmin=low, vmax=high)

    def label_overlays() -> tuple[_rendering.LabelOverlay, ...]:
        if resolved_labels is None:
            return ()
        overlays: list[_rendering.LabelOverlay] = []
        sliders = image_slider_values()
        for label_index in selected_labels():
            entry = resolved_labels.entries[label_index]
            controls = label_settings[label_index]
            overlays.append(
                _rendering.LabelOverlay(
                    plane=_labels.extract_label_plane(resolved_labels, entry, sliders),
                    kind=entry.kind,
                    color=mcolors.to_rgb(controls.color.value),
                    opacity=float(controls.opacity.value),
                    binary_mode=bool(
                        controls.binary_mode is not None and controls.binary_mode.value
                    ),
                )
            )
        return tuple(overlays)

    def redraw() -> None:
        nonlocal updating_controls
        if updating_controls:
            return
        updating_controls = True
        try:
            _controls.sync_visible_settings(image_settings_box, image_settings, selected_channels())
            if label_selector is not None:
                _controls.sync_visible_label_settings(
                    label_settings_box, label_settings, selected_labels()
                )
            for channel_index in selected_channels():
                _controls.sync_absolute_range(
                    image_settings[channel_index],
                    channel_volume(channel_index),
                    channel_plane(channel_index),
                )
        finally:
            updating_controls = False

        plane, plane_layout = current_plane()
        channels = selected_channels()
        limits = {channel_index: display_limits(channel_index) for channel_index in channels}
        fig = _rendering.draw_labeled_plane(
            plane=plane,
            layout=plane_layout,
            mode=current_display_mode(),
            selected_channels=channels,
            channel_names=names,
            limits=limits,
            label_overlays=label_overlays(),
            figsize=figsize,
            cmap=cmap,
            integer_label_cmap=integer_label_cmap,
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
    for controls in image_settings.values():
        watched_controls.extend(
            control
            for control in (
                controls.mode,
                controls.absolute,
                controls.absolute_slice_minmax,
                controls.percentile,
            )
            if control is not None
        )
    if label_selector is not None:
        watched_controls.append(label_selector)
    for controls in label_settings.values():
        watched_controls.extend(
            control
            for control in (controls.color, controls.opacity, controls.binary_mode)
            if control is not None
        )
    for control in watched_controls:
        control.observe(on_control_change, names="value")

    def on_slice_minmax_click(channel_index: int) -> None:
        _controls.set_absolute_to_slice_minmax(
            image_settings[channel_index],
            channel_volume(channel_index),
            channel_plane(channel_index),
        )
        redraw()

    for channel_index, controls in image_settings.items():
        controls.absolute_slice_button.on_click(
            lambda _button, channel_index=channel_index: on_slice_minmax_click(channel_index),
        )

    control_sections: list[widgets.Widget] = []
    if axis_sliders:
        control_sections.append(_controls.control_section("Layers", [*axis_sliders.values()]))
    if channel_selector is not None and display_mode is not None:
        control_sections.append(
            _controls.control_section("Image channels", [display_mode, channel_selector])
        )
    control_sections.append(
        _controls.control_section("Image display settings", [image_settings_box])
    )
    if label_selector is not None:
        control_sections.append(_controls.control_section("Labels", [label_selector]))
        control_sections.append(
            _controls.control_section("Label display settings", [label_settings_box])
        )

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
