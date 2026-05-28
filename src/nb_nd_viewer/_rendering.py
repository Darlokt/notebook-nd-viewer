"""Matplotlib rendering and notebook HTML serialization helpers."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from nb_nd_viewer._downsampling import downsample_for_render
from nb_nd_viewer._layout import extract_channel_plane, move_display_axes_to_end
from nb_nd_viewer._ranges import normalize_for_overlay

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from nb_nd_viewer._types import (
        DisplayMode,
        ImageArray,
        RenderDownsamplingMode,
        _AxisLayout,
        _DisplayLimits,
    )

_DEFAULT_DPI = 120


def validate_max_render_pixels(max_render_pixels: int | None) -> None:
    """Validate the optional render downsampling limit."""
    if max_render_pixels is None:
        return
    if max_render_pixels < 1:
        msg = "max_render_pixels must be a positive integer or None."
        raise ValueError(msg)


def validate_render_downsampling(render_downsampling: RenderDownsamplingMode) -> None:
    """Validate the render downsampling mode."""
    if render_downsampling in {"stride", "nearest", "bilinear", "bicubic", "none"}:
        return
    msg = "render_downsampling must be 'stride', 'nearest', 'bilinear', 'bicubic', or 'none'."
    raise ValueError(msg)


def draw_plane(  # noqa: PLR0913
    *,
    plane: ImageArray,
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
        image_2d = move_display_axes_to_end(plane, layout, None)
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
        extract_channel_plane(plane, layout, channel_index),
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
    plane: ImageArray,
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
    first = extract_channel_plane(plane, layout, selected_channels[0])
    first = downsample_for_render(first, max_render_pixels, render_downsampling)
    rgb = np.zeros((*first.shape, 3), dtype=np.float32)
    for channel_index in selected_channels:
        image = extract_channel_plane(plane, layout, channel_index)
        channel = normalize_for_overlay(
            downsample_for_render(image, max_render_pixels, render_downsampling),
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
    plane: ImageArray,
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
            extract_channel_plane(plane, layout, channel_index),
            _channel_cmap(channel_colors[channel_index]),
            limits[channel_index],
            max_render_pixels=max_render_pixels,
            render_downsampling=render_downsampling,
        )
        ax.set_title(channel_names[channel_index])
        _clean_axis(ax)
    _tight_figure(fig)
    return fig


def _imshow(  # noqa: PLR0913
    ax: Axes,
    image: ImageArray,
    cmap: str | mcolors.Colormap,
    limits: _DisplayLimits,
    *,
    max_render_pixels: int | None,
    render_downsampling: RenderDownsamplingMode,
) -> None:
    """Render a 2D image with optional explicit display limits."""
    image = downsample_for_render(image, max_render_pixels, render_downsampling)
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


def figure_html(fig: Figure) -> str:
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
