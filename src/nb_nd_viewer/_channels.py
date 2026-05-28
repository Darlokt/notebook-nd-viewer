"""Channel metadata validation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.colors as mcolors

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nb_nd_viewer._types import ImageArray

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


def validate_channel_names(
    array: ImageArray,
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


def validate_channel_colors(
    array: ImageArray,
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
