"""Axis layout and image-plane extraction helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nb_nd_viewer._types import _AxisLayout

if TYPE_CHECKING:
    from nb_nd_viewer._types import ImageArray

_DISPLAY_NDIMS = 2


def validate_layout(array: ImageArray, axis_order: str) -> _AxisLayout:
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


def remove_axes_from_layout(layout: _AxisLayout, removed_axes: tuple[int, ...]) -> _AxisLayout:
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


def extract_channel_plane(
    plane: ImageArray,
    layout: _AxisLayout,
    channel_index: int,
) -> ImageArray:
    """Extract one channel and move display axes to render rows/columns."""
    if layout.channel_axis is None:
        return move_display_axes_to_end(plane, layout, None)
    indexer: list[int | slice] = [slice(None)] * plane.ndim
    indexer[layout.channel_axis] = channel_index
    channel = np.asarray(plane[tuple(indexer)])
    return move_display_axes_to_end(channel, layout, layout.channel_axis)


def move_display_axes_to_end(
    plane: ImageArray,
    layout: _AxisLayout,
    removed_axis: int | None,
) -> ImageArray:
    """Move resolved display axes to the final row and column dimensions."""
    display_axes = tuple(
        axis - (1 if removed_axis is not None and axis > removed_axis else 0)
        for axis in layout.display_axes
    )
    return np.moveaxis(plane, display_axes, (-2, -1))
