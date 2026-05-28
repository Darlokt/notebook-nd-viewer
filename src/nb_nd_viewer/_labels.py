"""Label metadata, layout validation, and plane extraction helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import matplotlib.colors as mcolors
import numpy as np

if TYPE_CHECKING:
    from nb_nd_viewer._types import ImageArray, LabelArray, LabelKind, LabelKindSetting, _AxisLayout

_DEFAULT_LABEL_COLORS = (
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
class _LabelLayout:
    """Resolved axis roles for a label array."""

    axis_order: str
    channel_axis: int | None
    display_axes: tuple[int, int]
    slider_axis_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _LabelEntry:
    """One renderable label entry, optionally from a label channel."""

    index: int
    name: str
    kind: LabelKind
    channel_index: int | None
    color: tuple[float, float, float]
    opacity: float


@dataclass(frozen=True)
class _ResolvedLabels:
    """Validated label array and expanded label entries."""

    array: LabelArray
    layout: _LabelLayout
    entries: tuple[_LabelEntry, ...]


def validate_labels(  # noqa: PLR0913
    labels: object,
    label_axis_order: str | None,
    image: ImageArray,
    image_layout: _AxisLayout,
    label_names: Sequence[str] | None,
    label_colors: Sequence[str] | None,
    label_opacities: float | Sequence[float],
    label_kinds: LabelKindSetting | Sequence[LabelKindSetting],
) -> _ResolvedLabels:
    """Validate label data and return expanded renderable label entries."""
    if label_axis_order is None:
        msg = "label_axis_order is required when labels are provided."
        raise ValueError(msg)
    if _is_sequence_of_label_arrays(labels):
        msg = "labels must be a single array; use a 'C' label axis for multiple labels."
        raise ValueError(msg)

    array = np.asarray(labels)
    if not (np.issubdtype(array.dtype, np.bool_) or np.issubdtype(array.dtype, np.integer)):
        msg = "labels must have a bool or integer dtype."
        raise ValueError(msg)
    typed_array = cast("LabelArray", array)
    layout = validate_label_layout(typed_array, label_axis_order, image, image_layout)
    label_count = 1 if layout.channel_axis is None else typed_array.shape[layout.channel_axis]
    names = _validate_label_names(label_names, label_count)
    colors = _validate_label_colors(label_colors, label_count)
    opacities = _validate_label_opacities(label_opacities, label_count)
    kind_settings = _validate_label_kind_settings(label_kinds, label_count)
    entries = tuple(
        _LabelEntry(
            index=index,
            name=names[index],
            kind=_resolve_label_kind(
                _label_values(typed_array, layout.channel_axis, index),
                kind_settings[index],
            ),
            channel_index=None if layout.channel_axis is None else index,
            color=colors[index],
            opacity=opacities[index],
        )
        for index in range(label_count)
    )
    return _ResolvedLabels(array=typed_array, layout=layout, entries=entries)


def validate_label_layout(
    labels: LabelArray,
    label_axis_order: str,
    image: ImageArray,
    image_layout: _AxisLayout,
) -> _LabelLayout:
    """Validate label axes against the image layout."""
    normalized = _validate_label_axis_order(labels, label_axis_order)
    channel_axis = normalized.find("C")
    channel_axis_or_none = None if channel_axis == -1 else channel_axis
    _validate_label_axis_subset(normalized, channel_axis_or_none, image_layout)
    display_labels = (
        image_layout.axis_order[image_layout.display_axes[0]],
        image_layout.axis_order[image_layout.display_axes[1]],
    )
    _validate_label_display_axes(normalized, display_labels)
    _validate_label_axis_sizes(labels, normalized, channel_axis_or_none, image, image_layout)
    display_axes = (normalized.index(display_labels[0]), normalized.index(display_labels[1]))
    slider_axis_pairs = tuple(
        (image_axis, normalized.index(image_layout.axis_order[image_axis]))
        for image_axis in image_layout.slider_axes
        if image_layout.axis_order[image_axis] in normalized
    )
    _validate_label_remaining_axes(labels, channel_axis_or_none, slider_axis_pairs, display_axes)

    return _LabelLayout(
        axis_order=normalized,
        channel_axis=channel_axis_or_none,
        display_axes=display_axes,
        slider_axis_pairs=slider_axis_pairs,
    )


def extract_label_plane(
    labels: _ResolvedLabels,
    entry: _LabelEntry,
    image_slider_values: dict[int, int],
) -> LabelArray:
    """Extract one label entry and move display axes to render rows/columns."""
    layout = labels.layout
    indexer: list[int | slice] = [slice(None)] * labels.array.ndim
    removed_axes: set[int] = set()
    if layout.channel_axis is not None:
        if entry.channel_index is None:
            msg = "label entry is missing a channel index."
            raise ValueError(msg)
        indexer[layout.channel_axis] = entry.channel_index
        removed_axes.add(layout.channel_axis)
    for image_axis, label_axis in layout.slider_axis_pairs:
        indexer[label_axis] = image_slider_values[image_axis]
        removed_axes.add(label_axis)

    plane = np.asarray(labels.array[tuple(indexer)])
    adjusted_display_axes = tuple(
        axis - sum(removed_axis < axis for removed_axis in removed_axes)
        for axis in layout.display_axes
    )
    return np.moveaxis(plane, adjusted_display_axes, (-2, -1))


def _validate_label_names(
    label_names: Sequence[str] | None,
    label_count: int,
) -> tuple[str, ...]:
    """Return label names matching the expanded label count."""
    if label_names is None:
        return tuple(f"Label {index}" for index in range(label_count))
    if len(label_names) != label_count:
        msg = f"label_names has length {len(label_names)}, expected {label_count}."
        raise ValueError(msg)
    return tuple(str(name) for name in label_names)


def _is_sequence_of_label_arrays(labels: object) -> bool:
    """Return whether labels looks like multiple arrays instead of one array."""
    if isinstance(labels, np.ndarray | str | bytes) or not isinstance(labels, Sequence):
        return False
    return any(isinstance(item, np.ndarray) for item in labels)


def _validate_label_axis_order(labels: LabelArray, label_axis_order: str) -> str:
    """Validate label axis-order syntax and return the normalized value."""
    normalized = label_axis_order.upper()
    if len(normalized) != labels.ndim:
        msg = (
            f"label_axis_order has length {len(normalized)}, "
            f"but labels has {labels.ndim} dimensions."
        )
        raise ValueError(msg)
    if any(size == 0 for size in labels.shape):
        msg = "label axes must be non-empty."
        raise ValueError(msg)
    if normalized.count("C") > 1:
        msg = "label_axis_order may contain at most one label channel axis marked with 'C'."
        raise ValueError(msg)
    duplicate_labels = sorted({label for label in normalized if normalized.count(label) > 1})
    if duplicate_labels:
        duplicates = "', '".join(duplicate_labels)
        msg = f"label_axis_order labels must be unique; duplicated labels: '{duplicates}'."
        raise ValueError(msg)
    return normalized


def _validate_label_axis_subset(
    normalized: str,
    channel_axis: int | None,
    image_layout: _AxisLayout,
) -> None:
    """Validate that label non-channel axes are present in image non-channel axes."""
    label_non_channel_labels = {
        label for axis, label in enumerate(normalized) if axis != channel_axis
    }
    image_non_channel_labels = {
        label
        for axis, label in enumerate(image_layout.axis_order)
        if axis != image_layout.channel_axis
    }
    extra_labels = sorted(label_non_channel_labels - image_non_channel_labels)
    if extra_labels:
        joined = "', '".join(extra_labels)
        msg = f"label axes must be a subset of image non-channel axes; extra labels: '{joined}'."
        raise ValueError(msg)


def _validate_label_display_axes(normalized: str, display_labels: tuple[str, str]) -> None:
    """Validate that labels include both image display axes."""
    missing_display_labels = [label for label in display_labels if label not in normalized]
    if missing_display_labels:
        joined = "', '".join(missing_display_labels)
        msg = f"labels must include image display axes; missing labels: '{joined}'."
        raise ValueError(msg)


def _validate_label_axis_sizes(
    labels: LabelArray,
    normalized: str,
    channel_axis: int | None,
    image: ImageArray,
    image_layout: _AxisLayout,
) -> None:
    """Validate label axis sizes against matching image axes."""
    for label_axis, label in enumerate(normalized):
        if label_axis == channel_axis:
            continue
        image_axis = image_layout.axis_order.index(label)
        if labels.shape[label_axis] != image.shape[image_axis]:
            msg = (
                f"label axis '{label}' has size {labels.shape[label_axis]}, "
                f"expected {image.shape[image_axis]}."
            )
            raise ValueError(msg)


def _validate_label_remaining_axes(
    labels: LabelArray,
    channel_axis: int | None,
    slider_axis_pairs: tuple[tuple[int, int], ...],
    display_axes: tuple[int, int],
) -> None:
    """Validate that all non-channel label axes have an image role."""
    remaining_axes = {axis for axis in range(labels.ndim) if axis != channel_axis} - {
        label_axis for _image_axis, label_axis in slider_axis_pairs
    }
    if remaining_axes != set(display_axes):
        msg = "label axes must include only display axes, an optional 'C', and image slider axes."
        raise ValueError(msg)


def _validate_label_colors(
    label_colors: Sequence[str] | None,
    label_count: int,
) -> tuple[tuple[float, float, float], ...]:
    """Return RGB colors matching the expanded label count."""
    raw_colors = label_colors or _DEFAULT_LABEL_COLORS
    if len(raw_colors) < label_count:
        msg = f"label_colors has length {len(raw_colors)}, expected at least {label_count}."
        raise ValueError(msg)
    return tuple(mcolors.to_rgb(color) for color in raw_colors[:label_count])


def _validate_label_opacities(
    label_opacities: float | Sequence[float],
    label_count: int,
) -> tuple[float, ...]:
    """Return opacity values matching the expanded label count."""
    if isinstance(label_opacities, Sequence):
        if len(label_opacities) != label_count:
            msg = f"label_opacities has length {len(label_opacities)}, expected {label_count}."
            raise ValueError(msg)
        opacity_values = cast("Sequence[float]", label_opacities)
        opacities = tuple(float(opacity) for opacity in opacity_values)
    else:
        opacities = (float(label_opacities),) * label_count
    for opacity in opacities:
        if not 0.0 <= opacity <= 1.0:
            msg = "label_opacities values must be between 0 and 1."
            raise ValueError(msg)
    return opacities


def _validate_label_kind_settings(
    label_kinds: LabelKindSetting | Sequence[LabelKindSetting],
    label_count: int,
) -> tuple[LabelKindSetting, ...]:
    """Return kind settings matching the expanded label count."""
    if isinstance(label_kinds, str):
        raw_settings = (label_kinds,) * label_count
    else:
        if len(label_kinds) != label_count:
            msg = f"label_kinds has length {len(label_kinds)}, expected {label_count}."
            raise ValueError(msg)
        raw_settings = tuple(label_kinds)
    invalid = sorted({kind for kind in raw_settings if kind not in {"auto", "binary", "integer"}})
    if invalid:
        joined = "', '".join(invalid)
        msg = f"label_kinds contains unsupported values: '{joined}'."
        raise ValueError(msg)
    return tuple(cast("LabelKindSetting", kind) for kind in raw_settings)


def _label_values(
    labels: LabelArray,
    channel_axis: int | None,
    index: int,
) -> LabelArray:
    """Return values for one expanded label entry."""
    if channel_axis is None:
        return labels
    indexer: list[int | slice] = [slice(None)] * labels.ndim
    indexer[channel_axis] = index
    return np.asarray(labels[tuple(indexer)])


def _resolve_label_kind(values: LabelArray, setting: LabelKindSetting) -> LabelKind:
    """Resolve one label entry kind from user settings and data values."""
    if setting == "binary":
        return "binary"
    if setting == "integer":
        return "integer"
    if np.issubdtype(values.dtype, np.bool_):
        return "binary"
    if values.size == 0:
        return "binary"
    low = int(np.min(values))
    high = int(np.max(values))
    if low >= 0 and high <= 1:
        return "binary"
    return "integer"
