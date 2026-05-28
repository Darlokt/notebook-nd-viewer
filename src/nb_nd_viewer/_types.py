"""Shared private types for notebook ndarray viewers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import ipywidgets as widgets

DisplayMode = Literal["single", "overlay", "side-by-side"]
RenderDownsamplingMode = Literal["stride", "nearest", "bilinear", "bicubic", "none"]
ImageScalar: TypeAlias = np.bool_ | np.integer[Any] | np.floating[Any]
ImageArray: TypeAlias = NDArray[ImageScalar]


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
