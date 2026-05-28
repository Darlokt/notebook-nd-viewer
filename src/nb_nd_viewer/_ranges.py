"""Display-range calculations for rendered image planes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nb_nd_viewer._types import ImageArray, _DisplayLimits


def normalize_for_overlay(
    image: ImageArray, limits: _DisplayLimits
) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
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


def finite_min_max(values: ImageArray) -> tuple[float, float]:
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


def absolute_display_range(values: ImageArray) -> tuple[float, float]:
    """Return stable absolute display bounds for an image array."""
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.bool_):
        return 0.0, 1.0
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(str(array.dtype))
        return float(info.min), float(info.max)
    return finite_min_max(array)


def finite_percentiles(values: ImageArray, low: float, high: float) -> tuple[float, float]:
    """Return finite percentile limits."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    low_value, high_value = np.percentile(finite, [low, high])
    if low_value == high_value:
        high_value = low_value + 1.0
    return float(low_value), float(high_value)


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a value into a closed interval."""
    return min(max(value, low), high)


def slider_step(low: float, high: float) -> float:
    """Return a practical absolute slider step for the current data range."""
    span = high - low
    if span <= 0:
        return 1.0
    return max(span / 500.0, np.finfo(float).eps)
