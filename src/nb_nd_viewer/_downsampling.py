"""Render downsampling helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from nb_nd_viewer._types import ImageArray, RenderDownsamplingMode


def downsample_for_render(
    image: ImageArray,
    max_render_pixels: int | None,
    render_downsampling: RenderDownsamplingMode,
) -> ImageArray | np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
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


def _resize_nearest(image: ImageArray, shape: tuple[int, int]) -> ImageArray:
    """Resize a 2D image by nearest-neighbor sampling."""
    row_indices = np.rint(np.linspace(0, image.shape[0] - 1, shape[0])).astype(np.intp)
    column_indices = np.rint(np.linspace(0, image.shape[1] - 1, shape[1])).astype(np.intp)
    return image[np.ix_(row_indices, column_indices)]


def _resize_bilinear(
    image: ImageArray,
    shape: tuple[int, int],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
    """Resize a 2D image with separable linear interpolation."""
    values = np.asarray(image, dtype=np.float32)
    row_positions = np.linspace(0, values.shape[0] - 1, shape[0], dtype=np.float32)
    column_positions = np.linspace(0, values.shape[1] - 1, shape[1], dtype=np.float32)
    row_resampled = _linear_sample_axis(values, row_positions, axis=0)
    return _linear_sample_axis(row_resampled, column_positions, axis=1)


def _resize_bicubic(
    image: ImageArray,
    shape: tuple[int, int],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
    """Resize a 2D image with separable cubic interpolation."""
    values = np.asarray(image, dtype=np.float32)
    row_positions = np.linspace(0, values.shape[0] - 1, shape[0], dtype=np.float32)
    column_positions = np.linspace(0, values.shape[1] - 1, shape[1], dtype=np.float32)
    row_resampled = _cubic_sample_axis(values, row_positions, axis=0)
    return _cubic_sample_axis(row_resampled, column_positions, axis=1)


def _linear_sample_axis(
    values: np.ndarray[tuple[int, ...], np.dtype[np.float32]],
    positions: np.ndarray[tuple[int, ...], np.dtype[np.float32]],
    *,
    axis: Literal[0, 1],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
    """Sample one axis with linear interpolation."""
    moved = np.moveaxis(values, axis, 0)
    lower = np.floor(positions).astype(np.intp)
    upper = np.clip(lower + 1, 0, moved.shape[0] - 1)
    weight = (positions - lower).astype(np.float32)
    sampled = (1.0 - weight[:, np.newaxis]) * moved[lower] + weight[:, np.newaxis] * moved[upper]
    return np.moveaxis(sampled.astype(np.float32, copy=False), 0, axis)


def _cubic_sample_axis(
    values: np.ndarray[tuple[int, ...], np.dtype[np.float32]],
    positions: np.ndarray[tuple[int, ...], np.dtype[np.float32]],
    *,
    axis: Literal[0, 1],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
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
