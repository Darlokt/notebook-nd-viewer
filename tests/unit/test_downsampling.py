"""Unit tests for render downsampling helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from nb_nd_viewer._downsampling import downsample_for_render

if TYPE_CHECKING:
    from nb_nd_viewer._types import RenderDownsamplingMode


class TestDownsampleForRender:
    """Tests for downsampling large rendered planes."""

    def test_returns_original_image_when_limit_is_none(self) -> None:
        image = np.arange(12).reshape(3, 4)

        downsampled = downsample_for_render(image, None, "stride")

        assert downsampled is image

    def test_returns_original_image_when_under_limit(self) -> None:
        image = np.arange(12).reshape(3, 4)

        downsampled = downsample_for_render(image, 12, "stride")

        assert downsampled is image

    def test_returns_original_image_when_mode_is_none(self) -> None:
        image = np.arange(100).reshape(10, 10)

        downsampled = downsample_for_render(image, 25, "none")

        assert downsampled is image

    def test_stride_samples_large_image_toward_limit(self) -> None:
        image = np.arange(100).reshape(10, 10)

        downsampled = downsample_for_render(image, 25, "stride")

        np.testing.assert_array_equal(downsampled, image[::2, ::2])

    def test_nearest_samples_to_aspect_preserving_shape(self) -> None:
        image = np.arange(100).reshape(10, 10)

        downsampled = downsample_for_render(image, 25, "nearest")

        assert downsampled.shape == (5, 5)
        np.testing.assert_array_equal(downsampled[[0, -1], [0, -1]], np.array([0, 99]))

    @pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
    def test_interpolating_modes_return_float32_render_plane(
        self,
        mode: RenderDownsamplingMode,
    ) -> None:
        image = np.arange(100).reshape(10, 10)

        downsampled = downsample_for_render(image, 25, mode)

        assert downsampled.shape == (5, 5)
        assert downsampled.dtype == np.float32
