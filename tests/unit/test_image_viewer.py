"""Unit tests for the public interactive image viewer entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from ipywidgets import HBox

from nb_nd_viewer.image_viewer import view_image_layers

if TYPE_CHECKING:
    from nb_nd_viewer._types import RenderDownsamplingMode


class TestViewImageLayers:
    """Tests for the public ``view_image_layers`` function."""

    def test_returns_widget_for_2d_image(self) -> None:
        image = np.arange(9).reshape(3, 3)

        widget = view_image_layers(image, "YX")

        assert isinstance(widget, HBox)

    def test_returns_widget_for_channel_stack(self) -> None:
        image = np.arange(24).reshape(2, 3, 4)

        widget = view_image_layers(
            image,
            "CYX",
            channel_names=("DAPI", "Actin"),
            channel_colors=("blue", "green"),
        )

        assert isinstance(widget, HBox)

    def test_rejects_axis_order_with_wrong_length(self) -> None:
        image = np.zeros((2, 3))

        with pytest.raises(ValueError, match="axis_order has length 3"):
            view_image_layers(image, "ZYX")

    def test_rejects_channel_names_without_channel_axis(self) -> None:
        image = np.zeros((2, 3))

        with pytest.raises(ValueError, match="axis_order has no 'C' channel axis"):
            view_image_layers(image, "YX", channel_names=("Channel 0",))

    def test_rejects_too_few_channel_colors(self) -> None:
        image = np.zeros((2, 3, 4))

        with pytest.raises(ValueError, match="expected at least 2"):
            view_image_layers(image, "CYX", channel_colors=("red",))

    def test_rejects_non_positive_max_render_pixels(self) -> None:
        image = np.zeros((2, 3))

        with pytest.raises(ValueError, match="max_render_pixels must be a positive integer"):
            view_image_layers(image, "YX", max_render_pixels=0)

    def test_rejects_unknown_render_downsampling_mode(self) -> None:
        image = np.zeros((2, 3))

        with pytest.raises(ValueError, match="render_downsampling must be"):
            view_image_layers(
                image,
                "YX",
                render_downsampling=cast("RenderDownsamplingMode", "lanczos"),
            )

    def test_rejects_non_numeric_image_dtype(self) -> None:
        image = np.array([["a", "b"], ["c", "d"]])

        with pytest.raises(ValueError, match="bool, integer, or floating dtype"):
            view_image_layers(image, "YX")
