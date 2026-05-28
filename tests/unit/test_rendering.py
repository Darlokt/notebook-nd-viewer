"""Unit tests for Matplotlib rendering helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nb_nd_viewer._rendering import (
    draw_plane,
    figure_html,
    validate_max_render_pixels,
    validate_render_downsampling,
)
from nb_nd_viewer._types import RenderDownsamplingMode, _AxisLayout, _DisplayLimits

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestValidateMaxRenderPixels:
    """Tests for render downsampling limit validation."""

    def test_accepts_none_to_disable_downsampling(self) -> None:
        validate_max_render_pixels(None)

    def test_rejects_non_positive_limit(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            validate_max_render_pixels(0)


class TestValidateRenderDownsampling:
    """Tests for render downsampling mode validation."""

    @pytest.mark.parametrize(
        "mode",
        ["stride", "nearest", "bilinear", "bicubic", "none"],
    )
    def test_accepts_supported_modes(self, mode: RenderDownsamplingMode) -> None:
        validate_render_downsampling(mode)

    def test_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="render_downsampling must be"):
            validate_render_downsampling(cast("RenderDownsamplingMode", "lanczos"))


class TestDrawPlane:
    """Tests for Matplotlib rendering dispatch."""

    def test_draws_non_channel_plane_with_requested_limits(self) -> None:
        layout = _AxisLayout(
            axis_order="YX",
            channel_axis=None,
            display_axes=(0, 1),
            slider_axes=(),
        )

        fig = draw_plane(
            plane=np.arange(4).reshape(2, 2),
            layout=layout,
            mode="single",
            selected_channels=(0,),
            channel_names=(),
            channel_colors={},
            limits={0: _DisplayLimits(vmin=0.0, vmax=3.0)},
            figsize=(2.0, 2.0),
            cmap="gray",
            max_render_pixels=None,
            render_downsampling="stride",
        )

        try:
            assert len(fig.axes) == 1
            assert fig.axes[0].images[0].get_clim() == (0.0, 3.0)
        finally:
            plt.close(fig)

    def test_draws_reversed_x_y_plane_as_y_x(self) -> None:
        layout = _AxisLayout(
            axis_order="XY",
            channel_axis=None,
            display_axes=(1, 0),
            slider_axes=(),
        )

        fig = draw_plane(
            plane=np.arange(6).reshape(2, 3),
            layout=layout,
            mode="single",
            selected_channels=(0,),
            channel_names=(),
            channel_colors={},
            limits={0: _DisplayLimits(vmin=None, vmax=None)},
            figsize=(2.0, 2.0),
            cmap="gray",
            max_render_pixels=None,
            render_downsampling="stride",
        )

        try:
            np.testing.assert_array_equal(
                fig.axes[0].images[0].get_array(), np.arange(6).reshape(2, 3).T
            )
        finally:
            plt.close(fig)

    def test_draws_single_channel_plane_with_title(self) -> None:
        layout = _AxisLayout(
            axis_order="CYX",
            channel_axis=0,
            display_axes=(1, 2),
            slider_axes=(),
        )

        fig = draw_plane(
            plane=np.arange(8).reshape(2, 2, 2),
            layout=layout,
            mode="single",
            selected_channels=(1,),
            channel_names=("A", "B"),
            channel_colors={1: (0.0, 1.0, 0.0)},
            limits={1: _DisplayLimits(vmin=0.0, vmax=7.0)},
            figsize=(2.0, 2.0),
            cmap="gray",
            max_render_pixels=None,
            render_downsampling="stride",
        )

        try:
            assert fig.axes[0].get_title() == "B"
            np.testing.assert_array_equal(
                fig.axes[0].images[0].get_array(),
                np.array([[4, 5], [6, 7]]),
            )
        finally:
            plt.close(fig)

    def test_draws_overlay_plane_as_rgb_image(self) -> None:
        layout = _AxisLayout(
            axis_order="CYX",
            channel_axis=0,
            display_axes=(1, 2),
            slider_axes=(),
        )

        fig = draw_plane(
            plane=np.ones((2, 2, 2)),
            layout=layout,
            mode="overlay",
            selected_channels=(0, 1),
            channel_names=("Red", "Green"),
            channel_colors={0: (1.0, 0.0, 0.0), 1: (0.0, 1.0, 0.0)},
            limits={
                0: _DisplayLimits(vmin=0.0, vmax=1.0),
                1: _DisplayLimits(vmin=0.0, vmax=1.0),
            },
            figsize=(2.0, 2.0),
            cmap="gray",
            max_render_pixels=None,
            render_downsampling="stride",
        )

        try:
            assert fig.axes[0].get_title() == "Red, Green"
            image = fig.axes[0].images[0].get_array()
            assert image is not None
            assert image.shape == (2, 2, 3)
        finally:
            plt.close(fig)

    def test_draws_side_by_side_plane_with_one_axis_per_channel(self) -> None:
        layout = _AxisLayout(
            axis_order="CYX",
            channel_axis=0,
            display_axes=(1, 2),
            slider_axes=(),
        )

        fig = draw_plane(
            plane=np.ones((2, 2, 2)),
            layout=layout,
            mode="side-by-side",
            selected_channels=(0, 1),
            channel_names=("A", "B"),
            channel_colors={0: (1.0, 0.0, 0.0), 1: (0.0, 1.0, 0.0)},
            limits={
                0: _DisplayLimits(vmin=0.0, vmax=1.0),
                1: _DisplayLimits(vmin=0.0, vmax=1.0),
            },
            figsize=(2.0, 2.0),
            cmap="gray",
            max_render_pixels=None,
            render_downsampling="stride",
        )

        try:
            assert [axis.get_title() for axis in fig.axes] == ["A", "B"]
        finally:
            plt.close(fig)


class TestFigureHtml:
    """Tests for serializing figures into notebook HTML."""

    def test_returns_embedded_png_image_html(self) -> None:
        fig = draw_plane(
            plane=np.arange(4).reshape(2, 2),
            layout=_AxisLayout("YX", None, (0, 1), ()),
            mode="single",
            selected_channels=(0,),
            channel_names=(),
            channel_colors={},
            limits={0: _DisplayLimits(vmin=None, vmax=None)},
            figsize=(1.0, 1.0),
            cmap="gray",
            max_render_pixels=None,
            render_downsampling="stride",
        )

        try:
            html = figure_html(fig)
        finally:
            plt.close(fig)

        assert html.startswith('<img alt="Rendered image layer" src="data:image/png;base64,')
        assert 'style="display:block; max-width:100%; height:auto; overflow:hidden;"' in html

    def test_saves_without_tight_bbox_options(self, mocker: MockerFixture) -> None:
        fig = plt.figure()
        savefig = mocker.patch.object(
            fig,
            "savefig",
            side_effect=lambda buffer, **_kwargs: buffer.write(b"png"),
        )

        try:
            figure_html(fig)
        finally:
            plt.close(fig)

        kwargs = savefig.call_args.kwargs
        assert kwargs["format"] == "png"
        assert kwargs["dpi"] == 120
        assert "bbox_inches" not in kwargs
        assert "pad_inches" not in kwargs
