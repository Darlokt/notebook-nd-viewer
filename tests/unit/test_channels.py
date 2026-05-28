"""Unit tests for channel metadata helpers."""

from __future__ import annotations

import numpy as np
import pytest

from nb_nd_viewer._channels import validate_channel_colors, validate_channel_names


class TestValidateChannelNames:
    """Tests for channel display-name validation."""

    def test_generates_default_channel_names(self) -> None:
        names = validate_channel_names(np.zeros((2, 3, 4)), 0, None)

        assert names == ("Channel 0", "Channel 1")

    def test_rejects_wrong_name_count(self) -> None:
        with pytest.raises(ValueError, match="expected 2"):
            validate_channel_names(np.zeros((2, 3, 4)), 0, ("only one",))


class TestValidateChannelColors:
    """Tests for channel color validation."""

    def test_returns_rgb_channel_colors(self) -> None:
        colors = validate_channel_colors(np.zeros((2, 3, 4)), 0, ("red", "green"))

        assert colors[0] == pytest.approx((1.0, 0.0, 0.0))
        assert colors[1] == pytest.approx((0.0, 0.50196078, 0.0))

    def test_rejects_colors_without_channel_axis(self) -> None:
        with pytest.raises(ValueError, match="axis_order has no 'C' channel axis"):
            validate_channel_colors(np.zeros((3, 4)), None, ("red",))

    def test_rejects_invalid_color_values(self) -> None:
        with pytest.raises(ValueError, match="not-a-color"):
            validate_channel_colors(np.zeros((1, 3, 4)), 0, ("not-a-color",))
