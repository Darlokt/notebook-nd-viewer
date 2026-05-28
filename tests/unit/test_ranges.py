"""Unit tests for display-range calculations."""

from __future__ import annotations

import numpy as np

from nb_nd_viewer._ranges import (
    absolute_display_range,
    finite_min_max,
    finite_percentiles,
    normalize_for_overlay,
    slider_step,
)
from nb_nd_viewer._types import _DisplayLimits


class TestFiniteMinMax:
    """Tests for finite display-range extraction."""

    def test_ignores_non_finite_values(self) -> None:
        low, high = finite_min_max(np.array([np.nan, -2.0, np.inf, 4.0]))

        assert (low, high) == (-2.0, 4.0)

    def test_returns_default_range_when_no_finite_values_exist(self) -> None:
        assert finite_min_max(np.array([np.nan, np.inf])) == (0.0, 1.0)

    def test_expands_constant_range(self) -> None:
        assert finite_min_max(np.array([2.0, 2.0])) == (2.0, 3.0)


class TestAbsoluteDisplayRange:
    """Tests for absolute display-range extraction."""

    def test_uses_unsigned_integer_dtype_range(self) -> None:
        image = np.array([10, 20], dtype=np.uint8)

        assert absolute_display_range(image) == (0.0, 255.0)

    def test_uses_signed_integer_dtype_range(self) -> None:
        image = np.array([-10, 20], dtype=np.int16)

        assert absolute_display_range(image) == (-32768.0, 32767.0)

    def test_uses_finite_data_range_for_float_images(self) -> None:
        image = np.array([np.nan, 1.5, 4.5], dtype=np.float32)

        assert absolute_display_range(image) == (1.5, 4.5)


class TestFinitePercentiles:
    """Tests for percentile display-range extraction."""

    def test_returns_percentiles_of_finite_values(self) -> None:
        low, high = finite_percentiles(np.array([np.nan, 0.0, 10.0, 20.0]), 0.0, 50.0)

        assert (low, high) == (0.0, 10.0)

    def test_expands_equal_percentiles(self) -> None:
        assert finite_percentiles(np.array([5.0, 5.0]), 1.0, 99.0) == (5.0, 6.0)

    def test_returns_default_range_when_no_finite_values_exist(self) -> None:
        assert finite_percentiles(np.array([np.nan, np.inf]), 1.0, 99.0) == (0.0, 1.0)


class TestNormalizeForOverlay:
    """Tests for overlay normalization."""

    def test_normalizes_with_explicit_limits(self) -> None:
        normalized = normalize_for_overlay(
            np.array([-1.0, 0.0, 0.5, 2.0]),
            _DisplayLimits(vmin=0.0, vmax=1.0),
        )

        np.testing.assert_allclose(normalized, np.array([0.0, 0.0, 0.5, 1.0]))
        assert normalized.dtype == np.float32

    def test_returns_zero_when_limits_are_not_ordered(self) -> None:
        normalized = normalize_for_overlay(
            np.array([1.0, 2.0]),
            _DisplayLimits(vmin=1.0, vmax=1.0),
        )

        np.testing.assert_allclose(normalized, np.array([0.0, 0.0]))

    def test_uses_finite_data_range_when_limits_are_missing(self) -> None:
        normalized = normalize_for_overlay(
            np.array([np.nan, 1.0, 3.0]),
            _DisplayLimits(vmin=None, vmax=None),
        )

        np.testing.assert_allclose(normalized, np.array([np.nan, 0.0, 1.0]))


class TestSliderStep:
    """Tests for display-range slider step sizing."""

    def test_returns_fraction_of_positive_span(self) -> None:
        assert slider_step(0.0, 10.0) == 0.02

    def test_returns_default_for_non_positive_span(self) -> None:
        assert slider_step(1.0, 1.0) == 1.0
