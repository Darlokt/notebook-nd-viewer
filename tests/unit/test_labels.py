"""Unit tests for label metadata and layout helpers."""

from __future__ import annotations

import numpy as np
import pytest

from nb_nd_viewer._labels import extract_label_plane, validate_labels
from nb_nd_viewer._layout import validate_layout


class TestValidateLabels:
    """Tests for resolving label arrays into renderable entries."""

    def test_requires_label_axis_order(self) -> None:
        image = np.zeros((2, 3))
        image_layout = validate_layout(image, "YX")

        with pytest.raises(ValueError, match="label_axis_order is required"):
            validate_labels(
                np.zeros((2, 3), dtype=bool),
                None,
                image,
                image_layout,
                None,
                None,
                0.5,
                "auto",
            )

    def test_expands_label_channel_axis(self) -> None:
        image = np.zeros((4, 2, 3))
        image_layout = validate_layout(image, "ZYX")
        labels = np.zeros((2, 2, 3), dtype=bool)

        resolved = validate_labels(
            labels,
            "CYX",
            image,
            image_layout,
            ("A", "B"),
            ("red", "green"),
            (0.25, 0.75),
            "auto",
        )

        assert [entry.name for entry in resolved.entries] == ["A", "B"]
        assert [entry.channel_index for entry in resolved.entries] == [0, 1]
        assert [entry.kind for entry in resolved.entries] == ["binary", "binary"]
        assert [entry.opacity for entry in resolved.entries] == [0.25, 0.75]

    def test_detects_integer_label_kind(self) -> None:
        image = np.zeros((2, 3))
        image_layout = validate_layout(image, "YX")

        resolved = validate_labels(
            np.array([[0, 2, 0], [3, 0, 1]], dtype=np.uint16),
            "YX",
            image,
            image_layout,
            None,
            None,
            0.5,
            "auto",
        )

        assert resolved.entries[0].kind == "integer"

    def test_detects_integer_zero_one_as_binary(self) -> None:
        image = np.zeros((2, 3))
        image_layout = validate_layout(image, "YX")

        resolved = validate_labels(
            np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8),
            "YX",
            image,
            image_layout,
            None,
            None,
            0.5,
            "auto",
        )

        assert resolved.entries[0].kind == "binary"

    def test_rejects_non_integer_label_dtype(self) -> None:
        image = np.zeros((2, 3))
        image_layout = validate_layout(image, "YX")

        with pytest.raises(ValueError, match="bool or integer dtype"):
            validate_labels(
                np.zeros((2, 3), dtype=float),
                "YX",
                image,
                image_layout,
                None,
                None,
                0.5,
                "auto",
            )

    def test_rejects_sequence_of_label_arrays(self) -> None:
        image = np.zeros((2, 3))
        image_layout = validate_layout(image, "YX")

        with pytest.raises(ValueError, match="single array"):
            validate_labels(
                [np.zeros((2, 3), dtype=bool), np.ones((2, 3), dtype=bool)],
                "CYX",
                image,
                image_layout,
                None,
                None,
                0.5,
                "auto",
            )

    def test_rejects_extra_non_channel_axis(self) -> None:
        image = np.zeros((2, 3))
        image_layout = validate_layout(image, "YX")

        with pytest.raises(ValueError, match="extra labels: 'Z'"):
            validate_labels(
                np.zeros((1, 2, 3), dtype=bool),
                "ZYX",
                image,
                image_layout,
                None,
                None,
                0.5,
                "auto",
            )

    def test_rejects_mismatched_included_axis_size(self) -> None:
        image = np.zeros((2, 3))
        image_layout = validate_layout(image, "YX")

        with pytest.raises(ValueError, match="axis 'Y' has size 4"):
            validate_labels(
                np.zeros((4, 3), dtype=bool),
                "YX",
                image,
                image_layout,
                None,
                None,
                0.5,
                "auto",
            )

    def test_rejects_wrong_metadata_lengths(self) -> None:
        image = np.zeros((2, 2, 3))
        image_layout = validate_layout(image, "ZYX")

        with pytest.raises(ValueError, match="label_names has length 1"):
            validate_labels(
                np.zeros((2, 2, 3), dtype=bool),
                "CYX",
                image,
                image_layout,
                ("only one",),
                None,
                0.5,
                "auto",
            )


class TestExtractLabelPlane:
    """Tests for label extraction and omitted-axis broadcasting."""

    def test_omitted_image_slider_axis_broadcasts_across_slices(self) -> None:
        image = np.zeros((4, 2, 3))
        image_layout = validate_layout(image, "ZYX")
        labels = np.arange(6).reshape(2, 3).astype(np.uint8)
        resolved = validate_labels(
            labels,
            "YX",
            image,
            image_layout,
            None,
            None,
            0.5,
            "auto",
        )

        first = extract_label_plane(resolved, resolved.entries[0], {0: 0})
        later = extract_label_plane(resolved, resolved.entries[0], {0: 3})

        np.testing.assert_array_equal(first, labels)
        np.testing.assert_array_equal(later, labels)

    def test_included_image_slider_axis_selects_matching_label_slice(self) -> None:
        image = np.zeros((4, 2, 3))
        image_layout = validate_layout(image, "ZYX")
        labels = np.arange(24).reshape(4, 2, 3).astype(np.uint8)
        resolved = validate_labels(
            labels,
            "ZYX",
            image,
            image_layout,
            None,
            None,
            0.5,
            "auto",
        )

        plane = extract_label_plane(resolved, resolved.entries[0], {0: 2})

        np.testing.assert_array_equal(plane, labels[2])
