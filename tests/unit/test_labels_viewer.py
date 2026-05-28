"""Unit tests for the public interactive labels viewer entry point."""

from __future__ import annotations

import numpy as np
import pytest
from ipywidgets import HBox

from nb_nd_viewer.labels_viewer import view_labeled_image


class TestViewLabeledImage:
    """Tests for the public labeled image viewer function."""

    def test_accepts_labels_none(self) -> None:
        image = np.arange(9).reshape(3, 3)

        widget = view_labeled_image(image, "YX", labels=None)

        assert isinstance(widget, HBox)

    def test_returns_widget_for_binary_labels(self) -> None:
        image = np.arange(9).reshape(3, 3)
        labels = image > 4

        widget = view_labeled_image(image, "YX", labels=labels, label_axis_order="YX")

        assert isinstance(widget, HBox)

    def test_returns_widget_for_multichannel_labels_over_stack(self) -> None:
        image = np.arange(36).reshape(2, 2, 3, 3)
        labels = np.zeros((2, 3, 3), dtype=bool)

        widget = view_labeled_image(
            image,
            "ZCYX",
            labels=labels,
            label_axis_order="CYX",
            channel_names=("A", "B"),
            label_names=("Nuclei", "Cells"),
        )

        assert isinstance(widget, HBox)

    def test_requires_label_axis_order_when_labels_are_provided(self) -> None:
        image = np.zeros((2, 3))

        with pytest.raises(ValueError, match="label_axis_order is required"):
            view_labeled_image(image, "YX", labels=np.zeros((2, 3), dtype=bool))

    def test_rejects_non_numeric_image_dtype(self) -> None:
        image = np.array([["a", "b"], ["c", "d"]])

        with pytest.raises(ValueError, match="bool, integer, or floating dtype"):
            view_labeled_image(image, "YX")
