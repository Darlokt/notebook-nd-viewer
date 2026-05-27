"""Integration tests for the public package API."""

from __future__ import annotations

import nb_nd_viewer


def test_package_exports_view_image_layers() -> None:
    """The package root exposes the intended viewer entry point."""
    assert nb_nd_viewer.__all__ == ["view_image_layers"]
    assert callable(nb_nd_viewer.view_image_layers)
