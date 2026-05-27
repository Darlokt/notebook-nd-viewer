"""Shared pytest configuration for nb_nd_viewer."""

from __future__ import annotations

from pathlib import Path

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply unit and integration markers from each test file's path."""
    test_root = Path(str(config.rootpath)) / "tests"
    for item in items:
        try:
            relative_path = Path(str(item.path)).relative_to(test_root)
        except ValueError:
            continue

        top_level = relative_path.parts[0] if relative_path.parts else ""
        if top_level == "unit":
            item.add_marker(pytest.mark.unit)
        elif top_level == "integration":
            item.add_marker(pytest.mark.integration)
