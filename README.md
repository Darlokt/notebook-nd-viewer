# notebook-nd-viewer

![Status](https://img.shields.io/badge/Status-in_development-yellow)
[![Latest Release](https://img.shields.io/github/v/release/Darlokt/notebook-nd-viewer?display_name=tag&sort=semver)](https://github.com/Darlokt/notebook-nd-viewer/releases)
[![PyPI](https://img.shields.io/pypi/v/notebook-nd-viewer.svg)](https://pypi.org/project/notebook-nd-viewer/)
[![CI](https://github.com/Darlokt/notebook-nd-viewer/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Darlokt/notebook-nd-viewer/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/Coverage-90%25%2B-brightgreen)](https://github.com/Darlokt/notebook-nd-viewer/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/Package_Manager-uv-lightgrey)](https://github.com/astral-sh/uv)
[![Linter](https://img.shields.io/badge/Linter-ruff-lightgrey)](https://github.com/astral-sh/ruff)
[![Type Checker](https://img.shields.io/badge/Type_Checker-ty-lightgrey)](https://github.com/astral-sh/ty)

`notebook-nd-viewer` provides a small, copyable ndarray image viewer for
Jupyter notebooks. It is intended for quick interactive inspection of 2D images,
Z-stacks, and channel images without introducing a full image-viewer application
into an analysis notebook.

The public Python import package is `nb_nd_viewer`.

## Features

- Display 2D image planes from NumPy-compatible arrays.
- Use axis labels to browse stack dimensions with notebook sliders.
- Mark one axis as channels with `C` and inspect channels individually, as RGB
  overlays, or side by side.
- Configure channel names and Matplotlib-compatible channel colors.
- Adjust display contrast with absolute min/max or percentile thresholds.
- Overlay binary masks or integer labels with independent label display
  settings.
- Downsample large rendered planes by default to keep notebook interaction
  responsive.

Display controls only affect rendering. They do not modify, mask, or rescale the
source array.

## Installation

From another project, install the package into an environment that can run
Jupyter widgets:

```bash
uv add notebook-nd-viewer
```

Or with `pip`:

```bash
pip install notebook-nd-viewer
```

Do not run either command from inside this repository: `uv add` records a
dependency in the current project, and a project should not install itself as a
dependency.

For local development inside this repository:

```bash
git clone https://github.com/Darlokt/notebook-nd-viewer.git
cd notebook-nd-viewer
uv sync --dev
```

If you use classic Notebook, JupyterLab, VS Code notebooks, or another frontend,
make sure that `ipywidgets` is enabled for that environment.

## Usage

```python
import numpy as np

from nb_nd_viewer import view_image_layers, view_labeled_image

image = np.random.default_rng(0).normal(size=(12, 256, 256))

view_image_layers(image, axis_order="ZYX")
```

`axis_order` must contain one label per array dimension. When both `Y` and `X`
are present, they are rendered as the image plane with `Y` as rows and `X` as
columns, no matter where they appear in `axis_order`. Other non-channel axes
become layer sliders.

If either `Y` or `X` is missing, the viewer falls back to positional behavior:
the last two non-channel axes are rendered in their listed order, and earlier
non-channel axes become layer sliders. This keeps arbitrary labels such as `AB`,
`TZR`, or labels with only one of `Y`/`X` working as before.

For channel data, mark the channel axis with `C`:

```python
image = np.random.default_rng(0).normal(size=(8, 3, 256, 256))

view_image_layers(
    image,
    axis_order="ZCYX",
    channel_names=("DAPI", "Actin", "Tubulin"),
    channel_colors=("blue", "green", "magenta"),
)
```

`axis_order="ZCXY"` also renders rows as `Y` and columns as `X`; only the
underlying array-axis positions differ.

Large image planes are downsampled before rendering by default:

```python
view_image_layers(
    image,
    axis_order="ZCYX",
    max_render_pixels=1_000_000,
    render_downsampling="stride",
)
```

Set `max_render_pixels=None` or `render_downsampling="none"` to render every
source pixel.

Use `view_labeled_image` to inspect images with label overlays:

```python
image = np.random.default_rng(0).normal(size=(8, 3, 256, 256))
labels = np.zeros((256, 256), dtype=bool)
labels[80:180, 90:190] = True

view_labeled_image(
    image,
    axis_order="ZCYX",
    labels=labels,
    label_axis_order="YX",
    channel_names=("DAPI", "Actin", "Tubulin"),
    label_names=("Region",),
)
```

Label arrays may omit image stack axes. In the example above, the 2D `YX` mask
is overlaid on every `Z` slice. Multiple binary masks can be stored in one label
array by marking a label channel axis with `C`:

```python
labels = np.zeros((2, 256, 256), dtype=bool)
labels[0, 80:180, 90:190] = True
labels[1, 100:140, 120:220] = True

view_labeled_image(
    image,
    axis_order="ZCYX",
    labels=labels,
    label_axis_order="CYX",
    label_names=("Nuclei", "Cells"),
    label_colors=("cyan", "magenta"),
    label_opacities=(0.4, 0.6),
)
```

Integer labels are rendered with categorical colors from `tab20` by default,
with label value `0` treated as transparent background:

```python
instance_labels = np.zeros((256, 256), dtype=np.uint16)
instance_labels[40:100, 40:100] = 1
instance_labels[130:210, 150:230] = 2

view_labeled_image(
    image,
    axis_order="ZCYX",
    labels=instance_labels,
    label_axis_order="YX",
    label_names=("Instances",),
    integer_label_cmap="tab20",
)
```

Label overlays use the same render pixel budget as image planes, but labels are
always downsampled with nearest-neighbor sampling so binary masks and integer
IDs remain categorical.

## API

```python
view_image_layers(
    image,
    axis_order,
    channel_names=None,
    channel_colors=None,
    figsize=(6, 6),
    cmap="gray",
    *,
    continuous_update=False,
    max_render_pixels=1_000_000,
    render_downsampling="stride",
)
```

```python
view_labeled_image(
    image,
    axis_order,
    labels=None,
    label_axis_order=None,
    channel_names=None,
    label_names=None,
    label_colors=None,
    label_opacities=0.5,
    label_kinds="auto",
    figsize=(6, 6),
    cmap="gray",
    integer_label_cmap="tab20",
    *,
    continuous_update=False,
    max_render_pixels=1_000_000,
    render_downsampling="stride",
)
```

Parameters:

- `image`: NumPy-compatible image array.
- `axis_order`: axis labels matching `image.ndim`; use `C` for the optional
  channel axis. When both `Y` and `X` are present, they define image rows and
  columns; otherwise the last two non-channel axes are rendered.
- `channel_names`: optional names for the channel axis.
- `channel_colors`: optional Matplotlib-compatible colors for channel rendering.
- `figsize`: Matplotlib figure size used before notebook scaling.
- `cmap`: colormap for non-channel images.
- `continuous_update`: redraw continuously while dragging sliders when `True`.
- `max_render_pixels`: maximum rendered pixels per 2D plane; use `None` to
  disable pixel-count based downsampling.
- `render_downsampling`: one of `"stride"`, `"nearest"`, `"bilinear"`,
  `"bicubic"`, or `"none"`.

Additional `view_labeled_image` parameters:

- `labels`: optional bool or integer label array.
- `label_axis_order`: axis labels matching `labels.ndim`; required when labels
  are provided. Use `C` for a label channel axis.
- `label_names`: optional names for expanded label entries.
- `label_colors`: optional Matplotlib-compatible colors for binary label
  rendering.
- `label_opacities`: one opacity for all labels or one value per label.
- `label_kinds`: `"auto"`, `"binary"`, or `"integer"` for all labels, or one
  value per label.
- `integer_label_cmap`: Matplotlib colormap used for integer label IDs.

## Development

This project uses `uv` for dependency management:

```bash
uv sync
```

Run the main quality gates before submitting changes:

```bash
uv run ruff format src tests notebooks
uv run ruff check src tests notebooks
uv run ty check
uv run pytest
```

The default pytest configuration runs unit and integration tests while excluding
tests marked `external`. Tests under `tests/unit/` and `tests/integration/` are
marked automatically from their path, so focused runs can use:

```bash
uv run pytest -m unit
uv run pytest -m integration
```

Coverage is configured with a 90% minimum:

```bash
uv run pytest --cov
```

Install the repository's pre-commit hooks with:

```bash
uv run prek install
```

## Project Layout

```text
notebook-nd-viewer/
├── src/nb_nd_viewer/    # importable package and private viewer helpers
├── tests/unit/          # focused unit tests
├── tests/integration/   # public API and workflow tests
├── notebooks/           # Example notebooks
├── pyproject.toml       # package metadata and tool configuration
└── uv.lock              # locked development environment
```
