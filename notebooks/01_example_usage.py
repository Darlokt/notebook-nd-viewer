# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Example Usage

# %% [markdown]
# ## Imports

# %%
from __future__ import annotations

from pathlib import Path

import tifffile as tif

from nb_nd_viewer import view_image_layers

# %% [markdown]
# ## Load data

# %%
path_to_image = Path("path" / "to" / "example_image.ome.tif")

raw_image = tif.imread(path_to_image)

print(f"Image shape: {raw_image.shape}")
print(f"Image dtype: {raw_image.dtype}")

# channel order is (Z, C, Y, X) for this dataset

# %%
view_image_layers(
    raw_image,
    axis_order="ZCYX",
    continuous_update=True,
    max_render_pixels=1000000,
    render_downsampling="stride",
)
