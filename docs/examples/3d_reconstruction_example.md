# 3D Reconstruction Example: Assigning Z-Coordinates to Multiple Slices

This example demonstrates how to use Spateo's `assign_z_coordinates` function to build 3D models from multiple 2D spatial transcriptomics slices.

## Quick Start Example

```python
import spateo as st
import scanpy as sc

# Load your aligned 2D slices
# Assuming you have already aligned them using st.tl.morpho_align
slices = [
    sc.read_h5ad("aligned_slice_0.h5ad"),
    sc.read_h5ad("aligned_slice_1.h5ad"),
    sc.read_h5ad("aligned_slice_2.h5ad"),
    sc.read_h5ad("aligned_slice_3.h5ad"),
]

# Method 1: Default uniform spacing (spacing = 1.0)
st.tl.assign_z_coordinates(slices, spatial_key="align_spatial")

# Method 2: Use known tissue thickness (e.g., 20 µm sections)
st.tl.assign_z_coordinates(
    slices,
    spatial_key="align_spatial",
    tissue_thickness=20.0
)

# Method 3: Custom uniform spacing
st.tl.assign_z_coordinates(
    slices,
    spatial_key="align_spatial",
    z_spacing=15.0
)

# Method 4: Variable spacing (e.g., if section 2 is missing)
st.tl.assign_z_coordinates(
    slices,
    spatial_key="align_spatial",
    z_spacing=[20.0, 40.0, 20.0]  # Double spacing where section is missing
)

# Visualize the 3D reconstruction
st.pl.three_d_multi_plot.multi_models(
    *slices,
    spatial_key="align_spatial",
    group_key="cell_type",
    mode="single"
)
```

## Complete Workflow Example

```python
import spateo as st
import scanpy as sc
import numpy as np

# Step 1: Load your raw 2D slices
slice_files = [
    "slice_0.h5ad",
    "slice_1.h5ad", 
    "slice_2.h5ad",
    "slice_3.h5ad"
]

slices = []
for i, file_path in enumerate(slice_files):
    adata = sc.read_h5ad(file_path)
    adata.obs['slice_id'] = i
    slices.append(adata)

# Step 2: Perform pairwise alignment
# Align each slice to the previous one
aligned_slices = [slices[0]]  # First slice is the reference

for i in range(1, len(slices)):
    st.tl.morpho_align(
        sliceA=slices[i],
        sliceB=aligned_slices[-1],
        spatial_key='spatial',
        key_added='align_spatial',
        # Additional alignment parameters
        n_sampling=2000,
    )
    aligned_slices.append(slices[i])

# Step 3: Assign z-coordinates based on tissue thickness
# If you know your sections are 15 µm thick
st.tl.assign_z_coordinates(
    aligned_slices,
    spatial_key='align_spatial',
    tissue_thickness=15.0,
    z_offset=0.0
)

# Step 4: Optionally perform global refinement to reduce cumulative errors
st.tl.morpho_align_ref(
    models=aligned_slices,
    spatial_key='align_spatial',
    # Global refinement parameters
)

# Step 5: Visualize the 3D model
st.pl.three_d_multi_plot.multi_models(
    *aligned_slices,
    spatial_key='align_spatial',
    group_key='cell_type',
    mode='single',
    show_model=True
)
```

## Working with Different Coordinate Systems

If your spatial coordinates are in pixels and you want to convert to physical units:

```python
# Your imaging parameters
pixel_size_um = 0.5  # Each pixel represents 0.5 µm
section_thickness_um = 20.0  # Tissue sections are 20 µm thick

# Calculate spacing in pixel units
section_thickness_pixels = section_thickness_um / pixel_size_um  # 40 pixels

# Assign z-coordinates in pixel units
st.tl.assign_z_coordinates(
    slices,
    spatial_key='align_spatial',
    tissue_thickness=section_thickness_pixels,
    z_offset=0.0
)
```

## Handling Missing Sections

When you have missing tissue sections, use variable spacing:

```python
# You have 5 slices, but section 2 is missing
# Normal spacing is 20 µm, so missing section = 40 µm gap

st.tl.assign_z_coordinates(
    slices,
    spatial_key='align_spatial',
    z_spacing=[20.0, 40.0, 20.0, 20.0]  # Larger gap where section is missing
)
```

## Best Practices

### 1. **Match Units**
Always ensure z-spacing uses the same units as your xy coordinates:
```python
# If xy coordinates are in µm, use µm for z-spacing
# If xy coordinates are in pixels, convert appropriately
```

### 2. **Validate Visually**
After assigning z-coordinates, always visualize to check:
```python
# If slices appear too compressed or too separated, adjust z_spacing
st.pl.three_d_multi_plot.multi_models(*slices, spatial_key='align_spatial')
```

### 3. **Use Inplace Wisely**
```python
# Keep original data unchanged
slices_3d = st.tl.assign_z_coordinates(slices, inplace=False)

# Modify in place to save memory
st.tl.assign_z_coordinates(slices, inplace=True)
```

### 4. **Check Biological Structures**
Verify that known continuous structures (e.g., blood vessels, tissue layers) appear continuous across slices. If not, adjust your z-spacing or check alignment quality.

## Troubleshooting

**Issue: Slices appear too close together**
```python
# Increase z_spacing
st.tl.assign_z_coordinates(slices, z_spacing=50.0)  # Larger spacing
```

**Issue: Slices appear too far apart**
```python
# Decrease z_spacing
st.tl.assign_z_coordinates(slices, z_spacing=5.0)  # Smaller spacing
```

**Issue: Known structures don't align**
- Check your 2D alignment quality first
- Consider using global refinement with `st.tl.morpho_align_ref`
- Verify that you're using the correct spatial_key

## See Also

- [3D Reconstruction Technical Documentation](../technicals/3d_reconstruction.md)
- API Documentation: `st.tl.assign_z_coordinates`
- API Documentation: `st.tl.morpho_align`
- API Documentation: `st.pl.three_d_multi_plot`
