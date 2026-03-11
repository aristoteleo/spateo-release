# 3D Reconstruction from Multiple Tissue Slices

This guide provides comprehensive information on how to build 3D models from multiple 2D spatial transcriptomics slices using Spateo.

## Overview

Spateo provides robust tools for reconstructing 3D structures from sequential 2D tissue sections. The workflow involves:

1. **Slice Alignment**: Align 2D slices in the xy-plane using Spateo's alignment functions
2. **Z-Coordinate Assignment**: Calculate and assign appropriate z-axis coordinates for each slice
3. **3D Model Construction**: Combine aligned slices into a coherent 3D model
4. **Visualization and Analysis**: Explore the 3D structure using Spateo's visualization tools

## Z-Axis Coordinate Calculation

### Understanding Z-Spacing

When reconstructing 3D models from 2D slices, one of the most critical decisions is determining the spacing between slices along the z-axis. Spateo provides the `assign_z_coordinates` function to handle this systematically.

### Basic Usage

```python
import spateo as st

# Load your aligned 2D slices
slices = [slice1, slice2, slice3, slice4]  # List of AnnData objects

# Assign z-coordinates with default spacing (1.0 unit between slices)
st.tl.assign_z_coordinates(slices, spatial_key="align_spatial")
```

### Z-Spacing Strategies

Spateo supports three main strategies for z-coordinate assignment:

#### 1. Uniform Spacing (Default)

Use this when slices are evenly spaced or when relative positioning is more important than absolute measurements:

```python
# Default uniform spacing of 1.0
st.tl.assign_z_coordinates(slices, spatial_key="align_spatial")

# Custom uniform spacing
st.tl.assign_z_coordinates(slices, spatial_key="align_spatial", z_spacing=10.0)
```

**Best for:**
- Exploratory analysis where relative positions matter most
- When exact tissue thickness is unknown
- Consistent section thickness across all slices

#### 2. Tissue Thickness-Based Spacing

Use this when you know the physical thickness of your tissue sections:

```python
# If your tissue sections are 15 µm thick
st.tl.assign_z_coordinates(
    slices,
    spatial_key="align_spatial",
    tissue_thickness=15.0
)
```

**Best for:**
- When physical measurements are available
- Quantitative spatial analysis requiring accurate distances
- Comparing structures across different specimens

#### 3. Custom Variable Spacing

Use this when different slices have different spacing (e.g., non-uniform sectioning):

```python
# For 4 slices, provide 3 spacing values
# Spacing between: slice0-1: 10, slice1-2: 12, slice2-3: 10
st.tl.assign_z_coordinates(
    slices,
    spatial_key="align_spatial",
    z_spacing=[10.0, 12.0, 10.0]
)
```

**Best for:**
- Non-uniform section thickness
- Missing sections (use larger spacing to represent gaps)
- Variable sampling intervals

## Complete 3D Reconstruction Workflow

### Step 1: Prepare 2D Slices

```python
import spateo as st
import scanpy as sc

# Load your spatial transcriptomics data
slices = []
for i, file_path in enumerate(slice_files):
    adata = sc.read_h5ad(file_path)
    # Add slice identifier
    adata.obs['slice_id'] = i
    slices.append(adata)
```

### Step 2: Align Slices in 2D

```python
# Perform pairwise alignment
aligned_slices = []
for i in range(len(slices) - 1):
    # Align slice i+1 to slice i
    st.tl.morpho_align(
        sliceA=slices[i],
        sliceB=slices[i+1],
        spatial_key='spatial',
        key_added='align_spatial',
        # Additional alignment parameters
    )
    aligned_slices.append(slices[i])

aligned_slices.append(slices[-1])  # Add the last slice
```

### Step 3: Assign Z-Coordinates

```python
# Choose your z-spacing strategy
# Option A: Using known tissue thickness
st.tl.assign_z_coordinates(
    aligned_slices,
    spatial_key='align_spatial',
    tissue_thickness=15.0,  # 15 µm sections
    z_offset=0.0
)

# Option B: Using custom spacing for each gap
st.tl.assign_z_coordinates(
    aligned_slices,
    spatial_key='align_spatial',
    z_spacing=[10.0, 10.0, 15.0],  # Variable spacing
    z_offset=0.0
)
```

### Step 4: Visualize in 3D

```python
# Visualize all slices together in 3D
st.pl.three_d_multi_plot.multi_models(
    *aligned_slices,
    spatial_key='align_spatial',
    mode='single',  # or 'overlap' or 'both'
    show_model=True
)
```

## Best Practices and Recommendations

### Determining Appropriate Z-Spacing

1. **Physical Measurements First**: If you know the tissue thickness from your experimental protocol, use that value.

2. **Relative to XY Scale**: Consider the scale of your xy coordinates. If your spatial coordinates are in pixels and each pixel represents 1 µm, ensure your z_spacing uses the same units.

3. **Visual Inspection**: After initial reconstruction, visually inspect the 3D model:
   ```python
   # If slices appear too compressed, increase z_spacing
   # If slices appear too separated, decrease z_spacing
   ```

4. **Biological Validation**: Check if known structures appear continuous across slices. Discontinuities may indicate:
   - Incorrect z-spacing
   - Missing sections that need larger gaps
   - Need for non-uniform spacing

### Common Pitfalls

1. **Unit Mismatch**: Ensure z_spacing uses the same units as your xy coordinates
   ```python
   # If xy coordinates are in µm, z_spacing should also be in µm
   # If xy coordinates are in pixels, convert appropriately
   ```

2. **Missing Sections**: Account for missing tissue sections by using larger z-spacing
   ```python
   # If section 2 is missing between sections 1 and 3
   z_spacing = [10.0, 20.0, 10.0]  # Double spacing for the gap
   ```

3. **Coordinate Systems**: Ensure all slices use the same spatial coordinate system after alignment

## Advanced Topics

### Combining with Global Alignment Refinement

For more accurate 3D reconstruction with many slices:

```python
# First, perform pairwise alignment and assign z-coordinates
# Then, use global refinement to minimize cumulative errors
st.tl.morpho_align_ref(
    models=aligned_slices,
    spatial_key='align_spatial',
    # Global refinement parameters
)
```

### Adjusting Z-Coordinates Post-Reconstruction

If you need to modify z-coordinates after initial assignment:

```python
# Re-run with different parameters
st.tl.assign_z_coordinates(
    aligned_slices,
    spatial_key='align_spatial',
    z_spacing=20.0,  # New spacing
    inplace=True
)
```

### Working with Different Coordinate Keys

You can maintain both 2D and 3D coordinates:

```python
# Keep original 2D coordinates in 'spatial'
# Create 3D coordinates in 'spatial_3d'

# First copy 2D coordinates
for adata in slices:
    adata.obsm['spatial_3d'] = adata.obsm['align_spatial'].copy()

# Then assign z-coordinates to the 3D version
st.tl.assign_z_coordinates(
    slices,
    spatial_key='spatial_3d',
    tissue_thickness=15.0
)
```

## Example Workflows

### Example 1: Standard Uniform Sections

```python
# 10 tissue sections, each 20 µm thick
slices = load_aligned_slices()  # Your data loading function

st.tl.assign_z_coordinates(
    slices,
    spatial_key='align_spatial',
    tissue_thickness=20.0,
    z_offset=0.0
)

# Visualize
st.pl.three_d_multi_plot.multi_models(
    *slices,
    spatial_key='align_spatial',
    group_key='cell_type'
)
```

### Example 2: Non-Uniform Sections with Gaps

```python
# 5 sections with known spacing: 15, 15, 30 (missing section), 15 µm
slices = load_aligned_slices()

st.tl.assign_z_coordinates(
    slices,
    spatial_key='align_spatial',
    z_spacing=[15.0, 15.0, 30.0, 15.0],  # Note: n-1 spacing values for n slices
    z_offset=0.0
)
```

### Example 3: Pixel-Based Coordinates

```python
# If spatial coordinates are in pixels, and each pixel = 0.5 µm
# Tissue sections are 20 µm thick
slices = load_aligned_slices()

pixel_size = 0.5  # µm per pixel
section_thickness_um = 20.0  # µm
section_thickness_pixels = section_thickness_um / pixel_size  # 40 pixels

st.tl.assign_z_coordinates(
    slices,
    spatial_key='align_spatial',
    tissue_thickness=section_thickness_pixels,
    z_offset=0.0
)
```

## API Reference

### assign_z_coordinates

```python
st.tl.assign_z_coordinates(
    adatas,
    spatial_key='spatial',
    z_spacing=None,
    tissue_thickness=None,
    z_offset=0.0,
    inplace=True
)
```

**Parameters:**

- `adatas`: AnnData or List[AnnData] - Single or list of AnnData objects with 2D spatial coordinates
- `spatial_key`: str - Key in adata.obsm where spatial coordinates are stored
- `z_spacing`: float, List[float], or None - Spacing between slices
- `tissue_thickness`: float or None - Physical tissue thickness (overrides z_spacing if provided)
- `z_offset`: float - Starting z-coordinate for the first slice
- `inplace`: bool - Whether to modify input objects in place

**Returns:**

- None if inplace=True, otherwise modified AnnData object(s)

## See Also

- [Spatial Transcriptomics Alignment](spatial_transcriptomics_alignment.md) - Details on 2D slice alignment
- API Documentation: `st.tl.assign_z_coordinates`
- API Documentation: `st.tl.morpho_align`
- API Documentation: `st.pl.three_d_multi_plot`

## References

For more information about the 3D reconstruction methods in Spateo, please refer to:

Qiu, X., Zhu, D.Y., Lu, Y. et al. Spatiotemporal modeling of molecular holograms. *Cell* (2024). https://doi.org/10.1016/j.cell.2024.10.022
