import math
from typing import List, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from anndata import AnnData

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ...tools.cluster.utils import integrate, to_dense_matrix
from ...tools.utils import compute_smallest_distance
from .utils import save_return_show_fig_utils


def multi_slices(
    slices: Union[AnnData, List[AnnData]],
    slices_key: Optional[str] = None,
    label: Optional[str] = None,
    spatial_key: str = "align_spatial",
    layer: str = "X",
    point_size: Optional[float] = None,
    font_size: Optional[float] = 20,
    color: Optional[str] = "skyblue",
    palette: Optional[str] = None,
    alpha: float = 1.0,
    ncols: int = 4,
    ax_height: float = 1,
    dpi: int = 100,
    show_legend: bool = True,
    save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
    save_kwargs: Optional[dict] = None,
    **kwargs,
):

    # Check slices object.
    if isinstance(slices, list):
        adatas = [s.copy() for s in slices]
        for i, s in enumerate(adatas):
            s.X = s.layers[layer].copy() if layer != "X" else s.X.copy()
            s.uns = {"__type": "UMI"}
            if slices_key is None:
                slices_key = "slices"
                s.obs[slices_key] = f"slice_{i}"
        adata = integrate(adatas=adatas, batch_key=slices_key)
    else:
        assert slices_key != None, "When `slices` is an anndata object, `slices_key` cannot be None."
        adata = slices.copy()
        adata.X = adata.layers[layer].copy() if layer != "X" else adata.X.copy()

    # Check label data and generate plotting data.
    slices_data = pd.DataFrame(adata.obsm[spatial_key][:, :2], columns=["x", "y"], dtype=float)
    slices_data[slices_key] = adata.obs[slices_key].values

    if label is None:
        label = "spatial coordinates"
        slices_data[label] = label
    elif label in adata.obs_keys():
        slices_data[label] = adata.obs[label].values
    elif label in adata.var_names:
        adata.X = to_dense_matrix(adata.X)
        slices_data[label] = adata[:, label].X
    else:
        raise ValueError("`label` is not a valid column names or gene name.")

    # Set the arrangement of subgraphs
    slices_id = slices_data[slices_key].unique().tolist()
    nrows = math.ceil(len(slices_id) / ncols)

    # Set the aspect ratio of each subplot
    spatial_coords = slices_data[["x", "y"]].values.copy()
    ptp_vec = spatial_coords.ptp(0)
    aspect_ratio = ptp_vec[0] / ptp_vec[1]
    ax_height = 2 if nrows == 1 and ax_height == 1 else ax_height
    axsize = (ax_height * aspect_ratio, ax_height * 2)

    # Set multi-plot grid for plotting.
    sns.set_theme(
        context="paper",
        style="white",
        font="Arial",
        font_scale=1,
        rc={
            "font.size": font_size,
            "font.family": ["sans-serif"],
            "font.sans-serif": ["Arial", "sans-serif", "Helvetica", "DejaVu Sans", "Bitstream Vera Sans"],
        },
    )

    g = sns.FacetGrid(
        slices_data.copy(),
        col=slices_key,
        hue=label,
        palette=palette,
        sharex=True,
        sharey=True,
        height=axsize[1] * nrows,
        aspect=aspect_ratio,
        col_wrap=ncols,
        despine=False,
    )

    # Calculate the most suitable size of the point.
    if point_size is None:
        group_slices_data = slices_data.groupby(by=slices_key)
        min_dist_list = []
        for key, data in group_slices_data:
            sample_num = 1000 if len(data) > 1000 else len(data)
            min_dist_list.append(compute_smallest_distance(coords=data[["x", "y"]].values, sample_num=sample_num))

        point_size = min(min_dist_list) * axsize[0] / ptp_vec[0] * dpi
        point_size = point_size**2 * ncols * nrows

    # Draw scatter plots.
    g.map_dataframe(sns.scatterplot, x="x", y="y", alpha=alpha, color=color, s=point_size, legend="brief", **kwargs)

    # Set legend.
    label_values = slices_data[label].values
    if label_values.dtype in ["float16", "float32", "float64", "int16", "int32", "int64"]:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        ax = g.facet_axis(row_i=0, col_j=ncols - 1)
        norm = mpl.colors.Normalize(vmin=None, vmax=None)
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=palette)
        mappable.set_array(label_values)
        plt.colorbar(
            mappable,
            cax=inset_axes(
                ax,
                width="12%",
                height="100%",
                loc="center left",
                bbox_to_anchor=(1.02, 0.0, 0.5, 1.0),
                bbox_transform=ax.transAxes,
                borderpad=1.85,
            ),
            ax=ax,
            orientation="vertical",
            alpha=alpha,
            label=label,
        )
    else:
        g.add_legend()

    plt.tight_layout()
    return save_return_show_fig_utils(
        save_show_or_return=save_show_or_return,
        show_legend=show_legend,
        background="white",
        prefix="multi_slices",
        save_kwargs=save_kwargs,
        total_panels=len(slices_id),
        fig=g,
        axes=g,
        return_all=False,
        return_all_list=None,
    )
