import matplotlib.pyplot as plt
import seaborn as sns
from geopandas import GeoDataFrame
from matplotlib import colors


def lisa_quantiles(df: GeoDataFrame):
    """Scatterplot of the gene expression and its spatial lag. Categorization into the four regions (HH, HL, LH, LL) is
    also annotated. This function should be used in conjunction with `st.tl.lisa_geo_df`.

    Args:
        df: The GeoDataFrame returned from running `st.tl.lisa_geo_df(adata, gene)`.

    Returns:
        Nothing but plot the scatterplot.
    """
    # Setup the figure and axis
    f, ax = plt.subplots(1, figsize=(6, 6))
    # Plot values
    sns.regplot(x="exp_zscore", y="w_exp_zscore", data=df, ci=None, color="red")
    # Add vertical and horizontal lines
    plt.axvline(0, c="k", alpha=0.5)
    plt.axhline(0, c="k", alpha=0.5)
    # Add text labels for each quadrant
    plt.text(1, 1.5, "HH", fontsize=25)
    plt.text(1, -1.5, "HL", fontsize=25)
    plt.text(-1.5, 1.5, "LH", fontsize=25)
    plt.text(-1.5, -1.5, "LL", fontsize=25)
    # Display
    plt.show()


def lisa(df: GeoDataFrame):
    """Create a plot with four panels. The first one is for the raw lisa (Local Indicators of Spatial Association )
    score. The second one (the right one in the first row) is for the four quantiles. The third one is for the
    significance while the fourth one the five categories (not significant, hotspot, doughnut, coldspot and diamond.

    Args:
        df: The GeoDataFrame returned from running `st.tl.lisa_geo_df(adata, gene)`.

    Returns:
        Nothing but plot the four panels .
    """
    # Set up figure and axes
    f, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    # Make the axes accessible with single indexing
    axs = axs.flatten()

    # Subplot 1: raw lisa score
    ax = axs[0]
    df.plot(
        column="Is",
        cmap="viridis",
        scheme="quantiles",
        k=5,
        edgecolor="white",
        linewidth=0.1,
        alpha=0.75,
        legend=True,
        ax=ax,
    )
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Subplot 2: four quantiles
    ax = axs[1]
    hmap = colors.ListedColormap(["red", "lightblue", "blue", "pink"])
    df.plot(column="labels", categorical=True, k=2, cmap=hmap, linewidth=0.1, ax=ax, edgecolor="white", legend=True)

    ax.set_aspect("equal")
    ax.set_axis_off()

    # Subplot 3: significance
    ax = axs[2]
    hmap = colors.ListedColormap(["grey", "black"])

    df.plot(column="sig", categorical=True, k=2, cmap=hmap, linewidth=0.1, ax=ax, edgecolor="white", legend=True)

    ax.set_aspect("equal")
    ax.set_axis_off()

    # Subplot 4: five categories
    ax = axs[3]
    hmap = colors.ListedColormap(["grey", "red", "lightblue", "blue", "pink"])

    df.plot(column="group", categorical=True, k=2, cmap=hmap, linewidth=0.1, ax=ax, edgecolor="white", legend=True)

    ax.set_aspect("equal")
    ax.set_axis_off()

    # Display the figure
    plt.show()
