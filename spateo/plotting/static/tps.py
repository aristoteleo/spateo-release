def tps(reference, to_align):
    """Plot the warp grid figure of the thin plate spline interpolation"""

    import matplotlib.pyplot as plt

    x_min, x_max = reference[:, 0].min(), reference[:, 0].max()
    y_min, y_max = reference[:, 1].min(), reference[:, 1].max()

    tps.fit(reference, to_align)

    # Transform new points
    x_step, y_step = (x_max - x_min + 2) / 10, (y_max - y_min + 2) / 10
    x, y = np.meshgrid(np.arange(x_min - 1, x_max + 1, x_step), np.arange(y_min - 1, y_max + 1, y_step))
    predict = tps.transform(np.vstack((x.ravel(), y.ravel())).T)

    plt.scatter(*predict.T)
    x_predict_grid, y_predict_grid = predict[:, 0].reshape((x.shape)), predict[:, 1].reshape((x.shape))
    for i in range(y_predict_grid.shape[0]):
        plt.plot(y_predict_grid[i, :], y_predict_grid[i, :], c="r")

    for i in range(x_predict_grid.shape[1]):
        plt.plot(x_predict_grid[:, i], x_predict_grid[:, i], c="r")

    plt.scatter(*reference.T, alpha=0.1)
    plt.show()
