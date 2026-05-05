import joblib
from utilities import samplers, read_write, plot_utils
from utilities import distribution_functions as dfs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm


def plot_sliced_ishigami(samples, y, x3_labels, normalize=True, fig_title=None):
    """
    Visualize 2D slices of the Ishigami function (x1, x2) for fixed values of x3.

    Parameters
    ----------
    samples : numpy.ndarray of shape (2, N)
        Grid points for the first two input dimensions (x1, x2), typically
        generated from a meshgrid and flattened. N is assumed to be a perfect
        square (N = n_grid^2).

    y : numpy.ndarray of shape (6, N)
        Function values evaluated on the grid for six different fixed values
        of x3. Each row corresponds to one slice of the function.

    x3_labels : array-like of length 6
        Labels representing the fixed values of x3 used for each slice.
        These are displayed in the subplot titles.

    normalize : bool, optional (default=True)
        If True, uses analytically derived bounds of the Ishigami function
        to fix the color scale across all plots. If False, the color scale
        is computed from the minimum and maximum of `y`.

    Returns
    -------
    None
        Displays a 2x3 grid of filled contour plots with a shared colorbar.
        Each subplot shows the dependence of the function on (x1, x2) for
        a fixed x3 value.
    """
    n_plots = y.shape[0]
    n_grid = int(np.sqrt(samples.shape[1]))

    x = samples[0].reshape(n_grid, n_grid)
    y_coords = samples[1].reshape(n_grid, n_grid)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(fig_title)
    axes = axes.flatten()

    # Analytical min/max
    if normalize:
        vmin = -1 - (np.pi**4) / 10
        vmax = 8 + (np.pi**4) / 10

    else:
        vmin = y.min()
        vmax = y.max()
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    for i in range(n_plots):
        z = y[i].reshape(n_grid, n_grid)
        im = axes[i].contourf(
            x, y_coords, z, levels=50, vmin=vmin, vmax=vmax, cmap="plasma", norm=norm
        )
        axes[i].set_title(f"$x_3 = {x3_labels[i]}$", loc="left")
        if i in [3, 4, 5]:
            axes[i].set_xlabel(r"$x_1$")
        if i in [0, 3]:
            axes[i].set_ylabel(r"$x_2$")

    # Make space on the right for the colorbar
    fig.subplots_adjust(right=0.88, hspace=0.35)
    # Add a dedicated colorbar axis [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.90, 0.18, 0.02, 0.74])

    sm = cm.ScalarMappable(norm=norm, cmap=cm.plasma)
    fig.colorbar(sm, cax=cbar_ax)

    # Define tick locations and labels
    ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    tick_labels = [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"]

    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)


def main(gp_dir, gp_names):
    # Paths
    gp_dir_path = f"output/gp_models/{gp_dir}"

    # Load all GP models into a dict
    gps = {}
    for name in gp_names:
        gp_dict = joblib.load(f"{gp_dir_path}/{name}")
        gps[name] = gp_dict["gp"]

    # Setup
    setup = read_write.json_read_dictionary(f"{gp_dir_path}/setup.json")

    # Slices
    slices = np.array(
        [-np.pi, -2 * np.pi / 3, -np.pi / 3, np.pi / 3, 2 * np.pi / 3, np.pi]
    )
    x3_labels = ["-π", "-2π/3", "-π/3", "π/3", "2π/3", "π"]

    # Grid
    n_grid = 100
    samples = samplers.get_grid_samples(
        [np.linspace(-np.pi, np.pi, n_grid), np.linspace(-np.pi, np.pi, n_grid)]
    )
    samples = np.array(samples)

    # Storage
    y_preds = {name: np.zeros([len(slices), n_grid**2]) for name in gp_names}
    y_stds = {name: np.zeros([len(slices), n_grid**2]) for name in gp_names}
    y_true = np.zeros([len(slices), n_grid**2])

    # Loop slices
    for i, sliced in enumerate(slices):
        new_row = sliced * np.ones((1, n_grid**2))
        x_samples = np.vstack([samples, new_row])

        # Predict for each GP
        for name, gp in gps.items():
            y_preds[name][i], y_stds[name][i] = gp.predict(x_samples.T, return_std=True)

        # True values
        y_true[i] = dfs.ishigami(*x_samples)

    # Plot predictions
    for name in gp_names:
        plot_sliced_ishigami(samples, y_preds[name], x3_labels)

    # Plot true
    plot_sliced_ishigami(samples, y_true, x3_labels, fig_title="True Ishigami function")

    # Plot residuals
    for name in gp_names:
        plot_sliced_ishigami(
            samples,
            np.abs(y_preds[name] - y_true),
            x3_labels,
            normalize=False,
        )

    input("Press Enter to exit...")


if __name__ == "__main__":
    plot_utils.set_nes_plot_style(large_font=False)

    gp_directory = "noisy_ishigami"
    models = [f"{gp_directory}_{i}.joblib" for i in range(6)]
    main(gp_directory, models)
