import joblib
from ishigami_sensitivity import list_model_names
from utilities import samplers, read_write, plot_utils
from utilities import distribution_functions as dfs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm


def plot_sliced_ishigami(samples, y, x3_labels, normalize=True):
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


def main(gp_dir, gp_name_1, gp_name_2):
    # GP models
    gp_dir_path = f"output/gp_models/{gp_dir}"
    gp_model_names = list_model_names(gp_dir)
    gp_dict_1 = joblib.load(f"{gp_dir_path}/{gp_name_1}")
    gp_1 = gp_dict_1["gp"]
    gp_dict_2 = joblib.load(f"{gp_dir_path}/{gp_name_2}")
    gp_2 = gp_dict_2["gp"]

    # Setup
    setup = read_write.json_read_dictionary(f"output/gp_models/{gp_dir}/setup.json")

    # Fixed x3 value for sliced 3D space
    slices = np.array(
        [-np.pi, -2 * np.pi / 3, -np.pi / 3, np.pi / 3, 2 * np.pi / 3, np.pi]
    )
    x3_labels = [
        "-\pi",
        "-2 \pi / 3",
        "-\pi / 3",
        "\pi / 3",
        "2 \pi / 3",
        "\pi",
    ]

    # x1, x2 samples
    n_grid = 100
    samples = samplers.get_grid_samples(
        [np.linspace(-np.pi, np.pi, n_grid), np.linspace(-np.pi, np.pi, n_grid)]
    )
    samples = np.array(samples)

    # Predict y-values using GP regression and true function
    y_preds_1 = np.zeros([len(slices), n_grid**2])
    y_stds_1 = np.zeros([len(slices), n_grid**2])
    y_preds_2 = np.zeros([len(slices), n_grid**2])
    y_stds_2 = np.zeros([len(slices), n_grid**2])
    y_true = np.zeros([len(slices), n_grid**2])
    for i, sliced in enumerate(slices):
        new_row = sliced * np.ones((1, n_grid**2))
        x_samples = np.vstack([samples, new_row])

        # GP regression prediction
        y_preds_1[i], y_stds_1[i] = gp_1.predict(x_samples.T, return_std=True)
        y_preds_2[i], y_stds_2[i] = gp_2.predict(x_samples.T, return_std=True)

        # True values
        y_true[i] = dfs.ishigami(*x_samples)

    # Plot GP prediction and true
    plot_sliced_ishigami(samples, y_preds_1, x3_labels)
    plot_sliced_ishigami(samples, y_preds_2, x3_labels)
    plot_sliced_ishigami(samples, y_true, x3_labels)

    # Plot residuals
    plot_sliced_ishigami(
        samples, np.abs(y_preds_1 - y_true), x3_labels, normalize=False
    )
    plot_sliced_ishigami(
        samples, np.abs(y_preds_2 - y_true), x3_labels, normalize=False
    )

    input("Press Enter to exit...")


if __name__ == "__main__":
    plot_utils.set_nes_plot_style(large_font=True)
    main(
        "noiseless_ishigami",
        "noiseless_ishigami_0.joblib",
        "noiseless_ishigami_5.joblib",
    )
