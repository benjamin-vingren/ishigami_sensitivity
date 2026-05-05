# %%
import numpy as np
import utilities.read_write as rw
import utilities.plot_utils as pu
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from ishigami_sensitivity import list_model_names
from plot_ishigami_sensitivity import analytical_solution, get_sobol_indices


def plot_quantiles(quantiles_list, n_samples, standard_solution):
    """
    Plot KDE-based median and quantiles as function of number of samples.

    Parameters
    ----------
    quantiles_list : list of dict
        Output from plot_histograms (one dict per model).
    sens_models : list of str
        Model names containing number of samples (used for x-axis).
    """

    # Keys (e.g. S1, S2, S3, ST1, ...)
    keys = quantiles_list[0].keys()
    key_to_label = {
        "S1": r"$S_1$",
        "S2": r"$S_2$",
        "S3": r"$S_3$",
        "ST1": r"$S_{T_1}$",
        "ST2": r"$S_{T_2}$",
        "ST3": r"$S_{T_3}$",
        "S12": r"$S_{12}$",
        "S13": r"$S_{13}$",
        "S23": r"$S_{23}$",
    }

    # Analytical solution to Sobol indices
    analytical_solution_dict = analytical_solution(7, 0.1, return_dict=True)

    standard_solution_dict = {
        "S1": [
            get_sobol_indices("S1", 0, standard_solution),
            get_sobol_indices("S1_conf", 0, standard_solution),
        ],
        "S2": [
            get_sobol_indices("S1", 1, standard_solution),
            get_sobol_indices("S1_conf", 1, standard_solution),
        ],
        "S3": [
            get_sobol_indices("S1", 2, standard_solution),
            get_sobol_indices("S1_conf", 2, standard_solution),
        ],
        "ST1": [
            get_sobol_indices("ST", 0, standard_solution),
            get_sobol_indices("ST_conf", 0, standard_solution),
        ],
        "ST2": [
            get_sobol_indices("ST", 1, standard_solution),
            get_sobol_indices("ST_conf", 1, standard_solution),
        ],
        "ST3": [
            get_sobol_indices("ST", 2, standard_solution),
            get_sobol_indices("ST_conf", 2, standard_solution),
        ],
        "S12": [
            get_sobol_indices("S2", [0, 1], standard_solution),
            get_sobol_indices("S2_conf", [0, 1], standard_solution),
        ],
        "S13": [
            get_sobol_indices("S2", [0, 2], standard_solution),
            get_sobol_indices("S2_conf", [0, 2], standard_solution),
        ],
        "S23": [
            get_sobol_indices("S2", [1, 2], standard_solution),
            get_sobol_indices("S2_conf", [1, 2], standard_solution),
        ],
    }

    for key in keys:
        q_low = []
        q_med = []
        q_high = []

        for qdict in quantiles_list:
            low, med, high = qdict[key]
            q_low.append(low)
            q_med.append(med)
            q_high.append(high)

        q_low = np.array(q_low)
        q_med = np.array(q_med)
        q_high = np.array(q_high)

        plt.figure(f"{key} vs samples")
        plt.title(f"{key} vs number of samples", loc="left")

        # Analytical solution
        plt.gca().axhline(analytical_solution_dict[key], linestyle="--", color="k")

        # Standard solution
        plt.errorbar(
            n_samples,
            standard_solution_dict[key][0],
            yerr=standard_solution_dict[key][1],
            linestyle="None",
            marker=".",
            label="Standard solution",
            color="C1",
        )

        # Probabilistic solution
        plt.errorbar(
            n_samples,
            q_med,
            yerr=[q_med - q_low, q_high - q_med],
            linestyle="None",
            marker=".",
            label="GP predicted solution",
            color="C0",
        )

        plt.xlabel("Number of samples")
        plt.ylabel(key_to_label[key])
        plt.xscale("log")
        plt.xlim(1e2, 1e4)

        plt.legend(
            facecolor="lightgray",
            edgecolor="black",
            framealpha=0.3,
            frameon=True,
        )


def kde_quantiles(data, q=[0.025, 0.5, 0.975]):
    kde = gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 1000)
    pdf = kde(x)

    # Normalize PDF → CDF
    dx = x[1] - x[0]
    cdf = np.cumsum(pdf) * dx
    cdf /= cdf[-1]

    return x, pdf, [np.interp(qi, cdf, x) for qi in q]


def plot_hist_with_kde(ax, data, label, bins, true_value):
    counts, bin_edges, _ = ax.hist(data, bins=bins, alpha=0.6)

    # KDE + quantiles
    x, pdf, (q_low, q_med, q_high) = kde_quantiles(data)

    bin_width = bin_edges[1] - bin_edges[0]
    kde_scaled = pdf * len(data) * bin_width

    ax.plot(x, kde_scaled, label="KDE")

    # Vertical lines
    for q in [q_low, q_med, q_high]:
        ax.axvline(q, linestyle="--", color="black")
    ax.axvline(true_value, linestyle="-", color="red", label="True value")

    # Format title
    upper = q_high - q_med
    lower = q_med - q_low

    title = rf"{label} = {q_med:.2f}" rf"$^{{+{upper:.2f}}}_{{-{lower:.2f}}}$"
    ax.set_title(title, loc="left")

    # Round xlim to closest 0.05 s.t. xlim > 0.2
    if (
        np.abs(np.diff(ax.get_xlim())) < 0.2
        and np.abs(np.diff([true_value, q_med])) < 0.15
    ):
        ax.set_xlim(
            np.round((q_med - 0.1) * 20) / 20, np.round((q_med + 0.1) * 20) / 20
        )
    else:
        ax.set_xlim(
            np.round(ax.get_xlim()[0] * 20) / 20, np.round(ax.get_xlim()[1] * 20) / 20
        )
    ax.set_xlabel(label)
    ax.ticklabel_format(style="plain")
    ax.set_ylabel("Frequency")

    return (q_low, q_med, q_high)


def plot_histograms(sens_dict, bins=30):
    S1 = np.array(sens_dict["S1"])
    ST = np.array(sens_dict["ST"])
    S2 = np.array(sens_dict["S2"])

    s1_labels = [r"$S_1$", r"$S_2$", r"$S_3$"]
    st_labels = [r"$S_{T_1}$", r"$S_{T_2}$", r"$S_{T_3}$"]
    s2_labels = [r"$S_{12}$", r"$S_{13}$", r"$S_{23}$"]

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    # Analytical solution to Sobol indices
    analytical_solution_dict = analytical_solution(7, 0.1, return_dict=True)

    # First-order
    quantiles = {}
    for i in range(3):
        key = (
            s1_labels[i]
            .replace("{", "")
            .replace("}", "")
            .replace("_", "")
            .replace("$", "")
        )
        quantiles[key] = plot_hist_with_kde(
            axes[0, i], S1[:, i], s1_labels[i], bins, analytical_solution_dict[key]
        )

    # Total-order
    for i in range(3):
        key = (
            st_labels[i]
            .replace("{", "")
            .replace("}", "")
            .replace("_", "")
            .replace("$", "")
        )
        quantiles[key] = plot_hist_with_kde(
            axes[1, i], ST[:, i], st_labels[i], bins, analytical_solution_dict[key]
        )

    # Second-order
    for i in range(3):
        key = (
            s2_labels[i]
            .replace("{", "")
            .replace("}", "")
            .replace("_", "")
            .replace("$", "")
        )
        quantiles[key] = plot_hist_with_kde(
            axes[2, i], S2[:, i], s2_labels[i], bins, analytical_solution_dict[key]
        )

    plt.tight_layout()
    plt.show()

    return quantiles


def main(sens_dir, samples_dir):
    pu.set_nes_plot_style()
    sens_models = list_model_names(sens_dir)
    sens_models = [sens_model.replace(".joblib", ".json") for sens_model in sens_models]
    quantiles = []
    for sens_model in sens_models:
        # Read sensitivity indices
        model_path = (
            "output/sensitivity_analyses/sampled_gp_functions/"
            f"{sens_dir}/{samples_dir}/{sens_model}"
        )
        sens_dict = rw.json_read_dictionary(model_path)

        quantiles.append(plot_histograms(sens_dict))

    # Plot sensitivity indices as a function of number of samples
    setup = rw.json_read_dictionary(f"output/gp_models/{sens_dir}/setup.json")
    standard_solution = rw.json_read_dictionary(
        f"output/sensitivity_analyses/{sens_dir}.json"
    )
    plot_quantiles(quantiles, setup["n_samples"], standard_solution["sal_results"])


if __name__ == "__main__":
    main("noisy_ishigami", "n_32768")
    input("Press Enter to exit...")
