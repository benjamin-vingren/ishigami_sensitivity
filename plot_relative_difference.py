# %%
from plot_ishigami_sensitivity import analytical_solution, get_sobol_indices
import matplotlib.pyplot as plt
import utilities.read_write as rw
import utilities.plot_utils as pu
import numpy as np


def plot_total_indices(results):
    for i in range(3):
        ST_gp = get_sobol_indices("ST", i, results["gp_results"])
        ST_gp_std = get_sobol_indices("ST_conf", i, results["gp_results"])

        ST_sal = get_sobol_indices("ST", i, results["sal_results"])
        ST_sal_std = get_sobol_indices("ST_conf", i, results["sal_results"])

        # Plot analytical solution vs. GP prediction and Salib standard sln
        _, _, total = analytical_solution(7, 0.1)

        # Calculate differences
        gp_diff = np.abs(np.array(ST_gp) - total[i])
        sal_diff = np.abs(np.array(ST_sal) - total[i])

        # Plot differences
        plt.figure(f"Abs. diff. total indices x{i+1}")
        plt.title(
            f"Absolute difference w.r.t. analytical solution for $S_{{T_{i+1}}}$",
            loc="left",
        )

        plt.axhline(0, color="k", linestyle="--")

        # Salib standard solution
        plt.plot(
            results["n_samples"],
            sal_diff,
            color="C0",
            linestyle="-.",
            marker=".",
            label="Standard solution",
        )

        # GP predicted solution
        plt.plot(
            results["n_samples"],
            gp_diff,
            color="C1",
            linestyle="-",
            marker="s",
            label="GP predicted solution",
            markersize=3,
        )

        plt.xlabel("Number of samples")
        plt.ylabel(f"$| \Delta S_{{T_{i+1}}}$ |")
        plt.legend(
            facecolor="lightgray", edgecolor="black", framealpha=0.3, frameon=True
        )
        plt.xscale("log")
        plt.xlim(1e2, 1.1e4)


def plot_first_indices(results):
    for i in range(3):
        S1_gp = get_sobol_indices("S1", i, results["gp_results"])
        S1_gp_std = get_sobol_indices("S1_conf", i, results["gp_results"])

        S1_sal = get_sobol_indices("S1", i, results["sal_results"])
        S1_sal_std = get_sobol_indices("S1_conf", i, results["sal_results"])

        # Plot analytical solution vs. GP prediction and Salib standard sln
        first_order, _, _ = analytical_solution(7, 0.1)

        # Calculate differences
        gp_diff = np.abs(np.array(S1_gp) - first_order[i])
        sal_diff = np.abs(np.array(S1_sal) - first_order[i])

        # Plot differences
        plt.figure(f"Abs. diff. first indices x{i+1}")
        plt.title(
            f"Absolute difference w.r.t. analytical solution for $S_{i+1}$", loc="left"
        )

        plt.axhline(0, color="k", linestyle="--")

        # Salib standard solution
        plt.plot(
            results["n_samples"],
            sal_diff,
            color="C0",
            linestyle="-.",
            marker=".",
            label="Standard solution",
        )

        # GP predicted solution
        plt.plot(
            results["n_samples"],
            gp_diff,
            color="C1",
            linestyle="-",
            marker="s",
            label="GP predicted solution",
            markersize=3,
        )

        plt.xlabel("Number of samples")
        plt.ylabel(f"Absolute difference $| \Delta S_{i+1} |$")
        plt.legend(
            facecolor="lightgray", edgecolor="black", framealpha=0.3, frameon=True
        )
        plt.xscale("log")
        plt.xlim(1e2, 1.1e4)


def plot_second_indices(results):
    # Analytical solution
    _, second_order, _ = analytical_solution(7, 0.1)
    interactions = [[0, 1], [0, 2], [1, 2]]
    for i, interaction in enumerate(interactions):
        # GP predicted solution
        S2_gp = [
            np.array(res["S2"])[interaction[0], interaction[1]]
            for res in results["gp_results"]
        ]
        S2_gp_std = [
            np.array(res["S2_conf"])[interaction[0], interaction[1]]
            for res in results["gp_results"]
        ]

        # Salib standard solution
        S2_sal = [
            np.array(res["S2"])[interaction[0], interaction[1]]
            for res in results["sal_results"]
        ]
        S2_sal_std = [
            np.array(res["S2_conf"])[interaction[0], interaction[1]]
            for res in results["sal_results"]
        ]

        # Calculate differences
        gp_diff = np.abs(np.array(S2_gp) - second_order[i])
        sal_diff = np.abs(np.array(S2_sal) - second_order[i])

        # Plot differences
        plt.figure(f"Abs. diff. econd order effect x{interaction[0]}{interaction[1]}")
        plt.title(
            (
                f"Absolute difference w.r.t. analytical solution for "
                f"$S_{{{interaction[0] + 1}{interaction[1] + 1}}}$"
            ),
            loc="left",
        )

        plt.axhline(0, color="k", linestyle="--")

        # Salib standard solution
        plt.plot(
            results["n_samples"],
            sal_diff,
            color="C0",
            linestyle="-.",
            marker=".",
            label="Standard solution",
        )

        # GP predicted solution
        plt.plot(
            results["n_samples"],
            gp_diff,
            color="C1",
            linestyle="-",
            marker="s",
            label="GP predicted solution",
            markersize=3,
        )

        plt.xlabel("Number of samples")
        plt.ylabel(f"$| \Delta S_{{{interaction[0] + 1}{interaction[1] + 1}}} |$")
        plt.legend(
            facecolor="lightgray", edgecolor="black", framealpha=0.3, frameon=True
        )

        plt.xscale("log")
        plt.xlim(1e2, 1.1e4)


def main(file_name):
    """
    Load sensitivity analysis results and generate comparison plots.

    Parameters
    ----------
    file_name : str
        Path to the JSON file containing the GP and SALib sensitivity results.

    Returns
    -------
    None
        Generates and displays plots comparing analytical, GP-based, and
        standard SALib Sobol indices.
    """
    results = rw.json_read_dictionary(file_name)
    plot_total_indices(results)
    plot_first_indices(results)
    plot_second_indices(results)


if __name__ == "__main__":
    pu.set_nes_plot_style()
    # main("output/multirun_ishigami.json")
    main("output/sensitivity_analyses/noiseless_ishigami.json")
    plt.show()
    input("Press Enter to exit...")
