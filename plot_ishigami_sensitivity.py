# %%

import matplotlib.pyplot as plt
import numpy as np
import utilities.read_write as rw
import utilities.plot_utils as pu


def get_sobol_indices(order, var_index, results_dict):
    """
    Extract Sobol sensitivity indices from a results dictionary.

    Parameters
    ----------
    order : str
        Sobol index order to extract (e.g., "S1", "ST", "S2").
    var_index : int
        Index of the variable of interest (0 for x1, 1 for x2, 2 for x3).
    results_dict : list of dict
        List of result dictionaries, typically taken from
        `results['gp_results']` or `results['sal_results']`.

    Returns
    -------
    list of float
        Sobol indices corresponding to the specified order and variable index
        across all result entries.
    """
    return [res[order][var_index] for res in results_dict]


def analytical_solution(a, b, print_to_consol=False):
    """
    Compute analytical Sobol indices for the Ishigami function.

    Parameters
    ----------
    a : float
        Ishigami function parameter 'a'.
    b : float
        Ishigami function parameter 'b'.
    print_to_consol : bool, optional
        If True, prints computed Sobol indices to the console. Default
        is False.

    Returns
    -------
    first_order : tuple of float
        First-order Sobol indices (S1, S2, S3).
    second_order : tuple of float
        Second-order Sobol indices (S12, S13, S23).
    total_order : tuple of float
        Total-effect Sobol indices (ST1, ST2, ST3).
    """
    # Analytical solutions of Ishigami function
    # https://openturns.github.io/openturns/latest/usecases/use_case_ishigami.html
    V = a**2 / 8 + b * np.pi**4 / 5 + b**2 * np.pi**8 / 18 + 0.5
    V1 = 0.5 * (1 + b * np.pi**4 / 5) ** 2
    V2 = a**2 / 8
    V3 = 0.0

    V12 = 0.0
    V13 = b**2 * np.pi**8 * 8 / 225
    V23 = 0.0

    # Analytic Sobol indicies
    # Main effect (first order indices)
    S1 = V1 / V
    S2 = V2 / V
    S3 = V3 / V

    # Interactions (second order indices)
    S12 = V12 / V  # (= 0)
    S13 = V13 / V
    S23 = V23 / V  # (= 0)

    # Total order indices
    ST1 = (V1 + V13 + V12) / V
    ST2 = (V2 + V12 + V23) / V  # (= S2)
    ST3 = (V3 + V13 + V23) / V  # (= S13)

    if print_to_consol:
        print("")
        print("Analytical Sobol indices:")
        print("-------------------------")
        print("Total")
        print("-----")
        print(f"ST1: {ST1}")
        print(f"ST2: {ST2}")
        print(f"ST3: {ST3}")

        print("First order")
        print("-----------")
        print(f"S1: {S1}")
        print(f"S2: {S2}")
        print(f"S3: {S3}")

        print("Second order")
        print("------------")
        print(f"S12: {S12}")
        print(f"S13: {S13}")
        print(f"S23: {S23}")

    return (S1, S2, S3), (S12, S13, S23), (ST1, ST2, ST3)


def plot_total_indices(results):
    """
    Plot total-effect Sobol indices for the Ishigami function.

    Compares the total-order Sobol indices obtained from:
    - Standard SALib sensitivity analysis.
    - Gaussian Process (GP)-based surrogate model predictions.
    - Analytical reference solution.

    Parameters
    ----------
    results : dict
        Dictionary containing GP and SALib sensitivity results,
        including `gp_results`, `sal_results`, and `n_samples`.
    """
    for i in range(3):
        ST_gp = get_sobol_indices("ST", i, results["gp_results"])
        ST_gp_std = get_sobol_indices("ST_conf", i, results["gp_results"])

        ST_sal = get_sobol_indices("ST", i, results["sal_results"])
        ST_sal_std = get_sobol_indices("ST_conf", i, results["sal_results"])

        plt.figure(f"Total indices x{i+1}")
        plt.title(f"Total effect index $x_{i+1}$", loc="left")

        # Salib standard solution
        plt.errorbar(
            results["n_samples"],
            ST_sal,
            yerr=ST_sal_std,
            linestyle="None",
            marker=".",
            color="C0",
            label="Standard solution",
        )

        # GP predicted solution
        plt.errorbar(
            results["n_samples"],
            ST_gp,
            yerr=ST_gp_std,
            linestyle="None",
            marker=".",
            color="C1",
            label="GP predicted solution",
        )

        # Plot analytical solution vs. GP prediction and Salib standard sln
        first_order, second_order, total = analytical_solution(7, 0.1)

        plt.axhline(total[i], color="k", linestyle="--", label="Analytical solution")
        plt.xlabel("Number of samples")
        plt.ylabel(f"$S_{{T_{i+1}}}$")
        plt.legend(
            facecolor="lightgray", edgecolor="black", framealpha=0.3, frameon=True
        )
        plt.xscale("log")
        plt.xlim(1e2, 1.1e4)


def plot_first_indices(results):
    """
    Plot first-order Sobol indices for the Ishigami function.

    Compares the first-order Sobol indices obtained from:
    - Standard SALib sensitivity analysis.
    - Gaussian Process (GP)-based surrogate model predictions.
    - Analytical reference solution.

    Parameters
    ----------
    results : dict
        Dictionary containing GP and SALib sensitivity results,
        including `gp_results`, `sal_results`, and `n_samples`.
    """
    for i in range(3):
        S1_gp = get_sobol_indices("S1", i, results["gp_results"])
        S1_gp_std = get_sobol_indices("S1_conf", i, results["gp_results"])

        S1_sal = get_sobol_indices("S1", i, results["sal_results"])
        S1_sal_std = get_sobol_indices("S1_conf", i, results["sal_results"])

        plt.figure(f"First order effect x{i+1}")
        plt.title(f"First order effect index $x_{i+1}$", loc="left")

        # Salib standard solution
        plt.errorbar(
            results["n_samples"],
            S1_sal,
            yerr=S1_sal_std,
            linestyle="None",
            marker=".",
            color="C0",
            label="Standard solution",
        )

        # GP predicted solution
        plt.errorbar(
            results["n_samples"],
            S1_gp,
            yerr=S1_gp_std,
            linestyle="None",
            marker=".",
            color="C1",
            label="GP predicted solution",
        )

        # Plot analytical solution vs. GP prediction and Salib standard sln
        first_order, second_order, total = analytical_solution(7, 0.1)

        plt.axhline(
            first_order[i], color="k", linestyle="--", label="Analytical solution"
        )
        plt.xlabel("Number of samples")
        plt.ylabel(f"$S_{i+1}$")
        plt.legend(
            facecolor="lightgray", edgecolor="black", framealpha=0.3, frameon=True
        )
        plt.xscale("log")
        plt.xlim(1e2, 1.1e4)


def plot_second_indices(results):
    """
    Plot second-order (interaction) Sobol indices for the Ishigami function.

    Compares the second-order Sobol indices obtained from:
    - Standard SALib sensitivity analysis.
    - Gaussian Process (GP)-based surrogate model predictions.
    - Analytical reference solution.

    Parameters
    ----------
    results : dict
        Dictionary containing GP and SALib sensitivity results,
        including `gp_results`, `sal_results`, and `n_samples`.
    """
    # Analytical solution
    first_order, second_order, total = analytical_solution(7, 0.1)
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

        plt.figure(f"Second order effect x{interaction[0]}{interaction[1]}")
        plt.title(
            (
                f"Second order effect index "
                f"$x_{{{interaction[0] + 1}{interaction[1] + 1}}}$"
            ),
            loc="left",
        )

        # Salib standard solution
        plt.errorbar(
            results["n_samples"],
            S2_sal,
            yerr=S2_sal_std,
            linestyle="None",
            marker=".",
            color="C0",
            label="Standard solution",
        )

        # GP predicted solution
        plt.errorbar(
            results["n_samples"],
            S2_gp,
            yerr=S2_gp_std,
            linestyle="None",
            marker=".",
            color="C1",
            label="GP predicted solution",
        )

        plt.axhline(
            second_order[i], color="k", linestyle="--", label="Analytical solution"
        )

        plt.xlabel("Number of samples")
        plt.ylabel(f"$S_{{{interaction[0] + 1}{interaction[1] + 1}}}$")
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
    # main('output/multirun_ishigami.json')
    main("output/sensitivity_analyses/noiseless_ishigami.json")
