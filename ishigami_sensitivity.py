# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 12:49:35 2025

@author: benjer
"""

import utilities.distribution_functions as df
import matplotlib.pyplot as plt
import numpy as np
import utilities.samplers as samplers
import utilities.read_write as rw
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from SALib.sample.sobol import sample as SobolSample
from SALib.analyze.sobol import analyze
import os
import joblib


def draw_from_ishigami(samples, noise_level=0.5, seed=None, plot=False):
    """
    Generate noisy evaluations of the Ishigami function.

    Parameters
    ----------
    samples : dict of array_like
        Dictionary containing arrays for 'x1', 'x2', and 'x3' input variables.
    noise_level : float, optional
        Amplitude of uniform noise added to the function output. Default
        is 0.5.
    seed : int or None, optional
        Random seed for reproducibility. Default is None.
    plot : bool, optional
        Whether to display a plot of the noisy samples. Default is False.

    Returns
    -------
    f : ndarray of shape (n_samples,)
        Noisy function values.
    f_std : ndarray of shape (n_samples,)
        Standard deviation associated with the noise for each sample.
    """
    # Calculate Ishigami function
    a = 7
    b = 0.1
    f = df.ishigami(samples["x1"], samples["x2"], samples["x3"], a, b)
    n_samples = len(f)

    # Add nosie
    np.random.seed(seed)
    epsilon = np.random.uniform(-noise_level, noise_level, n_samples)
    f += epsilon
    f_std = noise_level * np.ones(n_samples)

    # Noisy samples
    if plot:
        plt.figure()
        plt.errorbar(np.arange(0, len(f)), f, yerr=f_std, linestyle="None", marker=".")
        plt.xlabel("Sample #")
        plt.ylabel("$f(x_1, x_2, x_3)$")

    return f, f_std


def GP_sensitivity_analysis(gp, problem, master_seed, n_upsamples=2**14):
    """
    Perform Sobol sensitivity analysis using a Gaussian Process
    surrogate model.

    Parameters
    ----------
    samples : dict of array_like
        Input samples used to train the GP model, with keys 'x1', 'x2',
        and 'x3'.
    f : ndarray of shape (n_samples,)
        Function evaluations corresponding to the samples.
    f_std : ndarray of shape (n_samples,)
        Standard deviation (uncertainty) for each function evaluation.
    problem : dict
        Problem definition dictionary compatible with SALib, containing the
        number of variables, names, and bounds.
    master_seed : int
        Random seed for reproducibility in Sobol sampling.
    n_upsamples : int, optional
        Number of upsampled Sobol samples for the surrogate model. Default
        is 2**14.

    Returns
    -------
    results : dict
        Dictionary containing Sobol sensitivity indices computed from the GP model.
    """

    # Create Sobol samples for GP and predict
    sens_samples = SobolSample(problem, n_upsamples, seed=master_seed)

    print("Predicting GP samples.")
    y_pred, y_std = gp.predict(sens_samples, return_std=True)

    # Perform sensitivity analysis on GP prediction
    print("Performing GP Sobol sensitivity analysis.")
    results = analyze(
        problem, y_pred, parallel=True, n_processors=12, num_resamples=100
    )

    return results


def list_model_names(gp_directory_path):
    gp_model_names = os.listdir(f"output/gp_models/{gp_directory_path}")
    gp_model_names.remove("setup.json")
    sort_keys = [int(gp.split("_")[-1].split(".")[0]) for gp in gp_model_names]
    sort_args = np.argsort(sort_keys)
    gp_model_names = np.array(gp_model_names)[sort_args]

    return gp_model_names


def main():
    """
    Run Gaussian Process and traditional Sobol sensitivity analyses for
    multiple sample sizes using the Ishigami function.

    Parameters
    ----------
    seed : int
        Random seed used for reproducibility across sampling and GP modeling.

    Returns
    -------
    gp_results : list of dict
        Results from the GP-based sensitivity analyses.
    sal_results : list of dict
        Results from the direct (non-GP) Sobol sensitivity analyses.
    n_samples_list : list of int
        List of sample sizes used in the analyses.
    seeds : ndarray
        Random seeds used in the GP sensitivity analyses.
    """
    gp_dir = "noiseless_ishigami"
    setup = rw.json_read_dictionary(f"output/gp_models/{gp_dir}/setup.json")
    n_samples_list = setup["n_samples"]

    # List of GP models sorted in ascending order
    gp_model_names = list_model_names(f"output/gp_models/{gp_dir}")

    gp_results = []
    sal_results = []
    for gp_model_name in gp_model_names:
        gp_model_dict = joblib.load(f"output/gp_models/{gp_dir}/{gp_model_name}")
        gp_samples = gp_model_dict["samples"]
        gp_model = gp_model_dict["gp"]
        seed = setup["seed"]
        gp_result = GP_sensitivity_analysis(gp_model, setup, seed)

        # Draw samples for traditional sensitivity analysis
        n_samples = len(gp_samples)
        if n_samples % 8 == 0:
            sal_samples = SobolSample(setup, int(n_samples / 8))
        else:
            raise ValueError("n_samples is not divisible by 8.")

        sal_samples = {
            f"x{i + 1}": sal_sample for i, sal_sample in enumerate(sal_samples.T)
        }

        f_sal, f_sal_std = draw_from_ishigami(sal_samples, noise_level=0.01, seed=seed)

        print("Performing traditional Sobol sensitivity analysis.")
        sal_result = analyze(setup, f_sal)

        gp_results.append(rw.listify(gp_result))
        sal_results.append(rw.listify(sal_result))
        print(f"{gp_model_name} ({n_samples} samples) done.")

    return gp_results, sal_results, n_samples_list


if __name__ == "__main__":
    gp_results, sal_results, n_samples = main()

    to_save = {
        "gp_results": gp_results,
        "sal_results": sal_results,
        "n_samples": n_samples,
    }

    rw.json_write_dictionary("noiseless_ishigami.json", to_save)
