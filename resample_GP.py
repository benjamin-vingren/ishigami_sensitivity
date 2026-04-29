import utilities.read_write as rw
from SALib.sample.sobol import sample as SobolSample
from SALib.analyze.sobol import analyze
import joblib
from ishigami_sensitivity import list_model_names
import matplotlib.pyplot as plt
import numpy as np


# def GP_resampling_analysis(
#     samples, f, f_std, problem, master_seed, n_upsamples=2**14, n_reruns=10
# ):
#     """
#     Perform resampled Sobol sensitivity analysis using a Gaussian Process
#     surrogate model.

#     The Ishigami function is sampled once and the GP surrogate is created based
#     on these samples. The GP surrogate model is then resampled multiple times
#     to estimate confidence intervals for the Sobol sensitivity indices.

#     Parameters
#     ----------
#     samples : dict of array_like
#         Input samples used to train the GP model, with keys 'x1', 'x2', and 'x3'.
#     f : ndarray of shape (n_samples,)
#         Function evaluations corresponding to the samples.
#     f_std : ndarray of shape (n_samples,)
#         Standard deviation (uncertainty) for each function evaluation.
#     problem : dict
#         Problem definition dictionary compatible with SALib, containing the number
#         of variables, names, and bounds.
#     master_seed : int
#         Random seed for reproducibility in Sobol sampling.
#     n_upsamples : int, optional
#         Number of Sobol samples used for each GP resampling. Default is 2**14.
#     n_reruns : int, optional
#         Number of resampled analyses to perform for uncertainty estimation.
#         Default is 10.

#     Returns
#     -------
#     results : dict
#         Dictionary containing mean and confidence intervals of Sobol sensitivity indices.
#     seeds : ndarray of shape (n_reruns,)
#         Random seeds used for each resampling iteration.

#     Notes
#     -----
#     This function can be computationally expensive due to multiple GP
#     resampling and Sobol analyses.
#     """
#     # Create GP surrogate model
#     x_train = np.array([samples["x1"], samples["x2"], samples["x3"]]).T
#     y_train = np.copy(f)
#     gp = gp_prediction(1, 3, f_std, x_train, y_train)

#     # Resample from the GP to incorporate uncertainties
#     rng = np.random.default_rng(master_seed)
#     seeds = rng.integers(0, 1e10, n_reruns)
#     results_dict = {
#         "S1": np.zeros([n_reruns, 3]),
#         "S2": np.zeros([n_reruns, 3, 3]),
#         "ST": np.zeros([n_reruns, 3]),
#     }

#     for i, seed in enumerate(seeds):
#         sens_samples = SobolSample(problem, n_upsamples, seed=seed)
#         y_pred, y_std = gp.predict(sens_samples, return_std=True)
#         result = analyze(problem, y_pred, num_resamples=1)

#         results_dict = update_results_dict(result, results_dict, i)

#     # Save results
#     results = save_results(results_dict)

#     return results, seeds


# def update_results_dict(result, results_dict, i):
#     """
#     Update the results dictionary with sensitivity indices from one iteration.

#     Parameters
#     ----------
#     result : dict
#         Dictionary containing Sobol sensitivity results from one analysis.
#     results_dict : dict
#         Dictionary accumulating all sensitivity results across iterations.
#     i : int
#         Index of the current iteration to update in the results dictionary.

#     Returns
#     -------
#     results_dict : dict
#         Updated dictionary with new sensitivity results added.
#     """
#     for key in results_dict.keys():
#         results_dict[key][i] = result[key]

#     return results_dict


# def save_results(results_dict):
#     """
#     Compute mean and confidence intervals for Sobol sensitivity indices.

#     Parameters
#     ----------
#     results_dict : dict
#         Dictionary containing Sobol sensitivity indices from multiple reruns.
#         Must include keys 'S1', 'S2', and 'ST'.

#     Returns
#     -------
#     results : dict
#         Dictionary with averaged sensitivity indices and their 95% confidence
#         intervals. Contains keys 'S1', 'S1_conf', 'S2', 'S2_conf', 'ST', and
#         'ST_conf'.
#     """
#     results = {}
#     results["S1"] = np.mean(results_dict["S1"], axis=0)
#     results["S1_conf"] = 2 * np.std(results_dict["S1"], axis=0)
#     results["S2"] = np.mean(results_dict["S2"], axis=0)
#     results["S2_conf"] = 2 * np.std(results_dict["S2"], axis=0)
#     results["ST"] = np.mean(results_dict["ST"], axis=0)
#     results["ST_conf"] = 2 * np.std(results_dict["ST"], axis=0)

#     return results


def resampled_sensitivity_analysis(gp_dir, gp_model_name, n_gp_samples, n_sens_samples):
    """ """
    setup = rw.json_read_dictionary(f"output/gp_models/{gp_dir}/setup.json")

    # Load GP model
    gp_dict = joblib.load(f"output/gp_models/{gp_dir}/{gp_model_name}")
    gp = gp_dict["gp"]

    # Sample GP functions
    if n_sens_samples % 8 == 0:
        sal_samples = SobolSample(setup, int(n_sens_samples / 8))
    else:
        raise ValueError("n_sens_samples is not divisible by 8.")

    print("Predicting GP functions.")
    gp_functions = gp.sample_y(sal_samples, n_samples=n_gp_samples)
    print("Performing sensitivity analysis.")
    sal_results = []
    for i, gp_function in enumerate(gp_functions.T):

        sal_results.append(analyze(setup, gp_function, num_resamples=1))
        print(f"{gp_model_name}: {i + 1}/{len(gp_functions.T)} done.")

    # Save Sobol indices for each GP function sample
    S1 = np.array([sal_result["S1"] for sal_result in sal_results])
    S2 = np.array(
        [
            [sal_result["S2"][0, 1], sal_result["S2"][0, 2], sal_result["S2"][1, 2]]
            for sal_result in sal_results
        ]
    )
    ST = np.array([sal_results["ST"] for sal_results in sal_results])

    # Write to file
    file_name = gp_model_name.split(".")[0]
    to_save = {"S1": S1, "S2": S2, "ST": ST}
    rw.json_write_dictionary(
        f"output/sensitivity_analyses/sampled_gp_functions/{gp_dir}/{file_name}.json",
        rw.listify(to_save),
    )


def main(gp_dir, n_gp_samples, n_sens_samples):
    # Get list of GP model names
    model_names = list_model_names(gp_dir)

    # Perform sensitivity analysis by sampling GP functions
    for model_name in model_names:
        resampled_sensitivity_analysis(
            gp_dir=gp_dir,
            gp_model_name=model_name,
            n_gp_samples=n_gp_samples,
            n_sens_samples=n_sens_samples,
        )


if __name__ == "__main__":
    # Run main
    # main(gp_dir="noisy_ishigami", n_gp_samples=2000, n_sens_samples=4096)
    main(gp_dir="noiseless_ishigami", n_gp_samples=500, n_sens_samples=2**14)
