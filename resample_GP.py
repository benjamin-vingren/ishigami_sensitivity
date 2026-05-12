import utilities.read_write as rw
from SALib.sample.sobol import sample as SobolSample
from SALib.analyze.sobol import analyze
import joblib
from ishigami_sensitivity import list_model_names
import matplotlib.pyplot as plt
import numpy as np
import time
from itertools import batched


def resampled_sensitivity_analysis(
    gp_dir, gp_model_name, n_gp_samples, n_sens_samples, batch_size=2**10
):
    """
    Perform a resampled Sobol sensitivity analysis by sampling from a Gaussian
    Process (GP) model.

    This function loads a pre-trained GP model, generates multiple realizations
    of the GP function using Sobol sequences, and calculates the first-order
    (S1), second-order (S2), and total-effect (ST) Sobol indices for each
    realization. The results are then saved as a JSON file.

    Parameters
    ----------
    gp_dir : str
        The directory path where the GP model and setup files are located
        within the 'output/gp_models/' folder.
    gp_model_name : str
        The filename of the loaded GP model (e.g., 'model.joblib').
    n_gp_samples : int
        The number of GP function realizations to sample for the analysis.
    n_sens_samples : int
        The number of Sobol samples to use for the sensitivity analysis.
        Must be a multiple of 8.
    batch_size : int, optional
        The number of samples to process at once when predicting GP
        function values to manage memory usage. Default is 1024 (2**10).

    Raises
    ------
    ValueError
        If `n_sens_samples` is not divisible by 8.

    Returns
    -------
    None
        The function saves the resulting Sobol indices to a JSON file in
        'output/sensitivity_analyses/sampled_gp_functions/'.
    """
    setup = rw.json_read_dictionary(f"output/gp_models/{gp_dir}/setup.json")

    # Load GP model
    gp_dict = joblib.load(f"output/gp_models/{gp_dir}/{gp_model_name}")
    gp = gp_dict["gp"]

    # Sample GP functions
    if n_sens_samples % 8 == 0:
        sal_samples = SobolSample(setup, int(n_sens_samples / 8))
    else:
        raise ValueError("n_sens_samples is not divisible by 8.")

    # If too many samples, do the GP prediction in chunks
    print("Predicting GP functions.")
    if n_sens_samples > batch_size:
        chunks = []
        for i, chunk in enumerate(batched(sal_samples, batch_size)):
            print(f"Batch {i + 1}/{int(np.ceil(n_sens_samples/batch_size))}")
            chunk = np.array(chunk)
            y = gp.sample_y(chunk, n_samples=n_gp_samples)
            chunks.append(y)

        # Concatenate along sample axis (axis=0)
        gp_functions = np.concatenate(chunks, axis=0)

    else:
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


def main(gp_dir, n_gp_samples, n_sens_samples, batch_size):
    # Get list of GP model names
    model_names = list_model_names(gp_dir)

    # Perform sensitivity analysis by sampling GP functions
    for model_name in model_names:
        t0 = time.time()
        resampled_sensitivity_analysis(
            gp_dir=gp_dir,
            gp_model_name=model_name,
            n_gp_samples=n_gp_samples,
            n_sens_samples=n_sens_samples,
            batch_size=batch_size,
        )
        print(f"Time for {model_name}: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    # Run main
    main(
        gp_dir="noiseless_ishigami",
        n_gp_samples=2000,
        n_sens_samples=2**15,
        batch_size=2**11,
    )
