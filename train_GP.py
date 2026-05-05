from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from utilities import read_write as rw
from utilities import samplers
import numpy as np
from ishigami_sensitivity import draw_from_ishigami
import joblib
import os
import argparse


def train_GP(
    l,
    sigma_f,
    sigma_n,
    X_train,
    y_train,
    constant_value_bounds=[1e-5, 1e5],
    length_scale_bounds=[1e-5, 1e5],
    white_kernel=False,
    normalize=True,
):
    """
    Fit a Gaussian Process (GP) regression model to training data.

    Depending on the value of ``white_kernel``, the noise level is either
    estimated from the data using a WhiteKernel or provided explicitly via
    the observation uncertainties.

    Parameters
    ----------
    l : float
        Initial length scale of the RBF kernel.
    sigma_f : float
        Initial signal variance (amplitude) for the ConstantKernel.
    sigma_n : array-like of shape (n_samples,) or float
        Observation noise standard deviation. Used as ``alpha`` if
        ``white_kernel`` is False, otherwise used to initialize the
        WhiteKernel noise level.
    X_train : array-like of shape (n_samples, n_features)
        Training input samples.
    y_train : array-like of shape (n_samples,) or (n_samples, n_targets)
        Training target values.
    constant_value_bounds : list of float, optional
        Bounds on the constant kernel value during optimization,
        by default [1e-5, 1e5].
    length_scale_bounds : list of float, optional
        Bounds on the RBF kernel length scale during optimization,
        by default [1e-5, 1e5].
    white_kernel : bool, optional
        If True, include a WhiteKernel to estimate noise from the data.
        If False, use ``sigma_n`` as fixed observation noise, by default False.

    Returns
    -------
    gp : sklearn.gaussian_process.GaussianProcessRegressor
        Trained Gaussian Process model with optimized kernel parameters.

    Notes
    -----
    The model uses multiple restarts of the optimizer (n_restarts_optimizer=10)
    to improve hyperparameter estimation.
    """

    if white_kernel:
        print(
            (
                "Estimating GP standard deviation from "
                "variance in data using a white kernel."
            )
        )
        kernel = ConstantKernel(
            constant_value=sigma_f, constant_value_bounds=constant_value_bounds
        ) * RBF(length_scale=l, length_scale_bounds=length_scale_bounds) + WhiteKernel(
            noise_level=np.mean(sigma_n), noise_level_bounds=(1e-5, 1e2)
        )
        gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, normalize_y=normalize
        )
    else:
        print("Estimating GP standard deviation from provided uncertainties in data.")
        kernel = ConstantKernel(
            constant_value=sigma_f, constant_value_bounds=constant_value_bounds
        ) * RBF(length_scale=l, length_scale_bounds=length_scale_bounds)

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=sigma_n**2,
            n_restarts_optimizer=10,
            normalize_y=normalize,
        )

    gp.fit(X_train, y_train)

    print(gp.kernel_)

    return gp


def main():
    n_samples_list = [2**i for i in range(7, 14)]
    seed = 68521736
    noise_level = 0.5
    white_kernel = False
    gp_name = "noisy_ishigami"
    gp_dir = f"output/gp_models/{gp_name}"

    # Salib problem
    setup = {
        "num_vars": 3,
        "names": ["x1", "x2", "x3"],
        "bounds": [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]],
        "n_samples": n_samples_list,
        "seed": seed,
        "noise_level": noise_level,
    }
    rw.json_write_dictionary(f"{gp_dir}/setup.json", setup)
    gp_output = {"white_kernel": white_kernel}

    # Draw unique Sobol samples for training GPs
    bounds = {f"x{i}": [-np.pi, np.pi] for i in range(1, 4)}

    for i, n_samples in enumerate(n_samples_list):
        samples = samplers.get_sobol_samples(bounds, n_samples, seed=seed)
        # Calculate noisy Ishigami
        f_gp, f_gp_std = draw_from_ishigami(samples, noise_level=noise_level, seed=seed)

        gp_samples = np.array([samples["x1"], samples["x2"], samples["x3"]]).T
        gp = train_GP(1, 1, f_gp_std, gp_samples, f_gp, white_kernel=white_kernel)

        gp_output["samples"] = gp_samples
        gp_output["gp"] = gp

        fn = f"{gp_dir}/{gp_name}_{i}.joblib"

        if os.path.exists(fn):
            inp = input(f"Overwrite {fn}? (y/n): ")
            if "y" not in inp.lower():
                continue

        joblib.dump(gp_output, fn)
        print(f"Wrote {fn}")


if __name__ == "__main__":
    main()
