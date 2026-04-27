# %%
import numpy as np
import matplotlib.pyplot as plt
from ishigami_sensitivity import list_model_names
import utilities.samplers as samplers
import utilities.read_write as rw
import utilities.distribution_functions as dfs
import utilities.plot_utils as pu
import joblib


if __name__ == "__main__":
    pu.set_nes_plot_style(large_font=True)
    # Get GP models
    gp_dir = "noiseless_ishigami"
    gp_model_names = list_model_names(gp_dir)

    # Generate samples
    n_samples = 10000
    setup = rw.json_read_dictionary(f"output/gp_models/{gp_dir}/setup.json")
    bounds = {key: value for key, value in zip(["x1", "x2", "x3"], setup["bounds"])}
    samples = samplers.get_latin_hypercube_samples(bounds, n_samples, return_dict=False)

    # Calculate residuals
    residuals_iqr = np.zeros([len(gp_model_names), 3])
    for i, gp_model_name in enumerate(gp_model_names):
        # Load GP model and predict
        gp_model_dict = joblib.load(f"output/gp_models/{gp_dir}/{gp_model_name}")
        gp = gp_model_dict["gp"]
        y_pred, y_std = gp.predict(samples, return_std=True)
        y_true = dfs.ishigami(*samples.T)

        # Average residuals
        # TODO: Implement Kernel density estimate using GP standard deviation as the
        # kernel along with Silvermans rule of thumb.
        residuals = np.abs(y_pred - y_true)
        residuals_iqr[i] = np.quantile(residuals, [0.25, 0.5, 0.75])
        plt.figure()
        plt.hist(np.abs(y_pred - y_true), bins=np.arange(0, 10, 0.01))
        ax = plt.gca()
        ax.axvline(residuals_iqr[i][0], linestyle="--", color="r")
        ax.axvline(residuals_iqr[i][1], linestyle="--", color="r")
        ax.axvline(residuals_iqr[i][2], linestyle="--", color="r")
        plt.xlabel("Residuals")
        plt.title(gp_model_name, loc="left")

    # %%
    plt.figure("Residuals")

    # Uncertainties are IQR
    plt.errorbar(
        setup["n_samples"],
        residuals_iqr[:, 1],
        yerr=np.diff(residuals_iqr).T,
        linestyle="None",
        color="k",
        marker=".",
    )
    plt.xscale("log")
    plt.xlabel("Number of samples")
    plt.ylabel("Residuals (IQR)")
    plt.ylim(top=1)
    plt.xlim(1e2, 1e4)
    plt.gca().axhline(0, color="k", linestyle="--")

    input("Press Enter to exit...")
