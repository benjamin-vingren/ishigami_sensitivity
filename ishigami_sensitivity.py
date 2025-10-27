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


def gp_prediction(l, sigma_f, sigma_n , x_train, y_train):
    """
    Apply Gaussian Process (GP) regression to fit a model to training data.

    Parameters
    ----------
    l : float
        Length scale parameter for the RBF kernel.
    sigma_f : float
        Signal variance parameter for the constant kernel.
    sigma_n : float
        Noise variance parameter for the GP model.
    x_train : array_like of shape (n_samples, n_features)
        Input training data.
    y_train : array_like of shape (n_samples,)
        Target training values.

    Returns
    -------
    gp : sklearn.gaussian_process.GaussianProcessRegressor
        Fitted Gaussian Process model.
    """ 

    # Kernel definition 
    # kernel = (ConstantKernel(constant_value=sigma_f) * 
    #           RBF(length_scale=l, length_scale_bounds=(0.5, 500)))
    
    kernel = (ConstantKernel(constant_value=sigma_f) * RBF(length_scale=l))
    
    # kernel = (ConstantKernel(constant_value=sigma_f, constant_value_bounds='fixed') *
    #           RBF(length_scale=l, length_scale_bounds='fixed'))
    
    # GP model
    gp = GaussianProcessRegressor(kernel=kernel, alpha=(2 * sigma_n)**2,
                                  n_restarts_optimizer=10)
        
    # Fitting in the gp model
    gp.fit(x_train, y_train)

    return gp


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
    f = df.ishigami(samples['x1'], samples['x2'], samples['x3'], a, b)
    n_samples = len(f)
    
    # Add nosie
    np.random.seed(seed)
    epsilon = np.random.uniform(-noise_level, noise_level, n_samples)
    f += epsilon
    f_std = noise_level * np.ones(n_samples)

    # Noisy samples
    if plot:
        plt.figure()
        plt.errorbar(np.arange(0, len(f)), f, yerr=f_std, linestyle='None',
                     marker='.')
        plt.xlabel('Sample #')
        plt.ylabel('$f(x_1, x_2, x_3)$')
    
    return f, f_std


def GP_sensitivity_analysis(samples, f, f_std, problem, master_seed,
                            n_upsamples=2**14):
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
    master_seed : int
        Seed used for Sobol sampling.
    """
    # Create GP surrogate model
    x_train = np.array([samples['x1'], samples['x2'], samples['x3']]).T
    y_train = np.copy(f)
    gp = gp_prediction(1, 3, f_std, x_train, y_train)

    # Create Sobol samples for GP and predict
    sens_samples = SobolSample(problem, n_upsamples, seed=master_seed)
    y_pred, y_std = gp.predict(sens_samples, return_std=True)
    
    # Perform sensitivity analysis on GP prediction
    results = analyze(problem, y_pred, parallel=True, n_processors=12,
                      num_resamples=100)

    return results, master_seed


def GP_resampling_analysis(samples, f, f_std, problem, master_seed,
                           n_upsamples=2**14, n_reruns=10):
    """
    Perform resampled Sobol sensitivity analysis using a Gaussian Process
    surrogate model.

    The GP surrogate is resampled multiple times to estimate confidence
    intervals for the Sobol sensitivity indices.

    Parameters
    ----------
    samples : dict of array_like
        Input samples used to train the GP model, with keys 'x1', 'x2', and 'x3'.
    f : ndarray of shape (n_samples,)
        Function evaluations corresponding to the samples.
    f_std : ndarray of shape (n_samples,)
        Standard deviation (uncertainty) for each function evaluation.
    problem : dict
        Problem definition dictionary compatible with SALib, containing the number
        of variables, names, and bounds.
    master_seed : int
        Random seed for reproducibility in Sobol sampling.
    n_upsamples : int, optional
        Number of Sobol samples used for each GP resampling. Default is 2**14.
    n_reruns : int, optional
        Number of resampled analyses to perform for uncertainty estimation.
        Default is 10.

    Returns
    -------
    results : dict
        Dictionary containing mean and confidence intervals of Sobol sensitivity indices.
    seeds : ndarray of shape (n_reruns,)
        Random seeds used for each resampling iteration.

    Notes
    -----
    This function can be computationally expensive due to multiple GP
    resampling and Sobol analyses.
    """
    # Create GP surrogate model
    x_train = np.array([samples['x1'], samples['x2'], samples['x3']]).T
    y_train = np.copy(f)
    gp = gp_prediction(1, 3, f_std, x_train, y_train)
    
    # Set up sensitivity analysis
    results = {}
    S1_results = np.zeros([10, 3])
    S1_conf_results = np.zeros([10, 3])
    S2_results = np.zeros([10, 3, 3])
    S2_conf_results = np.zeros([10, 3, 3])
    ST_results = np.zeros([10, 3])
    ST_conf_results = np.zeros([10, 3])

    # Resample from the GP to incorporate uncertainties
    rng = np.random.default_rng(master_seed)
    seeds = rng.integers(0, 1E10, n_reruns)
    for i, seed in enumerate(seeds):
        sens_samples = SobolSample(problem, n_upsamples, seed=seed)
        y_pred, y_std = gp.predict(sens_samples, return_std=True)
        result = analyze(problem, y_pred, num_resamples=1)
        S1_results[i] = result['S1']
        S1_conf_results[i] = result['S1_conf']
        S2_results[i] = result['S2']
        S2_conf_results[i] = result['S2_conf']
        ST_results[i] = result['ST']
        ST_conf_results[i] = result['ST_conf']

    # Save results
    results['S1'] = np.mean(S1_results, axis=0)
    results['S1_conf'] = 2 * np.std(S1_results, axis=0)
    results['S2'] = np.mean(S2_results, axis=0)
    results['S2_conf'] = 2 * np.std(ST_results, axis=0)
    results['ST'] = np.mean(ST_results, axis=0)
    results['ST_conf'] = 2 * np.std(ST_results, axis=0)
    
    return results, seeds


def main(seed):
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
    n_samples_list = [2**i for i in range(7, 14)]

    # Salib problem
    problem = {'num_vars': 3, 'names': ['x1', 'x2', 'x3'],
               'bounds': [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]]}
    
    # Draw unique Sobol samples for GP sensitivity analysis
    bounds = {f'x{i}': [-np.pi, np.pi] for i in range(1, 4)}

    gp_results = []
    sal_results = []
    for n_samples in n_samples_list:
        gp_samples = samplers.get_sobol_samples(bounds, n_samples, seed=seed)
        
        # Calculate noisy Ishigami
        f_gp, f_gp_std = draw_from_ishigami(gp_samples, seed=seed)
        
        gp_result, seeds = GP_sensitivity_analysis(gp_samples, f_gp, f_gp_std,
                                                   problem, seed)
        
        # Draw samples for traditional sensitivity analysis
        if n_samples % 8 == 0:
            sal_samples = SobolSample(problem, int(n_samples / 8))
        else:
            raise ValueError('n_samples is not divisible by 8.')
        
        sal_samples = {f'x{i + 1}': sal_sample for i, sal_sample in
                       enumerate(sal_samples.T)}
        
        f_sal, f_sal_std = draw_from_ishigami(sal_samples, seed=seed)
        sal_result = analyze(problem, f_sal)
        
        gp_results.append(rw.listify(gp_result))
        sal_results.append(rw.listify(sal_result))
    
    return gp_results, sal_results, n_samples_list, seeds


if __name__ == '__main__':
    # seed = 29475183  # results_1.json
    # seed = 19475183  # results_2.json
    # seed = 39475183  # results_3.json
    seed = np.random.randint(0, 1E8)
    
    gp_results, sal_results, n_samples, seeds = main(seed)
    
    to_save = {'gp_results': gp_results, 'sal_results': sal_results,
               'n_samples': n_samples, 'gp_seeds': seeds}

    rw.json_write_dictionary('results_test.json', to_save)
