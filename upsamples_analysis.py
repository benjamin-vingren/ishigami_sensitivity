# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 10:31:50 2025

@author: benjer
"""

import numpy as np
import utilities.samplers as samplers
import utilities.read_write as rw
from SALib.sample.sobol import sample as SobolSample
from SALib.analyze.sobol import analyze
from ishigami_sensitivity import draw_from_ishigami
from ishigami_sensitivity import gp_prediction


def upsamples_analysis(master_seed):
    """
    Analyze the effect of the number of Sobol upsamples on Gaussian Process 
    (GP)-based Sobol sensitivity index uncertainty.

    This function investigates how varying the number of Sobol samples used 
    for predictions from a trained GP surrogate model influences the estimated 
    uncertainty in Sobol sensitivity indices for the Ishigami function.

    Parameters
    ----------
    master_seed : int
        Random seed for reproducibility across sampling, GP modeling, and 
        Sobol sensitivity analysis.

    Returns
    -------
    gp_results : list of dict
        List of dictionaries containing Sobol sensitivity analysis results 
        for each upsample size.
    upsamples_list : list of int
        List of numbers of Sobol samples used for upsampling the GP
        surrogate model.
    seeds : ndarray of int, shape (len(upsamples_list),)
        Random seeds used for each upsample iteration.
    """
    # Salib problem
    problem = {'num_vars': 3, 'names': ['x1', 'x2', 'x3'],
               'bounds': [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]]}
    
    # Draw unique Sobol samples for GP sensitivity analysis
    bounds = {f'x{i}': [-np.pi, np.pi] for i in range(1, 4)}
    
    # Dependence of number of upsamples on the predicted uncertainty on Si
    upsamples_list = [2**i for i in range(7, 17)]
    
    # Use 2048 samples to create GP surrogate model
    samples = samplers.get_sobol_samples(bounds, 2**11)  
    
    # Calculate noisy Ishigami
    f_gp, f_gp_std = draw_from_ishigami(samples, seed=master_seed)
    
    # Create GP surrogate model
    x_train = np.array([samples['x1'], samples['x2'], samples['x3']]).T
    y_train = np.copy(f_gp)
    gp = gp_prediction(1, 3, f_gp_std, x_train, y_train)
    
    # Create Sobol samples for GP and predict    
    gp_results = []
    rng = np.random.default_rng(master_seed)
    seeds = rng.integers(0, 1E10, len(upsamples_list))
    for upsamples, seed in zip(upsamples_list, seeds):
        sens_samples = SobolSample(problem, upsamples, seed=seed)
        y_pred, y_std = gp.predict(sens_samples, return_std=True)
        
        # Perform sensitivity analysis on GP prediction
        results = analyze(problem, y_pred, parallel=True, n_processors=12,
                          num_resamples=100)
        
        gp_results.append(rw.listify(results))

    return gp_results, upsamples_list, seeds


if __name__ == '__main__':
    seed = np.random.randint(0, 1E8)

    up_results, up_list, up_seeds = upsamples_analysis(seed)
    
    to_save = {'up_results': up_results, 'up_list': up_list,
               'up_seeds': up_seeds.tolist()}
    
    rw.json_write_dictionary('upsample_test.json', to_save)
