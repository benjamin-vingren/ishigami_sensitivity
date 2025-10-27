# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 10:46:53 2025

@author: benjer
"""

import matplotlib.pyplot as plt
import numpy as np
import utilities.read_write as rw
import utilities.plot_utils as pu
from plot_ishigami_sensitivity import get_sobol_indices, analytical_solution


def plot_total_indices(results):
    """
    Plot total-effect Sobol indices for the Ishigami function.

    Compares the total-order Sobol indices obtained from:
    - Gaussian Process (GP)-based surrogate model predictions.
    - Analytical reference solution.

    Parameters
    ----------
    results : dict
        Dictionary containing the GP upsample sensitivity results, including
        `up_results` and `up_list`.
    """
    for i in range(3):
        ST_gp = get_sobol_indices('ST', i, results['up_results'])
        ST_gp_std = get_sobol_indices('ST_conf', i, results['up_results'])
        
        plt.figure(f'Total indices x{i + 1}')
        plt.title(f'Total effect index $x_{i + 1}$', loc='left')
        
        # GP predicted solution
        plt.errorbar(results['up_list'], ST_gp, yerr=ST_gp_std,
                     linestyle='None', marker='.', color='C1',
                     label='GP predicted solution')
        
        # Plot analytical solution vs. GP prediction and Salib standard sln
        first_order, second_order, total = analytical_solution(7, 0.1)
    
        plt.axhline(total[i], color='k', linestyle='--',
                    label='Analytical solution')
        plt.xlabel('Number of GP predictions')
        plt.ylabel(f'$S_{{T_{i + 1}}}$')
        plt.legend(facecolor='lightgray', edgecolor='black', framealpha=0.3,
                   frameon=True)
        plt.xscale('log')
        plt.xlim(1E2, 1E5)


def plot_first_indices(results):
    """
    Plot first-order Sobol indices for the Ishigami function.

    Compares the first-order Sobol indices obtained from:
    - Gaussian Process (GP)-based surrogate model predictions.
    - Analytical reference solution.

    Parameters
    ----------
    results : dict
        Dictionary containing the GP upsample sensitivity results, including
        `up_results` and `up_list`.
    """
    for i in range(3):
        S1_gp = get_sobol_indices('S1', i, results['up_results'])
        S1_gp_std = get_sobol_indices('S1_conf', i, results['up_results'])
        
        plt.figure(f'First order effect x{i + 1}')
        plt.title(f'First order effect index $x_{i + 1}$', loc='left')
        
        # GP predicted solution
        plt.errorbar(results['up_list'], S1_gp, yerr=S1_gp_std,
                     linestyle='None', marker='.', color='C1',
                     label='GP predicted solution')
        
        # Plot analytical solution vs. GP prediction and Salib standard sln
        first_order, second_order, total = analytical_solution(7, 0.1)
    
        plt.axhline(first_order[i], color='k', linestyle='--',
                    label='Analytical solution')
        plt.xlabel('Number of GP predictions')
        plt.ylabel(f'$S_{{T_{i + 1}}}$')
        plt.legend(facecolor='lightgray', edgecolor='black', framealpha=0.3,
                   frameon=True)
        plt.xscale('log')
        plt.xlim(1E2, 1E5)


def plot_second_indices(results):
    """
    Plot second-order (interaction) Sobol indices for the Ishigami function.

    Compares the second-order Sobol indices obtained from:
    - Gaussian Process (GP)-based surrogate model predictions.
    - Analytical reference solution.

    Parameters
    ----------
    results : dict
        Dictionary containing the GP upsample sensitivity results, including
        `up_results` and `up_list`.
    """
    # Analytical solution
    first_order, second_order, total = analytical_solution(7, 0.1)
    interactions = [[0, 1], [0, 2], [1, 2]]
    for i, interaction in enumerate(interactions):
        # GP predicted solution
        S2_gp = [np.array(res['S2'])[interaction[0], interaction[1]]
                 for res in results['up_results']]
        S2_gp_std = [np.array(res['S2_conf'])[interaction[0], interaction[1]]
                     for res in results['up_results']]
        
        plt.figure(f'Second order effect x{interaction[0]}{interaction[1]}')
        plt.title((f'Second order effect index '
                   f'$x_{{{interaction[0] + 1}{interaction[1] + 1}}}$'),
                  loc='left')
        
        # GP predicted solution
        plt.errorbar(results['up_list'], S2_gp, yerr=S2_gp_std,
                     linestyle='None', marker='.', color='C1',
                     label='GP predicted solution')
        
        plt.axhline(second_order[i], color='k', linestyle='--',
                    label='Analytical solution')
    
        plt.xlabel('Number of GP predictions')
        plt.ylabel(f'$S_{{{interaction[0] + 1}{interaction[1] + 1}}}$')
        plt.legend(facecolor='lightgray', edgecolor='black', framealpha=0.3,
                   frameon=True)
        plt.xscale('log')
        plt.xlim(1E2, 1E5)


def main(file_name):
    """
    Load GP upsampling sensitivity results and generate comparison plots.

    Parameters
    ----------
    file_name : str
        Path to the JSON file containing GP upsampled sensitivity results.

    Returns
    -------
    None
        Generates and displays plots comparing analytical and GP-based
        Sobol indices across different numbers of GP predictions.
    """    
    results = rw.json_read_dictionary(file_name)
    plot_total_indices(results)
    plot_first_indices(results)
    plot_second_indices(results)
    

if __name__ == '__main__':
    pu.set_nes_plot_style()
    main('output/upsample_1.json')
    