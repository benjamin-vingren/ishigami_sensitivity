
import numpy as np
import utilities.samplers as samplers
from SALib.sample.sobol import sample as SobolSample
import scipy.stats.qmc as qmc


if __name__ == '__main__':
    # Sample with regular Sobol sequence
    bounds = {'x1': [-np.pi, np.pi],
              'x2': [-np.pi, np.pi],
              'x3': [-np.pi, np.pi]}
    samples_1 = samplers.get_sobol_samples(bounds, 1024)
    samples_1 = np.array([samples_1['x1'], samples_1['x2'], samples_1['x3']]).T

    # Sample with Sobol sequence for sensitivity analysis
    problem = {'num_vars': 3, 'names': ['x1', 'x2', 'x3'],
               'bounds': [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]]}
    samples_2 = SobolSample(problem, 128)

    print('Number of unique samples - regular Sobol')
    for i in range(3):
        n_unique = len(np.unique(samples_1[:, i]))
        print(f'x{i + 1}: {n_unique}')

    print('Number of unique samples sensitivity Sobol')
    for i in range(3):
        n_unique = len(np.unique(samples_2[:, i]))
        print(f'x{i + 1}: {n_unique}')

    # Calculate discrepancy
    norm_1 = (samples_1 + np.pi) / (2 * np.pi)
    norm_2 = (samples_2 + np.pi) / (2 * np.pi)
    discrepancy_1 = qmc.discrepancy(norm_1, method='MD')
    discrepancy_2 = qmc.discrepancy(norm_2, method='MD')

    print(f'Discrepancy regular Sobol: {discrepancy_1 * 1E6}')
    print(f'Discrepancy sensitivity Sobol: {discrepancy_2 * 1E6}')