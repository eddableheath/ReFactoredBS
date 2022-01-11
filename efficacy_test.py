# Efficacy test of Klein sampler
# Author: Edmund Dable-Heath
# Testing the efficacy of the Klein sampler for given bases and dimension, for comparison with boson sampler. The idea
# is to sample until the shortest vector is sampled, record how long it took, repeat until I have a good sample of these


import numpy as np
import sampler


def efficacy_experiment(lattice_basis, shortest_vector, no_tests, upper_sample_limit):
    """
    Testing the efficacy of the klein sampler in comparison to the boson sampler
    :param lattice_basis: lattice basis matrix
    :param shortest_vector: shortest vector in lattice, for comparison
    :param no_tests: number of tests to be run to collect sample
    :param upper_sample_limit: how many samples before it's deemed a failure.
    :return: ndarray of sample number counts
    """
    target_norm = round(np.linalg.norm(shortest_vector), 7)
    results = np.zeros(no_tests)
    for i in range(no_tests):
        count = 1
        while count < upper_sample_limit:
            sample = round(np.linalg.norm(sampler.klein_sampler(lattice_basis,
                                                                64)), 7)
            if sample <= target_norm:
                if sample > 0:
                    results[i] = count
                    break
                else:
                    count += 1
            else:
                count += 1
    return results


def lengths_efficacy(lattice_basis, no_tests):
    """
    Testing the efficacy of the klein sampler in comparison to the boson sampler on the lengths of the vectors
    :param lattice_basis: lattice basis matrix
    :param no_tests: number of tests to be run
    :return: ndarray of lengths of sampled vectors
    """
    results = np.zeros(no_tests)
    for i in range(no_tests):
        results[i] = np.linalg.norm(sampler.klein_sampler(lattice_basis, 64))
    return results

