# Main file for the algorithm
# Author: Edmund Dable-Heath
# Here is where I'll be including the main file for the implementation of the Clifford X Clifford Boson Sampling
# algorithm. Assumption of photons = floor(sqrt(modes)). (modes = poly(photons))

import numpy as np
import logging
from scipy.stats import unitary_group as ug
import math
import random as rn
from Clifford_Clifford_Algorithm.Laplace_Expansion import laplace_expansion
import gc
import time


def boson_sampler(unitary):
    """
    Get's a single sample from the Boson sampler defined by the input unitary.
    :param unitary: control unitary, contains all information necessary for simulation.
    :return: a single sample of the boson sampler.
    """
    modes = unitary.shape[0]
    photons = math.floor(math.sqrt(modes))
    permuted_input_relevant_columns = np.random.permutation(unitary[:, :photons].T).T
    initial_probability = np.abs(permuted_input_relevant_columns[:, :1].T[0])**2
    # assert np.sum(initial_probability) == 1, \
    #     'Probabilities do not add up' in logging.WARNING
    sample_array = np.array(rn.choices(range(initial_probability.shape[0]), weights=initial_probability))
    for k in range(2, photons+1):
        # k here is indexed from one (but starts at 2 for the algorithm) because of the way indexing numpy arrays works.
        permanent_computation_matrix = permuted_input_relevant_columns[sample_array, :k]
        assert permanent_computation_matrix.shape[0] == k-1, \
            f'Permanent computation matrix is wrong shape' in logging.WARNING
        pre_proportioned_pmf = laplace_expansion(permanent_computation_matrix,
                                                                   permuted_input_relevant_columns[:, :k])
        proportioned_pmf = pre_proportioned_pmf / np.sum(pre_proportioned_pmf)
        sample_array = np.append(sample_array,
                                 np.array(rn.choices(range(initial_probability.shape[0]), weights=proportioned_pmf)))
    sample_array.sort()
    gc.collect()
    return sample_array


if __name__ == "__main__":
    time_start = time.time()
    print(boson_sampler(ug.rvs(5)))
    print(time.time() - time_start)
    print((time.time() - time_start)*10008)
