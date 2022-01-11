# Testing the effictive sample size needed for the boson distribution of a given unitary.
# Author: Edmund Dable-Heath
# Given a decent unitary, how useful will it be? I.e. what is the expected no of samples before it finds the shortest
# vector? To test this will require testing the number of samples needed until found for each unitary many times over.

import numpy as np
import Lattice_Sampler
import Boson_Dist_Simple
from scipy.stats import unitary_group
# import statistics as stats
import time


def basic_sampler(unitary, lattice, shortest_vector, no_tests, upper_sample_limit):
    """
    Testing the efficacy of the given unitary using the basic sampler algorithm.
    :param unitary: unitary to be tested (ndarray)
    :param lattice: lattice to be sampled from (ndarray)
    :param shortest_vector: shortest vector of the lattice (ndarray)
    :param no_tests: number of repeats of the tests from the unitary.
    :param upper_sample_limit: upper limit on the number of samples.
    :return: ndarray of sample number counts.
    """
    distribution = Boson_Dist_Simple.boson_sampler_dist(unitary)
    results = np.zeros(no_tests)
    boson_samples_per_sample = Lattice_Sampler.boson_samples_per_lattice_sample(lattice, shortest_vector)
    target_norm = round(np.linalg.norm(shortest_vector), 7)
    for i in range(no_tests):
        print('----------------------------')
        print(f'shortest_vector: {shortest_vector}')
        count = 1
        while count < upper_sample_limit:
            sample = round(np.linalg.norm(Lattice_Sampler.single_lattice_sampler_basic(lattice,
                                                                                       distribution,
                                                                                       boson_samples_per_sample), 7))
            print(f'sample: {sample}')
            if sample <= target_norm:
                if sample > 0:
                    results[i] = count
                    print(count)
                    break
                else:
                    count += 1
            else:
                count += 1
    return results


def basic_sampler_lengths(unitary, lattice, shortest_vector, no_tests):
    """
    Testing the efficacy of the given unitary using the basic sampler algorithm by looking at the lengths of vectors.
    :param unitary: unitary to be tested
    :param lattice: lattice to be sampled from
    :param shortest_vector: shortest vector in the lattice
    :param no_tests: number of lattice vectors to be sampled.
    :return: ndarray of lengths of vectors
    """
    distribution = Boson_Dist_Simple.boson_sampler_dist(unitary)
    boson_samples_per_sample = Lattice_Sampler.boson_samples_per_lattice_sample(lattice, shortest_vector)
    return np.linalg.norm(Lattice_Sampler.lattice_sampler(lattice, shortest_vector, no_tests, distribution), axis=1)


def clifford_algorithm(unitary, lattice, shortest_vector, no_tests, upper_sample_limit):
    """
    Testing the efficacy of the given unitary using the clifford sampler algorithm.
    :param unitary: unitary to be tested (ndarray)
    :param lattice: lattice to be sampled from (ndarray)
    :param shortest_vector: shortest vector of the lattice (ndarray)
    :param no_tests: number of repeats of the test from the unitary
    :param upper_sample_limit: upper limit on number of samples.
    :return: ndarray of sample number of counts.
    """
    results = np.zeros(no_tests)
    boson_samples_per_sample = Lattice_Sampler.boson_samples_per_lattice_sample(lattice,
                                                                                shortest_vector)
    target_norm = round(np.linalg.norm(shortest_vector), 7)
    for i in range(no_tests):
        count = 1
        while count < upper_sample_limit:
            sample = round(np.linalg.norm(Lattice_Sampler.single_lattice_sample_clifford_alg(lattice,
                                                                                             unitary,
                                                                                             boson_samples_per_sample)),
                           7)
            if sample <= target_norm:
                if sample > 0:
                    results[i] = count
                    break
                else:
                    count += 1
            else:
                count += 1
        if results[i] == 0:
            results[i] = count
        else:
            continue
    return results


# if __name__ == "__main__":
#
#     latt = np.genfromtxt('Lattices/2/0/0.csv', delimiter=',', dtype=None)
#     SV = np.genfromtxt('Lattices/2/0/4.csv', delimiter=',', dtype=None)
#
#     u = unitary_group.rvs(2*latt.shape[0] + 1)
#
#     basic_time_start = time.time()
#     r_1 = basic_sampler(u, latt, SV, 10, 100)
#     basic_time = time.time() - basic_time_start
#     clifford_time_start = time.time()
#     r_2 = clifford_algorithm(u, latt, SV, 10, 100)
#     clifford_time = time.time() - clifford_time_start
#     print(r_1)
#     print(np.mean(r_1))
#     print(f'Basic alg time {basic_time} ------------------')
#     print(r_2)
#     print(np.mean(r_2))
#     print(f'Clifford alg time {clifford_time} ------------')
