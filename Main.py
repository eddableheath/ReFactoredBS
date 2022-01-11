# Main file for this experiment
# Author: Edmund Dable-Heath
# Here is the main file, first it will generate the necessary distribution by dimension (in a parallelised manner) and
# then it will sample from those distributions (again in parallel), computing the necessary things as I go?
#
# Experiment 1: Comparing the average probability of the sampled vectors to the prob of the sampled shortest vector.
# Experiment 2: Kullback-Liebler divergence from a discrete Gaussian defined on the same sample.
# Experiment 3: Using the Gaussian Heuristic to test how spherical the distribution is.

import numpy as np
import Boson_Dist_Simple as sim
import Lattice_Sampler as sampler
import Stats as st
import multiprocessing as mp
import logging
import Config as config
from scipy.stats import unitary_group
import time
# from pympler import muppy, summary
# import sys
import gc
from Unitary_Construction import gram_unitary
import math


def main_experiment(lattice, shortest_vector, no_samples):
    """
    The main experiment, to be run on each lattice in each dimension from config file
    :param lattice: lattice basis
    :param shortest_vector: shortest vector
    :param no_samples: number of lattice vectors sampled.
    :return: shortest vector sample/ratio
    """
    gc.collect()
    dimension = lattice.shape[0]
    unitary = unitary_group.rvs(2*dimension+1)
    dist = sim.boson_sampler_dist(unitary)
    lattice_sample = sampler.lattice_sampler(lattice,
                                             shortest_vector,
                                             no_samples,
                                             dist)
    if st.check_for_smallest_vec(lattice_sample, shortest_vector):
        return st.shortest_sample_vs_mean_ratio(lattice_sample)
    else:
        return 0


def clifford_alg_main_experiment(lattice, shortest_vector, no_samples):
    """
    Main experiment with clifford x clifford algorithim for sampling.
    :param lattice: lattice basis
    :param shortest_vector: shortest vector
    :param no_samples: number of lattice vectors sampled
    :return: shortest vector sample/ratio
    """
    gc.collect()
    dimension = lattice.shape[0]
    unitary = unitary_group.rvs(2*dimension+1)
    lattice_sample = sampler.clifford_lattice_sampler(lattice,
                                                      shortest_vector,
                                                      no_samples,
                                                      unitary)
    if st.check_for_smallest_vec(lattice_sample, shortest_vector):
        return st.shortest_sample_vs_mean_ratio(lattice_sample)
    else:
        return 0


def kl_experiment_simple(lattice, shortest_vector, no_samples):
    """
    Main experiment with Kullback-Liebler divergence as cost function - for simple boson sampler
    :param lattice: lattice basis
    :param shortest_vector: shortest vector
    :param no_samples: number of lattice vectors sampled
    :return: Kullback-Liebler divergence between sampled distribution and Gaussian on same lattice.
    """
    gc.collect()
    dimension = lattice.shape[0]
    unitary = unitary_group.rvs(2*dimension + 1)
    dist = sim.boson_sampler_dist(unitary)
    lattice_sample = sampler.lattice_sampler(lattice,
                                             shortest_vector,
                                             no_samples,
                                             dist)
    if st.check_for_smallest_vec(lattice_sample, shortest_vector):
        return st.kl_div_from_gauss(lattice_sample, st.minkowski_bound(lattice))
    else:
        return 0


def kl_experiment_clifford(lattice, shortest_vector, no_samples):
    """
    Main experiment with Kullback-Liebler divergence as cost function - for clifford algorithm
    :param lattice: lattice basis
    :param shortest_vector: shortest vector
    :param no_samples: number of lattice vectors sampled
    :return: Kullback-Liebler divergence between the sample distribution and Gaussian distirbution defined on same Latt
    """
    gc.collect()
    dimension = lattice.shape[0]
    unitary = unitary_group.rvs(2*dimension+1)
    lattice_sample = sampler.clifford_lattice_sampler(lattice,
                                                      shortest_vector,
                                                      no_samples,
                                                      unitary)
    if st.check_for_smallest_vec(lattice_sample, shortest_vector):
        return st.kl_div_from_gauss(lattice_sample, st.minkowski_bound(lattice))
    else:
        return 0


def gauss_heuristic_experiment_simple(lattice, shortest_vector, no_samples, shortest_basis_vec_sphere=False):
    """
    Testing the Gaussian heuristic of the sampled lattice vectors for simple boson sampler algorithm
    :param lattice: lattice basis
    :param shortest_vector: shortest vector in lattice
    :param no_samples: number of lattice vectors to be sampled
    :param shortest_basis_vec_sphere: setting the size of the sphere for the GH to be the length of the shortest vector
    in the given lattice basis - optional, default is Minkowski bound as radius
    :return: abs(gh - number of lattice vectors within given sphere)
    """
    gc.collect()
    dimension = lattice.shape[0]
    unitary = unitary_group.rvs(2*dimension+1)
    dist = sim.boson_sampler_dist(unitary)
    lattice_samples = sampler.lattice_sampler(lattice,
                                              shortest_vector,
                                              no_samples,
                                              dist)
    if st.check_for_smallest_vec(lattice_samples, shortest_vector):
        if shortest_basis_vec_sphere:
            return st.gaussian_heuristic_test(lattice, lattice_samples, smallest_basis_volume=True)
        else:
            return st.gaussian_heuristic_test(lattice, lattice_samples)
    else:
        return 0


def gauss_heuristic_experiment_clifford(lattice, shortest_vector, no_samples, shortest_basis_vec_sphere=False):
    """
    Testing the Gaussian heuristic of the sampled lattice vectors for clifford boson sampler algorithm
    :param lattice: lattice basis
    :param shortest_vector: shortest vector in lattice
    :param no_samples: number of lattice vectors to be sampled
    :param shortest_basis_vec_sphere: setting the size of the sphere for the GH to be the length of the shortest vector
    in the given lattice basis - optional, default is Minkowski bound as radius
    :return: abs(gh - number of lattice vectors within given sphere)
    """
    gc.collect()
    dimension = lattice.shape[0]
    unitary = unitary_group.rvs(2*dimension+1)
    lattice_samples = sampler.clifford_lattice_sampler(lattice,
                                                       shortest_vector,
                                                       no_samples,
                                                       unitary)
    if st.check_for_smallest_vec(lattice_samples, shortest_vector):
        if shortest_basis_vec_sphere:
            return st.gaussian_heuristic_test(lattice, lattice_samples, smallest_basis_volume=True)
        else:
            return st.gaussian_heuristic_test(lattice, lattice_samples)
    else:
        return 0


def combined_experiment_simple(lattice, shortest_vector, no_samples,
                               shortest_basis_vec_sphere=False, gramm_unitary=False, theta=1):
    """
    Combination experiment for all three testable properties - for simple sampler algorithm.
    :param lattice: lattice basis
    :param shortest_vector: shortest vector in lattice
    :param no_samples: number of lattice vectors to be sampled
    :param shortest_basis_vec_sphere: setting the size of the sphere for the GH to be the length of the shortest vector
    in the given lattice basis - optional, default is Minkowski bound as radius
    :param gramm_unitary: decideds whether to use the gramm matrix to construct the unitary instead
    :param theta: theta value for the gramm unitary construction
    :return: np array containing the results of the three above experiments.
    """
    gc.collect()
    dimension = lattice.shape[0]
    if gramm_unitary:
        unitary = gram_unitary(lattice, zero=True, extended=True, theta=theta)
    else:
        unitary = unitary_group.rvs(2*dimension+1)
    dist = sim.boson_sampler_dist(unitary)
    lattice_samples = sampler.lattice_sampler(lattice,
                                              shortest_vector,
                                              no_samples,
                                              dist)
    if st.check_for_smallest_vec(lattice_samples, shortest_vector):
        results = [theta,
                   st.shortest_sample_vs_mean_ratio(lattice_samples),
                   st.kl_div_from_gauss(lattice_samples, st.minkowski_bound(lattice))]
        if shortest_basis_vec_sphere:
            results.append(st.gaussian_heuristic_test(lattice, lattice_samples, smallest_basis_volume=True))
        else:
            results.append(st.gaussian_heuristic_test(lattice, lattice_samples))
        return results
    else:
        return [theta, 0, 0, 0]


def combined_experiment_clifford(lattice, shortest_vector, no_samples,
                                 shortest_basis_vec_sphere=False, gramm_unitary=False, theta=1):
    """
    Combination experiment for all three testable properties - for clifford sampler algorithm
    :param lattice: lattice basis
    :param shortest_vector: shortest vector in lattice
    :param no_samples: number of lattice vectors to be sampled
    :param shortest_basis_vec_sphere: setting the size of the sphere for the GH to be the length of the shortest vector
    in the given lattice basis - optional, default is Minkowski bound as radius
    :param gramm_unitary: setting the unitary to be constructed from the gramm matrix instead
    :param theta: Changing the theta value for the gramm unitary
    :return: np array containnig the results of the three above experiments.
    """
    gc.collect()
    dimension = lattice.shape[0]
    if gramm_unitary:
        unitary = gram_unitary(lattice, zero=True, extended=True, theta=theta)
    else:
        unitary = unitary_group.rvs(2*dimension+1)
    lattice_samples = sampler.clifford_lattice_sampler(lattice,
                                                       shortest_vector,
                                                       no_samples,
                                                       unitary)
    if st.check_for_smallest_vec(lattice_samples, shortest_vector):
        results = [theta,
                   st.shortest_sample_vs_mean_ratio(lattice_samples),
                   st.kl_div_from_gauss(lattice_samples, st.minkowski_bound(lattice))]
        if shortest_basis_vec_sphere:
            results.append(st.gaussian_heuristic_test(lattice, lattice_samples, smallest_basis_volume=True))
        else:
            results.append(st.gaussian_heuristic_test(lattice, lattice_samples))
        return results
    else:
        return [theta, 0, 0, 0]



if __name__ == "__main__":

    # Tracking memeory usage
    # all_objects = muppy.get_objects()
    # sum1 = summary.summarize(all_objects)

    # Time stamp for results file.
    timestr = time.strftime("%Y-%m-%d-%H:%M")
    time_start = time.time()

    # Setting up logs.
    logging.basicConfig(filename='Logs/%s.txt' % timestr,
                        level=config.level,
                        filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s')

    pool = mp.Pool(config.cores)

    for latt in config.lattices:
        np.savetxt('Results/dim-%s-hnf.csv' % str(latt[0].shape[0]),
                   np.asarray([pool.apply(combined_experiment_simple,
                                          args=(latt[0],
                                                latt[2],
                                                config.no_samples,
                                                False,
                                                True,
                                                2*i*math.pi/config.unitary_samples))
                               for i in range(config.unitary_samples)]),
                   delimiter=',')
        np.savetxt('Results/dim-%s-lll.csv' % str(latt[0].shape[0]),
                   np.asarray([pool.apply(combined_experiment_simple,
                                          args=(latt[1],
                                                latt[2],
                                                config.no_samples,
                                                False,
                                                True,
                                                2*j*math.pi/config.unitary_samples))
                               for j in range(config.unitary_samples)]),
                   delimiter=',')

    pool.close()
    pool.join()

    # summary.print_(sum1)
