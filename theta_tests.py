# Testing the theta parameter
# Author: Edmund Dable-Heath
# Testing to see how common the values of theta that give a good probability of finding the shortest vector are over
# different lattices and different lattice bases (hnf and lll)

import numpy as np
import Boson_Dist_Simple as sim
import Lattice_Sampler as sampler
import Stats as st
import multiprocessing as mp
# import logging
import Theta_Config as config
# from scipy.stats import unitary_group
import gc
from Unitary_Construction import gram_unitary
import math
# for the restarting procedure
import os
from re import findall
from itertools import filterfalse
import time


def combined_kl_ratio(lattice, shortest_vector, no_samples, theta=1):
    """
    Combination KL and ratio test for testing the different theta values. Using simpler algorithm.
    :param lattice: lattice basis
    :param shortest_vector: shortest vector in lattice (for solving for number of samples needed per lattice sample)
    :param no_samples: number of lattice vectors to be sampled
    :param theta: theta value for construction of the unitary from the gramm matrix
    :return: [theta, ratio, KL]
    """
    gc.collect()
    unitary = gram_unitary(lattice, zero=True, extended=True, theta=theta)
    dist = sim.boson_sampler_dist(unitary)
    lattice_sample = sampler.lattice_sampler(lattice,
                                             shortest_vector,
                                             no_samples,
                                             dist)
    if st.check_for_smallest_vec(lattice_sample, shortest_vector):
        return [theta,
                st.shortest_sample_vs_mean_ratio(lattice_sample),
                st.kl_div_from_gauss(lattice_sample, st.minkowski_bound(lattice))]
    else:
        return [theta, 0, 0]


def combined_write_results_by_line(lattice, shortest_vector, no_samples,  iterable, unitary_samples):
    """
    Re writing of experiment to write line each time a result is found, not write to file once all results are found
    :param lattice: lattice basis
    :param shortest_vector: shortest vector in lattice
    :param no_samples: number of lattice samples to be found
    :param iterable: iteration of experiment, for recreation, ordering and restarting
    :param unitary_samples: number of unitary samples being taken
    :return: just writes to csv
    """
    np.savetxt('Results/'+str(iterable)+'.csv',
               combined_kl_ratio(lattice,
                                 shortest_vector,
                                 no_samples,
                                 2*math.pi*iterable/unitary_samples),
               delimiter=',')


def no_kl_experiment(lattice, shortest_vector, no_samples, theta=1):
    """
    Just the ratio test to save on computation power.
    :param lattice: lattice basis
    :param shortest_vector: shortest vector in lattice
    :param no_samples: number of lattices vectors to be sampled
    :param theta: theta value for construction of the unitary from the gramm matrix
    :return: [theta, ratio]
    """
    gc.collect()
    unitary = gram_unitary(lattice,
                           zero=True,
                           extended=True,
                           theta=theta)
    dist = sim.boson_sampler_dist(unitary)
    lattice_sample = sampler.alt_output_lattice_sampler(lattice,
                                                        shortest_vector,
                                                        no_samples,
                                                        dist)
    if st.check_for_smallest_vec(lattice_sample,
                                 shortest_vector):
        return [theta,
                st.shortest_sample_vs_mean_ratio(lattice_sample)]
    else:
        return [theta, 0]


def just_ratio_line_write(lattice, shortest_vector, no_samples, iterable, unitary_samples):
    """
    Writes the results of the above experiment as single line csv's to be combined later.
    :param lattice: lattice basis
    :param shortest_vector: shortest vector in lattice
    :param no_samples: number of lattice vectors to be sampled
    :param iterable: iteration of theta parameter, for labelling of the csv
    :param unitary_samples: total number of unitary samples
    :return: just writes to csv
    """
    np.savetxt('Results/'+str(iterable)+'.csv',
               no_kl_experiment(lattice,
                                shortest_vector,
                                no_samples,
                                2*math.pi*iterable / unitary_samples),
               delimiter=',')


if __name__ == "__main__":

    pool = mp.Pool(config.cores)

    latt = config.lattices

    # generating iterators - - - - - - - - -
    iterable_range = config.unitary_samples
    if len(os.listdir('Results/')) == 0:
        iterables = range(iterable_range)
    else:
        (_, _, result_names) = next(os.walk('Results/'))
        result_numbers = [int(findall(r'\d+', string)[0]) for string in result_names]
        iterables = filterfalse(lambda x: x in result_numbers, range(iterable_range))

    [pool.apply(just_ratio_line_write,
                args=(latt[0],
                      latt[1],
                      config.no_samples,
                      i,
                      config.unitary_samples)) for i in iterables]

    pool.close()
    pool.join()

    # Compiling results into single csv - - - - - -
    if len(os.listdir('Results/')) == iterable_range:
        hnf_results = np.zeros((len(os.listdir('Results/')), 2))
        for file in os.listdir('Results/'):
            hnf_results[int(findall(r'\d+', file)[0])] = np.genfromtxt('Results/'+file, delimiter=',', dtype=None)
        np.savetxt('Results/results-'+str(config.dimension)+str(config.which_latt)+'.csv', hnf_results, delimiter=',')
        for i in range(iterable_range):
            if os.path.exists('Results/'+str(i)+'.csv'):
                os.remove('Results/'+str(i)+'.csv')

    start_time = time.time()

    just_ratio_line_write(latt[0], latt[1], config.no_samples, 400, config.unitary_samples)

    print(time.time() - start_time)
