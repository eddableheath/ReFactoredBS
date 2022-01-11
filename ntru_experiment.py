# Testing ntru lattices for theta algorithm
# Author: Edmund Dable-Heath
# Testing to see what the outcome of the theta approach is for ntru lattices.

import numpy as np
import Boson_Dist_Simple as sim
import Lattice_Sampler as sampler
import Stats as st
import multiprocessing as mp
# import logging
import ntru_config_specific as config
# from scipy.stats import unitary_group
import gc
from Unitary_Construction import gram_unitary
import math
# for the restarting procedure
import os
from re import findall
from itertools import filterfalse
import time


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
    unitary = gram_unitary(lattice, zero=True, extended=True, theta=theta)
    dist = sim.boson_sampler_dist(unitary)
    lattice_sample = sampler.lattice_sampler(lattice,
                                             shortest_vector,
                                             no_samples,
                                             dist)
    if st.check_for_smallest_vec(lattice_sample, shortest_vector):
        return [theta, st.shortest_sample_vs_mean_ratio(lattice_sample)]
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


if __name__=="__main__":

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

