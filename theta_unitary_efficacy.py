# Testing the efficacy of the exponetially constructed unitaries
# Author: Edmund Dable-Heath
# Here we test how effective the good unitaries that have been found actually are via brute force. Sampling
# from them over and over until the shortest vector is found, rinse and repeat.

import numpy as np
import Testing_unitaries as ut
from Unitary_Construction import gram_unitary
import multiprocessing as mp
import gc
import math
import efficacy_config as config
import os
from re import findall
from itertools import filterfalse


def efficacy_experiment(lattice, theta, shortest_vector, no_tests, iterable):
    gc.collect()
    np.savetxt('Efficacy_results/'+str(iterable)+'.csv', ut.basic_sampler_lengths(gram_unitary(lattice,
                                                                                               zero=True,
                                                                                               extended=True,
                                                                                               theta=theta),
                                                                                  lattice,
                                                                                  shortest_vector,
                                                                                  no_tests),
               delimiter=',')


if __name__ == "__main__":
    # Take top ten percent of theta values for each lattice
    # run above experiment for each theta value for each lattice, see how they do on average.

    testing_results = config.lattice_results
    zeros_filter = [not bool(r_val[1] == 0) for r_val in testing_results]
    zero_filtered = testing_results[zeros_filter]
    percentile_filter = [bool(np.percentile(zero_filtered,
                                            config.lower_percentile_constraint,
                                            axis=0)[1]
                              < r_val[1] <
                              np.percentile(zero_filtered,
                                            config.upper_percentile_constraint,
                                            axis=0)[1])
                         for r_val in zero_filtered]
    filtered_results = zero_filtered[percentile_filter]

    pool = mp.Pool(config.cores)

    latt = config.lattice

    # Genera_ting iterators
    iterable_range = sum(percentile_filter)
    if len(os.listdir('Efficacy_results/')) == 0:
        iterables = range(iterable_range)
    else:
        (_, _, result_names) = next(os.walk('Efficacy_results/'))
        result_numbers = [int(findall(r'\d+', string)[0]) for string in result_names]
        iterables = filterfalse(lambda x: x in result_numbers, range(iterable_range))

    [pool.apply(efficacy_experiment,
                args=(latt[0],
                      filtered_results[i][0],
                      latt[1],
                      config.no_tests,
                      i))
     for i in iterables]

    pool.close()
    pool.join()