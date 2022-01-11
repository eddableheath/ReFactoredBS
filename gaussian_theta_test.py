# Theta parameter tests via Gaussian Boson Sampling
# Author: Edmund Dable-Heath
# Similar experiments to the previous tests of the theta parameter set up, but this time using Gaussian boson
# sampling to see how effective this is. Potentially then leading on to using the inbuilt training systems.
#

import numpy as np
import Stats as st
import Gaussian_config as config
import multiprocessing as mp
import gc
from Unitary_Construction import gram_unitary
import os
from re import findall
from itertools import filterfalse
from Lattice_Sampler import boson_samples_per_lattice_sample as sample_count
from StrawberryFields import gbs_lattice_sampler as sampler
import math


def ratio_experiment(lattice, shortest_vector, no_samples, theta=1):
    """
    Experimenting on the Gaussian boson sampler using the ratio figure of merit.
    :param lattice: lattice basis
    :param shortest_vector: shortest vector in lattice
    :param no_samples: number of lattice samples
    :param theta: theta parameter for construction of unitary
    :return: returns theta value and ratio value in list
    """
    gc.collect()
    unitary = gram_unitary(lattice, extended=True, theta=theta)
    lattice_sample = sampler(lattice, unitary, no_samples, sample_count(lattice, shortest_vector))
    print(f'lattice sample: {lattice_sample}')
    if st.check_for_smallest_vec(lattice_sample, shortest_vector):
        return [theta, st.shortest_sample_vs_mean_ratio(lattice_sample)]
    else:
        return [theta, 0]


def experiment_writer(lattice, shortest_vector, no_samples, iterable, unitary_samples):
    np.savetxt('Gaussian_Results/'+str(iterable)+'.csv',
               ratio_experiment(lattice,
                                shortest_vector,
                                no_samples,
                                2*math.pi*iterable / unitary_samples),
               delimiter=',')


if __name__ == "__main__":

    pool = mp.Pool(config.cores)

    latt = config.lattices

    # generating iterators - - - - - - - -
    iterable_range = config.unitary_samples
    if len(os.listdir('Gaussian_Results/')) == 0:
        iterables = range(iterable_range)
    else:
        (_, _, result_names) = next(os.walk('Results/'))
        result_numbers = [int(findall(r'\d+', string)[0]) for string in result_names]
        iterables = filterfalse(lambda x: x in result_numbers, range(iterable_range))

    [pool.apply(experiment_writer,
                args=(latt[0],
                      latt[1],
                      config.no_samples,
                      i,
                      config.unitary_samples)) for i in iterables]

    pool.close()
    pool.join()
