# Testing klein sampler efficacy
# Author: Edmund Dable-Heath
# Testing to see how effective the klein sampler is.

import numpy as np
import multiprocessing as mp
import gc
import os
from re import findall
from itertools import filterfalse
import efficacy_test as ke
import klein_config as config


def klein_efficacy_experiment(iterable, no_tests, upper_sample_limit, basis_type, dimension):
    gc.collect()
    file_path = str(dimension)+'/'+str(iterable)+'/'
    np.savetxt('Efficacy_results/'+str(iterable)+'.csv',
               ke.lengths_efficacy(np.genfromtxt('Lattices/'+file_path+basis_type+'.csv',
                                                 delimiter=',',
                                                 dtype=None),
                                   no_tests),
               delimiter=',')


if __name__=="__main__":
    iterable_range = 32
    if len(os.listdir('Efficacy_results/')) == 0:
        iterables = range(iterable_range)
    else:
        (_, _, results_names) = next(os.walk('Efficacy_results/'))
        results_numbers = [int(findall(r'\d+', string)[0]) for string in results_names]
        iterables = filterfalse(lambda x: x in results_numbers, range(iterable_range))

    pool = mp.Pool(config.cores)
    [pool.apply(klein_efficacy_experiment,
                args=(i,
                      config.no_tests,
                      config.sample_limit,
                      config.basis_type,
                      config.dimension))
     for i in iterables]

    pool.close()
    pool.join()