# Config for efficacy test
# Author: Edmund Dable-Heath
# Configuration file for the testing of the efficacy of the lattices.

import numpy as np


def lattice_collector(dim, latt, latt_type):
    return (np.genfromtxt('Lattices/'+str(dim)+'/'+str(latt)+'/'+latt_type+'.csv', delimiter=',', dtype=None),
             np.genfromtxt('Lattices/'+str(dim)+'/'+str(latt)+'/sv.csv', delimiter=',', dtype=None))


# Testing parameters
no_tests = 10
sample_limit = 100
lower_percentile_constraint = 0
upper_percentile_constraint = 10

# How many cores?
cores = 4

# Which lattice?
dimension = 2
which_latt = 0
latt_type = 'hnf'
lattice_results = np.genfromtxt('Results/results-'+str(dimension)+str(which_latt)+'.csv',
                                delimiter=',',
                                dtype=None)
lattice = lattice_collector(dimension, which_latt, latt_type)