# Configuration file for Gaussian Boson Sampling experiment
# Author: Edmund Dable-Heath
# Configuration for the GBS experiement can be tweaked here.

import numpy as np


def lattice_collector(dim, latt, latt_type):
    return (np.genfromtxt('Lattices/'+str(dim)+'/'+str(latt)+'/'+latt_type+'.csv', delimiter=',', dtype=None),
             np.genfromtxt('Lattices/'+str(dim)+'/'+str(latt)+'/sv.csv', delimiter=',', dtype=None))

# cpu params
cores = 4

# Testing parameters
unitary_samples = 10
no_samples = 10

# lattice parameters
which_latt = 0
dimension = 2
lattice_type = 'hnf'
lattices = lattice_collector(dimension, which_latt, lattice_type)