# Configuration File for experiment
# Author: Edmund Dable-Heath
# Here will be the controls for the experiment, how many cores, samples, which lattices etc.

import numpy as np
import logging


def lattice_collector(dim, latt, latt_type):
    return (np.genfromtxt('Lattices/'+str(dim)+'/'+str(latt)+'/'+latt_type+'.csv', delimiter=',', dtype=None),
             np.genfromtxt('Lattices/'+str(dim)+'/'+str(latt)+'/sv.csv', delimiter=',', dtype=None))


# cores = mp.cpu_count()
cores = 4
unitary_samples = 1200
no_samples = 10000
which_latt = 10
dimension = 2
lattice_type = 'lll'
lattices = lattice_collector(dimension, which_latt, lattice_type)


# Logging level:
level = logging.WARNING
