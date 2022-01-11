# Configuration File for experiment
# Author: Edmund Dable-Heath
# Here will be the controls for the experiment, how many cores, samples, which lattices etc.

import numpy as np
import multiprocessing as mp
import logging


# def lattice_collector(min_dim, max_dim):
#     return [(np.genfromtxt('Lattices/'+str(dim)+'/0/0.csv', delimiter=',', dtype=None),
#              np.genfromtxt('Lattices/' + str(dim) + '/0/2.csv', delimiter=',', dtype=None),
#              np.genfromtxt('Lattices/'+str(dim)+'/0/4.csv', delimiter=',', dtype=None)) for dim in range(min_dim, max_dim+1)]


# cores = mp.cpu_count()
cores = 4
unitary_samples = 10
no_samples = 10000
min_lattice_dim = 2
max_lattice_dim = 6
lattices = lattice_collector(min_lattice_dim, max_lattice_dim)

# Logging level:
level = logging.WARNING
