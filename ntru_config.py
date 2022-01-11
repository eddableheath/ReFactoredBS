# Configuration for ntru experiment
# Author Edmund Dable-Heath
#  Config file for the ntru experiment

import numpy as np
import logging


def lattice_collector(latt, latt_type):
    return (np.genfromtxt('Lattices/ntru_lattices/'+str(latt)+'/'+latt_type+'.csv', delimiter=',', dtype=None),
             np.genfromtxt('Lattices/ntru_lattices/'+str(latt)+'/sv.csv', delimiter=',', dtype=None))

cores = 32
unitary_samples = 1200
no_samples = 20000
which_latt = 0
lattice_type = 'hnf'
lattices = lattice_collector(which_latt, lattice_type)