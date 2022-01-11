# Gaussian Functions
# Author: Edmund Dable-Heath
# Computing the discrete gaussian for a finite portion of a lattice.

import numpy as np


def gaussian(vector, mean='zero', var=1):
    """
    Computing the gaussian for a vector
    :param vector: vector
    :param mean: mean of the distribution (default= zero vector of dimension len(vector)
    :param var: variance of the distribution (default=1)
    :return: the value of Gaussian
    """
    # print(mean)
    if mean == 'zero':
        mean = np.zeros(len(vector))
    return np.exp(-((np.linalg.norm(vector - mean)**2) / (2 * var**2)))


def lattice_gaussian(vectors, mean=None, var=1):
    """
    Computing the discrete gaussian for a finite slice of a lattice
    :param vectors: set of input vectors (numpy arrays)
    :param mean: mean of the distribution (default = 0)
    :param var: variance of the distribution (default = 1)
    :return: set of vectors with their weighted lattice gaussian value
    """
    weights = [
        gaussian(vector, mean, var)
        for vector in vectors
    ]
    total_weight = sum(weights)
    return [
        (vector, weight/total_weight)
        for vector, weight in zip(vectors, weights)
    ]

