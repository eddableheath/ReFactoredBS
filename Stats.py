# Statistical fuckery
# Author: Edmund Dable-Heath
# In here we'll find the statistic's needed for working with the boson sampler and find out what we think can be done
# with it. Specifically we mostly care about the norms, and how good this is at random enumeration from the perspective
# of finding the shortest vector.

import numpy as np
import statistics as st
import NP_Array_Counting as count
import Gaussian_Functions as gf
import math
from scipy.special import gamma


def check_for_smallest_vec(lattice_vectors, shortest_vector):
    """
    Checks that the shortest vector is within the sample.
    :param lattice_vectors: sampled lattice vectors
    :param shortest_vector: shortest vector in lattice.
    :return:
    """
    norms = np.linalg.norm(lattice_vectors, axis=1)
    return bool(np.round(
        np.min(norms[np.nonzero(norms)]), 7)
                / np.round(np.linalg.norm(shortest_vector), 7) <= 1)


def shortest_sample_vs_mean_ratio(lattice_vectors):
    """
    Finding the ratio of the shortest vector occurrence rate to the mean occurrence rate
    :param lattice_vectors: sampled lattice vectors
    :return: mean lattice count / svp count
    """
    unique_norms, norm_counts = np.unique(np.linalg.norm(lattice_vectors, axis=1), return_counts=True)
    return np.mean(norm_counts) / norm_counts[0]


def energy_expectation(lattice_vectors):
    """
    Finding the sample mean of the energy of the sampled Ising states (lattice vectors) using David's Hamiltonian.
    However this just gives the vector norm squared, so I'll just take the average of that. We're seeking to just about
    minimise it.
    :param lattice_vectors: Sampled lattice vectors
    :return: Mean of norm squared
    """
    return st.mean([np.dot(vec, vec) for vec in lattice_vectors])


def kl_div_from_gauss(lattice_vectors, gauss_var=None):
    """
    Finds the kullback-liebler divergence from a discrete lattice Gaussian defined over the integers.
    :param lattice_vectors: Sampled lattice vectors
    :param gauss_var: variance of the Gaussian, taken to be the Minkowski bound as ideal
    :return: KL divergence
    """
    unique_vectors, counts = np.unique(lattice_vectors, axis=0, return_counts=True)
    sample_distribution = np.asarray([count for count in counts]) / len(lattice_vectors)

    def kl_divergence(p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    return kl_divergence(np.asarray([vals[1]
                                     for vals in gf.lattice_gaussian(unique_vectors, gauss_var)]),
                         sample_distribution)


def minkowski_bound(lattice):
    """
    Computation of the Minkowski bound.
    :param lattice: lattice basis
    :return: Minkowski bound
    """
    return math.sqrt(lattice.shape[0]) * (abs(np.linalg.det(lattice))**(1 / lattice.shape[0]))


def gaussian_heuristic_test(lattice, lattice_sample, smallest_basis_volume=False):
    """
    Cost function to be minimised that gives and idea of how circular the distribution is in the smallest regions.
    :param lattice: lattice basis.
    :param lattice_sample: sampled lattice vectors
    :param smallest_basis_volume: defining the set S, True sets to radius of smalled basis vector, False sets to Minkowski
    :return: abs(gaussian heuristic - no of lattice vectors within set S)
    """

    if smallest_basis_volume:
        sphere_radius = np.linalg.norm(lattice[np.argmin(np.linalg.norm(lattice, axis=0))])
    else:
        sphere_radius = minkowski_bound(lattice)

    dimension = lattice.shape[0]
    sphere_volume = ((math.pi**(dimension/2)) * (sphere_radius**dimension)) / gamma(dimension/2 + 1)
    gaussian_heuristic = sphere_volume / abs(np.linalg.det(lattice))
    return abs(gaussian_heuristic - sum((np.array([np.linalg.norm(vec)
                                                   for vec in np.unique(lattice_sample, axis=0)]) <= sphere_radius)))


if __name__ == "__main__":
    latt2hnf = np.genfromtxt("Lattices/2/0/2.csv", delimiter=',', dtype=None)
    small_basis_rad = np.linalg.norm(latt2hnf[np.argmin(np.linalg.norm(latt2hnf, axis=0))])
    sphere_vol = (math.pi**(latt2hnf.shape[0]/2) / gamma(latt2hnf.shape[0]/2 +1))
    print(np.unique(latt2hnf, axis=0))
    print(np.linalg.norm(latt2hnf, axis=0))
    print(np.linalg.norm(np.unique(latt2hnf, axis=0), axis=0))
    print((np.linalg.norm(latt2hnf, axis=0) <= minkowski_bound(latt2hnf)))
    print(sum((np.linalg.norm(latt2hnf, axis=0) <= minkowski_bound(latt2hnf))))
