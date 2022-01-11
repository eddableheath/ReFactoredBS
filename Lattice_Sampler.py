# Sampling from a lattice
# Author: Edmund Dable-Heath
# Here we actually sample from the lattice. This takes an input of a lattice basis and a number of samples and returns
# the required number of lattice vectors as specified by the sample number. Each lattice vector will be the sum of a
# number of basis vectors, chosen to ensure the existence of the shortest vector within the sample space. For simplicity
# number of basis vectors used will be found by simple linear algebra, with the recognition that a future algorithm to
# decide this number will have to be developed. An upper bound on this number for HNF lattice bases has been derived,
# but that's for the theory for now. Any considerations of 'space multiplication' (the mapping of the basis to the
# output modes multiple time) will have to be left for future experiments.
#
# Could this be done using classes? Do simple version first.

import numpy as np
import Boson_Dist_Simple as sim
from scipy.stats import unitary_group
import random as rn
from Clifford_Clifford_Algorithm.Main import boson_sampler
import sys
import gc


def boson_samples_per_lattice_sample(lattice, shortest_vector):
    """
    Giving the number of samples per lattice sample by linear algebra.
    :param lattice: lattice basis (ndarray)
    :param shortest_vector: shortest vector in lattice (ndarray)
    :return: number of samples required (int)
    """
    return int(sum(abs(np.linalg.solve(lattice.T, shortest_vector))))


def unitary_test(unitary):
    """
    Compute the probability that there are repeats of the modes chosen by the bosons based on the control unitary.
    :param unitary: Control unitary for Boson Sampling.
    :return: probability of repeat elements
    """
    computing_unitary = unitary.T
    # return 2 * np.sum(np.abs(np.concatenate([[np.multiply(computing_unitary[i],
    #                                                       computing_unitary[k])
    #                                           for k in range(i+1, unitary.shape[0])]
    #                                          for i in range(unitary.shape[0] - 1)]))**2)
    row_mults = []
    for k in range(unitary.shape[0]-1):
        for l in range(k+1, unitary.shape[0]):
            row_mult = np.multiply(computing_unitary[k], computing_unitary[l])
            row_mults.append(row_mult)
    all_terms = np.concatenate(row_mults)
    abs_terms = np.abs(all_terms)
    squared_terms = abs_terms**2
    summed_terms = np.sum(squared_terms)
    return 2*summed_terms


def lattice_sampler(lattice, shortest_vector, number_of_samples, distribution):
    """
    Samples from a given lattice a given number of times using a given distribution.
    :param lattice: lattice basis (ndarray)
    :param shortest_vector: shortest vector in lattice (ndarray)
    :param number_of_samples: number of lattice vectors in sample (int)
    :param distribution: the boson sampler distribution to be used in the sample.
    :return: numpy array of lattice vectors (ndarray: no_samples X lattice_dimension)
    """
    dimension = lattice.shape[0]
    modes = 2*dimension+1
    basis_vectors_per_vector = boson_samples_per_lattice_sample(lattice, shortest_vector)
    # print(basis_vectors_per_vector)
    near_identity = np.identity(modes)
    near_identity[0][0] = 0.
    latt_vecs = []
    for i in range(number_of_samples):
        boson_sample = np.delete(np.sum(np.sum([near_identity[i]
                                                for i in rn.choices([i[0]
                                                                     for i in distribution],
                                                                    weights=[i[1]
                                                                             for i in distribution],
                                                                    k=basis_vectors_per_vector)],
                                               axis=0),
                                        axis=0),
                                 0)
        assert len(boson_sample) % 2 == 0, \
            f'Not an even sample, somethings has gone wrong {boson_sample}'
        split = np.split(boson_sample, 2)
        integer_coefficients = split[0] - split[1]
        assert integer_coefficients.shape[0] == lattice.shape[0], \
            f"Integer vector does not concur with lattice {integer_coefficients}"
        latt_vecs.append(np.dot(integer_coefficients, lattice))
        gc.collect()
    return np.asarray(latt_vecs)


def alt_output_lattice_sampler(lattice, shortest_vector, number_of_samples, distribution):
    """
    Samples from a given lattice a given number of times using a given distribution using the alternative output mapping
    :param lattice: lattice basis (ndarray)
    :param shortest_vector: shortest vector in lattice (ndarray)
    :param number_of_samples: number of lattice vectors in sample (int)
    :param distribution: the boson sampler distribution to be used in the sample.
    :return: numpy array of lattice vectors (ndarray: no_samples X lattice_dimension)
    """
    dimension = lattice.shape[0]
    modes = 2*dimension+1
    basis_vectors_per_vector = boson_samples_per_lattice_sample(lattice, shortest_vector)
    # print(basis_vectors_per_vector)
    near_identity = np.identity(modes)
    near_identity[0][0] = 0.
    latt_vecs = []
    for i in range(number_of_samples):
        boson_sample = np.delete(np.sum(np.sum([near_identity[i]
                                                for i in rn.choices([i[0]
                                                                     for i in distribution],
                                                                    weights=[i[1]
                                                                             for i in distribution],
                                                                    k=basis_vectors_per_vector)],
                                               axis=0),
                                        axis=0),
                                 0)
        assert len(boson_sample) % 2 == 0, \
            f'Not an even sample, somethings has gone wrong {boson_sample}'
        split = np.split(boson_sample, 2)
        integer_coefficients = split[0] - split[1]
        assert integer_coefficients.shape[0] == lattice.shape[0], \
            f"Integer vector does not concur with lattice {integer_coefficients}"
        latt_vecs.append(np.dot(integer_coefficients, lattice))
        gc.collect()
    return np.asarray(latt_vecs)


def single_lattice_sampler_basic(lattice, distribution, samples_per_sample):
    """
    Single sample algorithm for improved efficieny when required.
    :param lattice: lattice basis (ndarray)
    :param distribution: boson sampler distribution
    :param samples_per_sample: number of samples from the boson sampler distribution per lattice sample.
    :return: single lattice vector.
    """
    dimension = lattice.shape[0]
    modes = 2*dimension+1
    near_identity = np.identity(modes)
    near_identity[0][0] = 0.
    boson_sample = np.delete(np.sum(np.sum([near_identity[i]
                                            for i in rn.choices([i[0]
                                                                 for i in distribution],
                                                                weights=[i[1]
                                                                         for i in distribution],
                                                                k=samples_per_sample)],
                                           axis=0),
                                    axis=0),
                             0)
    split = np.split(boson_sample, 2)
    integer_coefficients = split[0] - split[1]
    return np.dot(integer_coefficients, lattice)


def clifford_lattice_sampler(lattice, shortest_vector, number_of_samples, unitary):
    """
    Lattice sampler using the clifford algorithm instead of the basic one.
    :param lattice: lattice basis, numpy array.
    :param shortest_vector: shortest vector in the lattice, used for finding how many times to sample per vector.
    :param number_of_samples: total number of vectors to sample from the lattice.
    :param unitary: control unitary for boson sampling.
    :return: lattice vectors.
    """
    dimension = lattice.shape[0]
    modes = 2*dimension+1
    basis_vectors_per_vector = boson_samples_per_lattice_sample(lattice, shortest_vector)
    near_identity = np.identity(modes)
    near_identity[0][0] = 0.
    latt_vecs = []
    for i in range(number_of_samples):
        boson_sample = np.delete(np.sum([np.sum([near_identity[i]
                                                 for i in boson_sampler(unitary)],
                                                axis=0)
                                         for j in range(basis_vectors_per_vector)],
                                        axis=0),
                                 0)
        assert len(boson_sample) % 2 == 0, \
            f'Not an even sample, something has gone wrong {boson_sample}'
        split = np.split(boson_sample, 2)
        integer_coefficients = split[0] - split[1]
        assert integer_coefficients.shape[0] == lattice.shape[0], \
            f'Integer vector does not concur with lattice dimension {integer_coefficients}'
        latt_vecs.append(np.dot(integer_coefficients, lattice))
        gc.collect()
    return np.asarray(latt_vecs)


def single_lattice_sample_clifford_alg(lattice, unitary, samples_per_sample):
    """
    Single lattice sample algorithm for efficiency purposes.
    :param lattice: lattice basis (ndarray)
    :param unitary: control unitary.
    :param samples_per_sample: number of boson samples needed per lattice vector.
    :return: single lattice vector.
    """
    dimension = lattice.shape[0]
    modes = 2*dimension+1
    near_identity = np.identity(modes)
    near_identity[0][0] = 0.
    boson_sample = np.delete(np.sum([np.sum([near_identity[i]
                                             for i in boson_sampler(unitary)],
                                            axis=0)
                                     for j in range(samples_per_sample)],
                                    axis=0),
                             0)
    split = np.split(boson_sample, 2)
    integer_coefficients = split[0] - split[1]
    return np.dot(integer_coefficients, lattice)


# if __name__ == "__main__":
#     latt = np.genfromtxt('Lattices/3/0/0.csv', delimiter=',', dtype=None)
#     # print(latt)
#     SV = np.genfromtxt('Lattices/3/0/4.csv', delimiter=',', dtype=None)
#     # print(SV)
#     # print(np.linalg.solve(latt.T, SV))
#     dimension = latt.shape[0]
#     modes = 2*dimension+1
#     U = unitary_group.rvs(15)
#     # print(f'unitary: {U}')
#     # print('------------------')
#     # print(clifford_lattice_sampler(latt, SV, 10, U))
#     print(lattice_sampler(latt, SV, 10, sim.boson_sampler_dist(unitary_group.rvs(modes))))
#
#     # a = np.array([[1, 2, 3],
#     #               [1, 2, 3],
#     #               [1, 2, 3]])
#     # print(unitary_test(a))
