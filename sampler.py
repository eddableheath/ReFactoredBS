# Implementation of the Klein Sampler
# Author: Edmund Dable-Heath
# Using Thomas Prest's algorithm

import numpy as np
import math

# Defaul rng for numpy
rng = np.random.default_rng()


def one_d_int_gauss(centre, standard_deviation, security_parameter):
    """
    One dimensional discrete gaussian sampler over the integers.
    :param centre: centre of the distribution
    :param standard_deviation: standard deviation of the distribution
    :param security_parameter: security parameter for computing range of support.
    :return: an integer sample from the relevant distribution
    """
    while(1):
        uniform_pick = rng.integers(centre-(standard_deviation*math.log2(security_parameter)),
                                    centre+(standard_deviation*math.log2(security_parameter)))
        rejection_bound = (1/(standard_deviation*math.sqrt(2*math.pi)))*np.exp(-0.5*(((uniform_pick-centre)/standard_deviation)**2))
        rejection_test = rng.uniform(0, 1)
        if rejection_test <= rejection_bound:
            return uniform_pick
        else:
            continue


def klein_sampler(lattice_basis,  security_parameter, standard_deviation=1, target=0):
    """
    Klein Sampler Algorithm for sampling from a discrete gaussian over a lattice.
    :param lattice_basis: lattice basis
    :param security_parameter: security parameter for rejection sampling step
    :param standard_deviation: standard deviation of the gaussian
    :param target: centre of the distribution, assumed to be origin if not specified.
    :return: sample from discrete gaussian over the lattice.
    """
    dimension = lattice_basis.shape[0]
    gso_basis, _ = np.linalg.qr(lattice_basis)
    gso_basis = gso_basis.T
    if target==0:
        target = np.zeros(dimension)
    update_vector = np.zeros(dimension)
    for i in range(dimension, 0, -1):
        i -= 1
        d_val = np.dot(target, gso_basis[i]) / np.linalg.norm(gso_basis[i])**2
        # print(f'new centre: {d_val}')
        sigma_val = standard_deviation / np.linalg.norm(gso_basis[i])
        z_val = one_d_int_gauss(d_val, sigma_val, security_parameter)
        # print(f'integer value: {z_val}')
        target -= z_val*lattice_basis[i]
        # print(f'target: {target}')
        update_vector += z_val*lattice_basis[i]
        # print(f'vector: {update_vector}')
    return update_vector


if __name__=="__main__":
    latt_basis = np.array([[32, 0],
                           [9, 1]])
    for i in range(20):
        print(klein_sampler(latt_basis, 64, standard_deviation=5))