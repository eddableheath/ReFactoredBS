# Simple Boson Distribution Machine
# Author: Edmund Dable-Heath
# This code will take an input of a random unitary matrix and output the boson sample distribution over the out puts of
# a machine of corresponding size. For this we will assume that the number of modes is polynomial in the number of
# photons, i.e. n = floor(sqrt(m)). We also assume (wlog) that the photons are all inputted in the first n input modes.

import numpy as np
import logging
from scipy.stats import unitary_group
import math
import copy
import time
import random
import os
import multiprocessing as mp
from pathos.multiprocessing import ProcessPool as pool
import Unitary_Construction as uc


def permanent(matrix):
    """
    Implementation of Glynn's formula for a single matrix.
    :param matrix: Input matrix (numpy array)
    :return: Permanent (float)
    """
    n = matrix.shape[0]
    d = np.ones(n)
    j = 0
    s = 1
    f = np.arange(n)
    v = matrix.sum(axis=0)
    p = np.prod(v)
    while (j < n-1):
        v = v - 2*d[j]*matrix[j]
        d[j] = -d[j]
        s = -s
        prod = np.prod(v)
        p += s*prod
        f[0] = 0
        f[j] = f[j+1]
        f[j+1] = j+1
        j = f[0]
    return p/2**(n-1)


def bucket_enumerator_generator(photon_number, mode_number, previous_set=None, counter=0):
    """
    Recursive function that enumerates all output of a repeated combinatorics set.
    :param photon_number: Number of to be put in a bucket (int)
    :param mode_number: Number of buckets (int)
    :param previous_set: previous set (for recursion) (list of lists)
    :param counter: counter (for recursion) (int)
    :return: list of lists of possible outputs
    """
    if counter == photon_number:
        return previous_set
    else:
        if counter == 0:
            new_set = [[i] for i in range(mode_number)]
        else:
            new_set = []
            for i in range(mode_number):
                for j in previous_set:
                    c = copy.copy(j)
                    if all(number >= i for number in j):
                        c.append(i)
                        new_set.append(c)
        return bucket_enumerator_generator(photon_number, mode_number, new_set, counter+1)


def bucket_enumerator_retrieval(photon_number, mode_number):
    """
    Recursion was causing stack problems, so we're gonna put it in a depository when computed.
    :param photon_number: number of photons (int)
    :param mode_number: number of modes (int)
    :return: buckets (numpy array)
    """
    # Bucket search
    def find_buckets(m):
        return bool(any(str(m) in entry for entry in os.listdir('Buckets/')))
    if find_buckets(mode_number):
        # print(f'Already got it! Photons: {photon_number} Modes: {mode_number}')
        return np.genfromtxt('Buckets/' + str(mode_number) + '.csv',
                             delimiter=',')
    else:
        logging.warning(f'Could not find bucket for {mode_number}')
        new_bucket = np.asarray(bucket_enumerator_generator(photon_number, mode_number))
        np.savetxt('Buckets/' + str(mode_number) + '.csv',
                   new_bucket,
                   delimiter=',')
        return new_bucket


def submatrix_gen(input_modes, output_modes, unitary):
    """
    From unitary matrix picks relevant submatrix for boson sampler.
    :param input_modes: List of input modes with photons, can be repeated.
    :param output_modes: List of output modes with photons, can be repeated.
    :param unitary: Control unitary.
    :return: Submatrix for relevant input-->output photon transition
    """
    # input_modes.sort()
    output_modes.sort()
    return np.asarray([np.asarray([unitary.T[i]
                                   for i in input_modes]).T[j]
                       for j in output_modes])


def boson_sampler_dist(unitary):
    """
    Naive boson sampler.
    :param unitary: Control Unitary (optional, random if not given) (ndarray)
    :return: List of tuples with (output modes, weight)
    """
    mode_number = unitary.shape[0]
    photon_number = math.floor(math.sqrt(mode_number))
    photon_position = np.arange(photon_number).tolist()
    output_modes = bucket_enumerator_retrieval(photon_number, mode_number).astype(int)
    distribution = [(output, abs(permanent(submatrix_gen(photon_position, output, unitary)))**2)
                    for output in output_modes]
    weight = sum([j[1] for j in distribution])
    return [(j[0], j[1]/weight) for j in distribution]


def dist_gen(mode_number):
    """
    For simplicity of multiprocessing when testing.
    :param mode_number: input/output modes (int)
    :return: boson_dist
    """
    return boson_sampler_dist(unitary_group.rvs(mode_number))


# Tests
if __name__ == "__main__":
    L = np.array([[256, 0],
                  [33, 1]])
    picker5 = np.array([[1, 2, 3, 4, 5],
                        [2, 6, 7, 8, 9],
                        [3, 7, 10, 11, 12],
                        [4, 8, 11, 13, 14],
                        [5, 9, 12, 14, 15]])
    picker4 = np.array([[1, 2, 3, 4],
                         [2, 5, 6, 7],
                         [3, 6, 8, 9],
                         [4, 7, 9, 10]])
    for i in range(10000):
        output_dist = boson_sampler_dist(uc.gram_unitary(L, extended=True, theta=2*i*math.pi/10000))
        x_axis = [picker4[j[0][0], j[0][1]] for j in output_dist]
        y_axis = [output[1] for output in output_dist]
        print(output_dist)
        plt.bar(x_axis, y_axis)
        plt.show()