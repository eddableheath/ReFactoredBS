# Laplace Expansion of Permanent Computation
# Author: Edmund Dable-Heath
# Implementation of the Laplace expansion of the computation of a permanent of a matrix. The last time I tried to do
# this I don't think I implemented it in the correct way, like the paper suggests.

from functools import reduce
from operator import mul
from math import floor
import numpy as np
# import logging
from scipy.stats import unitary_group


def cmp(a, b):
    return int(a > b) - int(a < b)


def fast_glynn_perm(matrix):
    """
    Got this from Stack exchange, appears to work, should test it some more.
    :param matrix: unitary matrix
    :return: permanent
    """
    assert matrix.shape[0] == matrix.shape[1], f'Non-sqare matrix permanent can not be done {matrix}'
    row_comb = [sum(c) for c in zip(*matrix)]
    n = len(matrix)

    total = 0
    old_gray = 0
    sign = +1

    binary_power_dict = {2**i: i for i in range(n)}
    num_loops = 2**(n-1)

    for bin_index in range(1, num_loops+1):
        total += sign * reduce(mul, row_comb)

        new_gray = bin_index ^ floor(bin_index/2)
        gray_diff = old_gray ^ new_gray
        gray_diff_index = binary_power_dict[gray_diff]

        new_vector = matrix[gray_diff_index]
        direction = 2 * cmp(old_gray, new_gray)

        for i in range(n):
            row_comb[i] += new_vector[i] * direction

        sign = -sign
        old_gray = new_gray

    return total/num_loops


def collect_submatrix_perms(matrix):
    """
    Collection all the submatrix perms for a n * n+1 matrix by removing a row each time.
    NB: According to algorithm this can be sped up using clever trick with the laplace expansion.
    :param matrix: n * n-1 submatrix of a unitary matrix
    :return: list of permanents
    """
    return np.asarray([fast_glynn_perm(np.delete(matrix, i, axis=1)) for i in range(matrix.shape[1])])


# @profile
def laplace_expansion(permanents_matrix, full_matrix):
    """
    Coefficients from full matrix.
    :param permanents_matrix: matrix for which to collect the perms of the submatrices from.
    :param full_matrix: initial unitary matrix
    :return: output distribution
    """
    laplace_perms = collect_submatrix_perms(permanents_matrix)
    return np.abs(np.asarray([np.dot(row, laplace_perms) for row in full_matrix]))**2


if __name__ == "__main__":
    A = unitary_group.rvs(4)
    print(f'Unitary matrix: {A}')
    New_A = A[:, :2]
    print(f'Relevant columns of unitary matrix {New_A}')
    rand_index_1 = 2
    print(f'Randomly selected index {rand_index_1}')
    reduced_for_perms_A = New_A[rand_index_1, :2]
    print(f'Reduced matrix for permanent computation {reduced_for_perms_A}')
    l_expansion = laplace_expansion(np.array([reduced_for_perms_A]), New_A)
    print(f'Laplace expansion {l_expansion}')
    print(f'Sum of laplace expansion {np.sum(l_expansion)}')
    normalised_l_expansion = l_expansion/np.sum(l_expansion)
    print(f'Normalised laplace expansion {normalised_l_expansion}')
    print(f'sum of normalised laplace expansion {np.sum(normalised_l_expansion)}')
