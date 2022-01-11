# Constructing Unitaries from the Ising Problem
# Author: Edmund Dable-Heath
# Constructing the control unitary for the Boson Sampler from the Gram matrix and exponentiation of said matrix.

import numpy as np
from scipy.linalg import expm


def gram_unitary(lattice_basis, zero=False, extended=False, theta=1):
    """
    Computing a unitary matrix from the gram matrix to test the implementation.
    :param lattice_basis: square matrix representing lattice basis.
    :param zero: optional addition of zero mode (arbitrarily put in position zero)
    :param extended: optional inclusion of negative basis elements
    :param theta: optimisation parameter
    :return: unitary matrix which is exponentiated matrix form of Gram matrix of extended Gram matrix
    """
    basis = lattice_basis
    if zero:
        basis = np.delete(np.append(lattice_basis, np.zeros((2, lattice_basis.shape[0])), axis=0),
                          -1, axis=0)
    if extended:
        basis = np.append(basis, -lattice_basis, axis=0)
    return expm(1j * np.matmul(basis, basis.T) * theta)


def qr_unitary(lattice_basis, zero=False, extended=False):
    basis = lattice_basis
    if zero:
        basis = np.delete(np.append(lattice_basis, np.zeros((2, lattice_basis.shape[0])), axis=0),
                          -1, axis=0)
    if extended:
        basis = np.append(basis, -lattice_basis, axis=0)
    q, r = np.linalg.qr(np.matmul(basis, basis.T))
    return q


if __name__ == "__main__":

    L = np.array([[256, 0],
                  [33, -1]])

    print(gram_unitary(L, extended=True, zero=True))
    print(qr_unitary(L, extended=True, zero=True))
