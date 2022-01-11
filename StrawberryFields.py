# Using Strawberry Fields
# Author: Edmund Dable-Heath
# Using the Gaussian Boson Sampling from Strawberry Fields to test how the system works for straight up using the Gramm
# matrix of a basis as the basic convariance matrix to start with. The maybe implement a system to train it on?

from strawberryfields.apps import sample
import numpy as np
import statistics as stats


def lattice_from_bosons(extended_basis, boson_sample):
    """
    Creating a lattice vector from a set of boson samples
    :param extended_basis: extended basis: [B : -B]
    :param boson_sample: boson sample, list of lists which contain the output of the boson sampler
    :return: lattice vector
    """
    return np.sum([np.dot(extended_basis.T, np.asarray(bosons)) for bosons in boson_sample], axis=0)


def gbs_lattice_sampler(basis, unitary, no_samples, gbs_samples_per_sample):
    """
    Using the GBS sampler to sample from the lattice, using the gram matrix of the lattice at the covariance matrix.
    :param basis: lattice basis
    :param unitary: control unitary
    :param no_samples: number of lattice samples
    :param gbs_samples_per_sample: number of samples from the GBS needed for each sample.
    :return: lattice vectors
    """
    modes = unitary.shape[0]
    extended_basis = np.append(basis, -basis, axis=0)
    return [
        lattice_from_bosons(extended_basis, sample.sample(unitary, 2*modes, gbs_samples_per_sample, threshold=False))
        for i in range(no_samples)
    ]


if __name__ == "__main__":
    lattice_basis = np.array([[256, 0],
                              [33, 1]])
    results = gbs_lattice_sampler(lattice_basis, 100, 20)
    norms = [np.linalg.norm(vec) for vec in results]
    print(stats.mean(norms))
