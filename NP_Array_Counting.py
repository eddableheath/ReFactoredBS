# Counting array
# Author: Edmund Dable-Heath
# When a count is needed a count is needed.

import numpy as np


def np_array_equals(array1, array2):
    """Equals implementation"""
    array_type = type(np.array(1))
    assert array_type == type(array1)
    assert array_type == type(array2)
    return (array1 == array2).all()


def count(list_of_arrays, target):
    """Return the number of arrays in supplied list that match a supplied target array."""
    assert type([np.array(1)]) == type(list_of_arrays)
    assert type(target) in [type(e) for e in list_of_arrays]
    return len([array for array in list_of_arrays if np_array_equals(array, target)])


def get_unique_arrays(list_of_arrays):
    found_unique_arrays = []
    for candidate_array in list_of_arrays:
        if (not found_unique_arrays) or (
        not any([np_array_equals(candidate_array, found) for found in found_unique_arrays])):
            found_unique_arrays.append(candidate_array)
    return found_unique_arrays


def get_counts(list_of_arrays):
    uniques = get_unique_arrays(list_of_arrays)
    for unique in uniques:
        yield unique, count(list_of_arrays, unique)