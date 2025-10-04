import numpy as np


def normalize(v):
    return v / np.linalg.norm(v)


def det3x3(mat3x3):
    return (
        mat3x3[0, 0] * mat3x3[1, 1] * mat3x3[2, 2]
        - mat3x3[0, 0] * mat3x3[1, 2] * mat3x3[2, 1]
        + mat3x3[0, 1] * mat3x3[1, 2] * mat3x3[2, 0]
        - mat3x3[0, 1] * mat3x3[1, 0] * mat3x3[2, 2]
        + mat3x3[0, 2] * mat3x3[1, 0] * mat3x3[2, 1]
        - mat3x3[0, 2] * mat3x3[1, 1] * mat3x3[2, 0]
    )


def partition(array, start, end, predict):
    """
    Reorder the array in range [start, end) into two parts and return a
    partition position mid. The predict is a lambda functor which takes the
    array item and returns a boolean value. The values in reordered part in
    [start, mid) will satisfy the predict, and the values in [mid, end) will
    violate the predict.
    """
    i, j = start, end
    while i < j:
        while i < j and predict(array[i]):
            i += 1
        while i < j and not predict(array[j - 1]):
            j -= 1
        if i < j:
            array[i], array[j - 1] = array[j - 1], array[i]
    return i


def partition_cmp(array, pivot, start, end, value_func=lambda value: value):
    """
    Reorder the array in range [start, end) into two parts and return a
    partition position mid. The values in reordered part in [start, mid)
    will be smaller than pivot, and the values in [mid, end) will be
    large than or equal to the pivot.
    """
    return partition(
        array, start, end, predict=lambda array_i: value_func(array_i) < pivot
    )


def nth_element(array, n, start, end, value_func=lambda value: value):
    """
    Split the array into two parts such that the elements in the first part in
    range [start, start + n) are no greater than the elements in the second part
    in range [start + n, end), return the position start + n.
    """
    # Directly return if position n is at the start and end.
    if n == 0 or n == end - start:
        return start + n

    values = [value_func(array[i]) for i in range(start, end)]
    min_value = np.min(values)
    max_value = np.max(values)
    # Directly return if all elements are equal.
    if np.isclose(min_value, max_value):
        return start + n

    pivot = (min_value + max_value) / 2
    pos = partition_cmp(array, pivot, start, end, value_func)
    if pos < start + n:
        return nth_element(array, start + n - pos, pos, end, value_func)
    if pos > start + n:
        return nth_element(array, n, start, pos, value_func)
    return pos


def radix_sort_binary(
    array, total_bits, bits_per_pass, start, end, value_func=lambda value: value
):
    n_pass = int(np.ceil(total_bits / bits_per_pass))
    n_buckets = int(np.power(2, bits_per_pass))
    mask = (1 << bits_per_pass) - 1
    for pass_i in range(n_pass):
        bucket_sizes = [0] * n_buckets
        values = [value_func(array[i]) for i in range(start, end)]
        for value in values:
            bucket_sizes[(value >> (pass_i * bits_per_pass)) & mask] += 1

        bucket_start = [start] * n_buckets
        for i in range(1, n_buckets):
            bucket_start[i] = bucket_start[i - 1] + bucket_sizes[i - 1]

        array_copy = [array[i] for i in range(start, end)]
        for idx, value in enumerate(values):
            bucket_id = (value >> (pass_i * bits_per_pass)) & mask
            array[bucket_start[bucket_id]] = array_copy[idx]
            bucket_start[bucket_id] += 1
