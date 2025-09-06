import numpy as np
from raytracing.util import partition_cmp


def test_partition_cmp():
    # Case 1: Mixed values
    arr = np.array([5, 2, 8, 1, 7, 3])
    pivot = 5
    mid = partition_cmp(arr, pivot, 0, len(arr))
    print("Test 1:", arr, "mid:", mid)
    assert all(arr[i] < pivot for i in range(mid))
    assert all(arr[i] >= pivot for i in range(mid, len(arr)))

    # Case 2: All less than pivot
    arr = np.array([1, 2, 3])
    pivot = 5
    mid = partition_cmp(arr, pivot, 0, len(arr))
    print("Test 2:", arr, "mid:", mid)
    assert mid == len(arr)
    assert all(arr[i] < pivot for i in range(mid))

    # Case 3: All greater than or equal to pivot
    arr = np.array([5, 6, 7])
    pivot = 5
    mid = partition_cmp(arr, pivot, 0, len(arr))
    print("Test 3:", arr, "mid:", mid)
    assert mid == 0
    assert all(arr[i] >= pivot for i in range(mid, len(arr)))

    # Case 4: Some equal to pivot
    arr = np.array([3, 5, 2, 5, 1])
    pivot = 5
    mid = partition_cmp(arr, pivot, 0, len(arr))
    print("Test 4:", arr, "mid:", mid)
    assert all(arr[i] < pivot for i in range(mid))
    assert all(arr[i] >= pivot for i in range(mid, len(arr)))

    # Case 5: Empty array
    arr = np.array([])
    pivot = 5
    mid = partition_cmp(arr, pivot, 0, len(arr))
    print("Test 5:", arr, "mid:", mid)
    assert mid == 0

    # Case 6: Single element less than pivot
    arr = np.array([2])
    pivot = 5
    mid = partition_cmp(arr, pivot, 0, len(arr))
    print("Test 6:", arr, "mid:", mid)
    assert mid == 1
    assert arr[0] < pivot

    # Case 7: Single element greater than or equal to pivot
    arr = np.array([7])
    pivot = 5
    mid = partition_cmp(arr, pivot, 0, len(arr))
    print("Test 7:", arr, "mid:", mid)
    assert mid == 0
    assert arr[0] >= pivot

    print("All tests passed!")


if __name__ == "__main__":
    test_partition_cmp()
