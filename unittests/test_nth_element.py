import numpy as np
from raytracing.util import nth_element


def test_nth_element():
    # Test 1: Simple case
    arr = np.array([5, 2, 8, 1, 7, 3])
    n = 3
    pos = nth_element(arr, n, 0, len(arr))
    first_part = arr[:n]
    second_part = arr[n:]
    assert all(
        first_part[i] <= x for i in range(n) for x in second_part
    ), f"Test 1 failed: {arr}"

    # Test 2: All elements equal
    arr = np.array([4, 4, 4, 4, 4])
    n = 2
    pos = nth_element(arr, n, 0, len(arr))
    first_part = arr[:n]
    second_part = arr[n:]
    assert all(
        first_part[i] <= x for i in range(n) for x in second_part
    ), f"Test 2 failed: {arr}"

    # Test 3: Already sorted
    arr = np.array([1, 2, 3, 4, 5])
    n = 2
    pos = nth_element(arr, n, 0, len(arr))
    first_part = arr[:n]
    second_part = arr[n:]
    assert all(
        first_part[i] <= x for i in range(n) for x in second_part
    ), f"Test 3 failed: {arr}"

    # Test 4: Reverse sorted
    arr = np.array([5, 4, 3, 2, 1])
    n = 4
    pos = nth_element(arr, n, 0, len(arr))
    first_part = arr[:n]
    second_part = arr[n:]
    assert all(
        first_part[i] <= x for i in range(n) for x in second_part
    ), f"Test 4 failed: {arr}"

    # Test 5: n = 0
    arr = np.array([1, 2, 3])
    n = 0
    pos = nth_element(arr, n, 0, len(arr))
    assert pos == 0, f"Test 5 failed: {arr}"

    # Test 6: n = len(arr)
    arr = np.array([3, 1, 2])
    n = len(arr)
    pos = nth_element(arr, n, 0, len(arr))
    assert pos == len(arr), f"Test 6 failed: {arr}"

    print("All tests passed!")


if __name__ == "__main__":
    test_nth_element()
