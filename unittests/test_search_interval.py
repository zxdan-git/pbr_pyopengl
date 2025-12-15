import numpy as np
from raytracing.util import search_interval


def test_search_interval():
    # Case 1: Basic case - pivot in the middle
    arr = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    result = search_interval(arr, 0.3, 0, len(arr) - 1)
    print("Test 1: pivot=0.3 in", arr, "-> index:", result)
    assert result == 1  # 0.2 <= 0.3 < 0.5

    # Case 2: Pivot at exact boundary
    arr = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    result = search_interval(arr, 0.5, 0, len(arr) - 1)
    print("Test 2: pivot=0.5 in", arr, "-> index:", result)
    assert result == 2  # 0.5 <= 0.5 < 0.8

    # Case 3: Pivot in first interval
    arr = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    result = search_interval(arr, 0.1, 0, len(arr) - 1)
    print("Test 3: pivot=0.1 in", arr, "-> index:", result)
    assert result == 0  # 0.0 <= 0.1 < 0.2

    # Case 4: Pivot in last interval
    arr = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    result = search_interval(arr, 0.9, 0, len(arr) - 1)
    print("Test 4: pivot=0.9 in", arr, "-> index:", result)
    assert result == 3  # 0.8 <= 0.9 < 1.0

    # Case 5: Pivot equals first value
    arr = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    result = search_interval(arr, 0.0, 0, len(arr) - 1)
    print("Test 5: pivot=0.0 in", arr, "-> index:", result)
    assert result == 0  # 0.0 <= 0.0 < 0.2

    # Case 6: Pivot equals last value (edge case)
    arr = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    result = search_interval(arr, 1.0, 0, len(arr) - 1)
    print("Test 6: pivot=1.0 in", arr, "-> index:", result)
    # This should return -1 or the last interval depending on requirements
    # With current implementation, 1.0 >= arr[4] so returns -1
    print("  Note: pivot at upper bound")

    # Case 7: Pivot below range
    arr = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    result = search_interval(arr, -0.1, 0, len(arr) - 1)
    print("Test 7: pivot=-0.1 in", arr, "-> index:", result)
    assert result == -1  # Out of range

    # Case 8: Pivot above range
    arr = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    result = search_interval(arr, 1.5, 0, len(arr) - 1)
    print("Test 8: pivot=1.5 in", arr, "-> index:", result)
    assert result == -1  # Out of range

    # Case 9: Search in subrange
    arr = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    result = search_interval(arr, 0.35, 2, 7)  # Search in [0.2, 0.7]
    print("Test 9: pivot=0.35 in subrange [2:7]", arr[2:8], "-> index:", result)
    assert result == 3  # 0.3 <= 0.35 < 0.4

    # Case 10: CDF-like array (monotonically increasing)
    arr = np.array([0.0, 0.15, 0.35, 0.60, 0.85, 1.0])
    result = search_interval(arr, 0.5, 0, len(arr) - 1)
    print("Test 10: pivot=0.5 in CDF", arr, "-> index:", result)
    assert result == 2  # 0.35 <= 0.5 < 0.60

    # Case 11: Very small array
    arr = np.array([0.0, 1.0])
    result = search_interval(arr, 0.5, 0, len(arr) - 1)
    print("Test 11: pivot=0.5 in", arr, "-> index:", result)
    assert result == 0  # 0.0 <= 0.5 < 1.0

    # Case 12: Array with value_func
    arr = np.array([{"value": 0.0}, {"value": 0.3}, {"value": 0.6}, {"value": 1.0}])
    result = search_interval(arr, 0.4, 0, len(arr) - 1, value_func=lambda x: x["value"])
    print("Test 12: pivot=0.4 with value_func -> index:", result)
    assert result == 1  # 0.3 <= 0.4 < 0.6

    # Case 13: Uniform distribution
    arr = np.linspace(0, 1, 11)  # [0.0, 0.1, 0.2, ..., 1.0]
    result = search_interval(arr, 0.55, 0, len(arr) - 1)
    print("Test 13: pivot=0.55 in uniform", arr, "-> index:", result)
    assert result == 5  # 0.5 <= 0.55 < 0.6

    # Case 14: Nearly equal pivot to boundary
    arr = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    result = search_interval(arr, 0.2000001, 0, len(arr) - 1)
    print("Test 14: pivot=0.2000001 in", arr, "-> index:", result)
    assert result == 1  # 0.2 <= 0.2000001 < 0.5

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_search_interval()
