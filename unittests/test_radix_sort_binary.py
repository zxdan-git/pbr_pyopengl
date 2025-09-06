from raytracing.util import radix_sort_binary


def test_radix_sort_binary():
    # Test 1: Sort entire array of small positive integers
    arr = [5, 3, 8, 1, 7, 2, 4, 6]
    radix_sort_binary(arr, total_bits=4, bits_per_pass=2, start=0, end=len(arr))
    assert arr == sorted(arr), f"Test 1 failed: {arr}"

    # Test 2: Sort a subarray
    arr = [10, 5, 3, 8, 1, 7, 2, 4, 6, 9]
    radix_sort_binary(arr, total_bits=4, bits_per_pass=2, start=1, end=9)
    assert arr[1:9] == sorted(arr[1:9]), f"Test 2 failed: {arr}"

    # Test 3: Including zero and max value for 4 bits
    arr = [15, 0, 7, 8, 3, 12, 4, 1]
    radix_sort_binary(arr, total_bits=4, bits_per_pass=2, start=0, end=len(arr))
    assert arr == sorted(arr), f"Test 3 failed: {arr}"

    # Test 4: 8-bit values
    arr = [255, 128, 64, 32, 16, 8, 4, 2, 1, 0]
    radix_sort_binary(arr, total_bits=8, bits_per_pass=4, start=0, end=len(arr))
    assert arr == sorted(arr), f"Test 4 failed: {arr}"

    # Test 5: Already sorted
    arr = [1, 2, 3, 4, 5, 6, 7, 8]
    radix_sort_binary(arr, total_bits=4, bits_per_pass=2, start=0, end=len(arr))
    assert arr == sorted(arr), f"Test 5 failed: {arr}"

    # Test 6: Reverse sorted
    arr = [8, 7, 6, 5, 4, 3, 2, 1]
    radix_sort_binary(arr, total_bits=4, bits_per_pass=2, start=0, end=len(arr))
    assert arr == sorted(arr), f"Test 6 failed: {arr}"

    # Test 7: Duplicates
    arr = [2, 3, 2, 3, 1, 1, 4, 4]
    radix_sort_binary(arr, total_bits=3, bits_per_pass=1, start=0, end=len(arr))
    assert arr == sorted(arr), f"Test 7 failed: {arr}"

    # Test 8: Custom value_func (sort by negative value)
    arr = [5, 3, 8, 1, 7, 2, 4, 6]
    radix_sort_binary(
        arr,
        total_bits=4,
        bits_per_pass=2,
        start=0,
        end=len(arr),
        value_func=lambda x: -x,
    )
    assert arr == sorted(arr, reverse=True), f"Test 8 failed: {arr}"

    print("All radix_sort_binary tests passed.")


if __name__ == "__main__":
    test_radix_sort_binary()
