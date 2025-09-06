import numpy as np
from raytracing.bvh_util.morton_code_util import vec3_to_morton


def test_vec3_to_morton():
    # Test 1: All zeros
    v = np.array([0, 0, 0], dtype=np.uint32)
    code = vec3_to_morton(v)
    print(f"Morton code for {v}: {code:#010x}")
    assert code == 0, f"Test 1 failed: {code} != 0"

    # Test 2: All ones
    v = np.array([1, 1, 1], dtype=np.uint32)
    code = vec3_to_morton(v)
    print(f"Morton code for {v}: {code:#010x}")
    assert code == 7, f"Test 2 failed: {code} != 7"

    # Test 3: Max values (1023, 1023, 1023)
    v = np.array([1023, 1023, 1023], dtype=np.uint32)
    code = vec3_to_morton(v)
    print(f"Morton code for {v}: {code:#010x}")
    assert code == 0x3FFFFFFF, f"Test 3 failed: {code:#010x} != 0x3fffffff"

    # Test 4: Different values
    v = np.array([5, 10, 20], dtype=np.uint32)
    code = vec3_to_morton(v)
    print(f"Morton code for {v}: {code:#010x}")
    assert code == 0x00004551, f"Test 4 failed: {code} != 0x00004551"

    # Test 5: Only one axis set
    v = np.array([1023, 0, 0], dtype=np.uint32)
    code = vec3_to_morton(v)
    print(f"Morton code for {v}: {code:#010x}")
    assert code == 0x9249249, f"Test 5 failed: {code:#010x} != 0x9249249"

    v = np.array([0, 1023, 0], dtype=np.uint32)
    code = vec3_to_morton(v)
    print(f"Morton code for {v}: {code:#010x}")
    assert code == 0x12492492, f"Test 5 failed: {code:#010x} != 0x12492492"

    v = np.array([0, 0, 1023], dtype=np.uint32)
    code = vec3_to_morton(v)
    print(f"Morton code for {v}: {code:#010x}")
    assert code == 0x24924924, f"Test 5 failed: {code:#010x} != 0x24924924"

    print("All test cases executed.")


if __name__ == "__main__":
    test_vec3_to_morton()
