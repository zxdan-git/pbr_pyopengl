import numpy as np
from typing import List

from .build_node import BuildNode
from ..bounding_box import AABB
from .morton_code_util import (
    MortonObject,
    generate_morton_objects_from_intersect_objects,
)
from .partition_util import sah_partition_on_target_aixs
from ..ray_intersect_object import RayIntersectObject
from .recursive_build import build_by_axis_spread, build_by_sorted_morton
from ..util import radix_sort_binary


def build_with_morton_code(objects: List[RayIntersectObject]):
    morton_objects = generate_morton_objects_from_intersect_objects(objects)

    bits_per_dim = 10
    total_bits = 3 * bits_per_dim
    n_objects = len(morton_objects)
    radix_sort_binary(
        morton_objects,
        total_bits=total_bits,
        bits_per_pass=6,
        start=0,
        end=n_objects,
        value_func=lambda morton_object: morton_object.morton,
    )

    subtree_roots = []
    start, end = 0, 1
    subtree_bits = int(np.min([12, total_bits]))
    subtree_mask = ((1 << subtree_bits) - 1) << (total_bits - subtree_bits)

    while end != n_objects + 1:
        if end == n_objects or (morton_objects[start].morton & subtree_mask) != (
            morton_objects[end].morton & subtree_mask
        ):
            subtree_root = build_by_sorted_morton(
                morton_objects, start, end, total_bits - subtree_bits - 1
            )
            if not subtree_root is None:
                subtree_roots.append(subtree_root)
            start = end
        end += 1

    return build_by_axis_spread(
        subtree_roots, 0, len(subtree_roots), sah_partition_on_target_aixs
    )
