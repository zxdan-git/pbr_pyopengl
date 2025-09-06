from typing import List

from ..bounding_box import AABB
from ..ray_intersect_object import RayIntersectObject


class BucketInfo:
    def __init__(self):
        self.objects: List[RayIntersectObject] = []
        self.total_cost = 0
        self.bbx = AABB()


def surface_area_heuristic(left: BucketInfo, right: BucketInfo):
    total_bbx = AABB.union(left.bbx, right.bbx)
    if total_bbx.empty():
        return 0

    if len(left.objects) == 0 or left.bbx.empty():
        return right.total_cost

    if len(right.objects) == 0 or right.bbx.empty():
        return left.total_cost

    # cost = traversal cost + p_left * cost_left + p_right * cost_right.
    # Here, traversal cost = 1/8, p_left = left bbx area / total bbx area,
    # p_right = right bbx area / total bbx area
    return (
        0.125
        + (
            left.bbx.surface_area() * left.total_cost
            + right.bbx.surface_area() * right.total_cost
        )
        / total_bbx.surface_area()
    )


def min_sah_partition(buckets: List[BucketInfo]) -> tuple[BucketInfo, BucketInfo]:
    n_bucket = len(buckets)
    # Get all partition cases.
    # left_groups[i] is a group contains the bucket from [0, i].
    # right_groups[i] is a group contains the bucket from [i + 1, n_bucket).
    left_groups = [BucketInfo() for _ in range(n_bucket)]
    right_groups = [BucketInfo() for _ in range(n_bucket)]
    left_groups[0] = buckets[0]
    for i in range(1, n_bucket):
        left_groups[i].objects = buckets[i].objects + left_groups[i - 1].objects
        left_groups[i].total_cost = (
            buckets[i].total_cost + left_groups[i - 1].total_cost
        )
        left_groups[i].bbx = AABB.union(buckets[i].bbx, left_groups[i - 1].bbx)

        right_groups[-i - 1].objects = buckets[-i].objects + right_groups[-i].objects
        right_groups[-i - 1].total_cost = (
            buckets[-i].total_cost + right_groups[-i].total_cost
        )
        right_groups[-i - 1].bbx = AABB.union(buckets[-i].bbx, right_groups[-i].bbx)

    # Find the partition with the lowest surface area heuristic.
    return min(
        [(left, right) for left, right in zip(left_groups, right_groups)],
        key=lambda partition: surface_area_heuristic(partition[0], partition[1]),
    )
