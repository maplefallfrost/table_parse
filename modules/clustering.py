import heapq

import numpy as np
from rtree import index as rindex

from modules.bounding_box import BoundingBox

HORIZONTAL_DISTANCE_PERCENTILE = 60
RTREE_NEIGHBORS = 5


def compute_horizontal_distance_threshold(bboxes: list[BoundingBox]) -> float:
    sorted_bboxes = sorted(bboxes, key=lambda x: (x.center[0], x.center[1]))
    horizontal_distances = [
        sorted_bboxes[i + 1].min_col - sorted_bboxes[i].max_col
        for i in range(len(sorted_bboxes) - 1)
    ]
    horizontal_distances = [dist for dist in horizontal_distances if dist > 0]
    threshold = np.percentile(horizontal_distances, HORIZONTAL_DISTANCE_PERCENTILE)
    return threshold


def can_merge_by_projection(
    cluster: BoundingBox, bbox: BoundingBox, projection: np.ndarray
) -> bool:
    min_row = min(cluster.min_row, bbox.min_row)
    max_row = max(cluster.max_row, bbox.max_row)
    segment_projection = projection[min_row:max_row]
    return np.all(segment_projection > 0)


def can_merge_by_horizontal_distance(
    cluster: BoundingBox, bbox: BoundingBox, threshold: float
) -> bool:
    return (
        cluster.min_col - bbox.max_col < threshold
        and bbox.min_col - cluster.max_col < threshold
    )


def update_distances(
    heap: list[tuple[float, int, int]],
    bboxes: list[BoundingBox],
    i: int,
    j: int,
    rtree_index: rindex.Index,
):
    rtree_index.delete(i, bboxes[i].bounds)
    rtree_index.delete(j, bboxes[j].bounds)
    bboxes[i].merge(bboxes[j])

    bboxes.append(bboxes[i].copy())
    new_bbox_index = len(bboxes) - 1
    rtree_index.insert(new_bbox_index, bboxes[new_bbox_index].bounds)
    bboxes[i] = bboxes[j] = None

    nearby_indices = list(
        rtree_index.nearest(bboxes[new_bbox_index].bounds, RTREE_NEIGHBORS + 1)
    )
    for nearby_index in nearby_indices:
        if nearby_index != new_bbox_index:
            dist = bboxes[new_bbox_index].min_distance(bboxes[nearby_index])
            heapq.heappush(heap, (dist, new_bbox_index, nearby_index))


def clustering(bboxes: list[BoundingBox], image_height: int) -> list[BoundingBox]:
    # Use in can_merge_by_projection
    sorted_bboxes = sorted(bboxes, key=lambda x: (x.center[0], x.center[1]))
    projection = np.zeros(image_height)
    for bbox in sorted_bboxes:
        projection[bbox.min_row : bbox.max_row] += 1

    # Use in can_merge_by_horizontal_distance
    horizontal_distance_threshold = compute_horizontal_distance_threshold(sorted_bboxes)

    # Use rtree to speed up the process of updating distances
    rtree_index = rindex.Index()
    for i, bbox in enumerate(bboxes):
        rtree_index.insert(i, bbox.bounds)

    heap = []
    for i in range(len(bboxes)):
        nearby_indices = list(
            rtree_index.nearest(bboxes[i].bounds, RTREE_NEIGHBORS + 1)
        )
        for j in nearby_indices:
            dist = bboxes[i].min_distance(bboxes[j])
            heapq.heappush(heap, (dist, i, j))

    while heap:
        dist, i, j = heapq.heappop(heap)
        if bboxes[i] is not None and bboxes[j] is not None:
            can_merge_flags = [
                can_merge_by_projection(bboxes[i], bboxes[j], projection),
                can_merge_by_horizontal_distance(
                    bboxes[i], bboxes[j], horizontal_distance_threshold
                ),
            ]
            if np.all(can_merge_flags):
                update_distances(heap, bboxes, i, j, rtree_index)

    return [bbox for bbox in bboxes if bbox is not None]
