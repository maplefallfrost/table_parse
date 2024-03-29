from typing import List

import numba
import numpy as np

from modules.bounding_box import BoundingBox


def transform_bboxes_to_points(
    bboxes: List[BoundingBox],
) -> List[List[tuple[int, int]]]:
    points_list = []
    for bbox in bboxes:
        min_row, max_row, min_col, max_col = (
            bbox.min_row,
            bbox.max_row,
            bbox.min_col,
            bbox.max_col,
        )
        # Define points in clockwise order starting from top-left
        points = [
            (min_col, min_row),  # Top-Left
            (max_col, min_row),  # Top-Right
            (max_col, max_row),  # Bottom-Right
            (min_col, max_row),  # Bottom-Left
        ]
        points_list.append(points)
    return points_list


@numba.jit(nopython=True)
def find_root(parent: np.ndarray, x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # Path compression
        x = parent[x]
    return x


@numba.jit(nopython=True)
def union(parent: int, rank: np.ndarray, x: int, y: int):
    root_x = find_root(parent, x)
    root_y = find_root(parent, y)
    if root_x != root_y:
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        else:
            parent[root_y] = root_x
            rank[root_x] += 1


@numba.jit(nopython=True)
def find_connected_components(matrix: np.ndarray) -> List[List[int]]:
    FOREGROUND = 0
    rows, cols = matrix.shape
    parent = np.arange(rows * cols)
    rank = np.zeros(rows * cols, dtype=np.int32)

    for row in range(rows):
        for col in range(cols):
            if matrix[row, col] == FOREGROUND:
                if col + 1 < cols and matrix[row, col + 1] == FOREGROUND:
                    union(parent, rank, row * cols + col, row * cols + col + 1)

                if row + 1 < rows and matrix[row + 1, col] == FOREGROUND:
                    union(parent, rank, row * cols + col, (row + 1) * cols + col)

                if (
                    row + 1 < rows
                    and col + 1 < cols
                    and matrix[row + 1, col + 1] == FOREGROUND
                ):
                    union(parent, rank, row * cols + col, (row + 1) * cols + col + 1)

    root_to_bbox = np.zeros((rows * cols, 4), dtype=numba.int32)
    root_to_bbox[:, 0] = np.iinfo(np.int32).max
    root_to_bbox[:, 1] = np.iinfo(np.int32).min
    root_to_bbox[:, 2] = np.iinfo(np.int32).max
    root_to_bbox[:, 3] = np.iinfo(np.int32).min

    for row in range(rows):
        for col in range(cols):
            if matrix[row, col] == FOREGROUND:
                # Find the root representative for the current pixel
                root = find_root(parent, row * cols + col)

                # Update the bounding box for this component
                min_row, max_row, min_col, max_col = root_to_bbox[root]
                root_to_bbox[root, 0] = min(min_row, row)
                root_to_bbox[root, 1] = max(max_row, row)
                root_to_bbox[root, 2] = min(min_col, col)
                root_to_bbox[root, 3] = max(max_col, col)

    final_bboxes = root_to_bbox[root_to_bbox[:, 0] != np.iinfo(np.int32).max]
    return final_bboxes


def get_text_bboxes(bboxes: List[BoundingBox]) -> List[BoundingBox]:
    areas = [bbox.area for bbox in bboxes]
    Q3 = np.percentile(areas, 75)
    text_bboxes = [bbox for bbox in bboxes if bbox.area < Q3 * 10]
    return text_bboxes
