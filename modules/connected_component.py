from typing import List, Tuple

import numpy as np


class DisjointSet:
    def __init__(self, size):
        self.parent = [i for i in range(size)]

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x


def find_connected_components(
    matrix: np.ndarray,
) -> Tuple[List[List[int]], List[List[int]]]:
    FOREGROUND = 0
    rows, cols = matrix.shape
    disjoint_set = DisjointSet(rows * cols)

    for row in range(rows):
        for col in range(cols):
            if matrix[row, col] == FOREGROUND:
                if col + 1 < cols and matrix[row, col + 1] == FOREGROUND:
                    disjoint_set.union(row * cols + col, row * cols + col + 1)

                if row + 1 < rows and matrix[row + 1, col] == FOREGROUND:
                    disjoint_set.union(row * cols + col, (row + 1) * cols + col)

                if (
                    row + 1 < rows
                    and col + 1 < cols
                    and matrix[row + 1, col + 1] == FOREGROUND
                ):
                    disjoint_set.union(row * cols + col, (row + 1) * cols + col + 1)

    root_to_pixels = {}
    root_to_bbox = {}

    for row in range(rows):
        for col in range(cols):
            if matrix[row, col] == FOREGROUND:
                # Find the root representative for the current pixel
                root = disjoint_set.find(row * cols + col)

                # Initialize the lists and bounding boxes if this root is encountered for the first time
                if root not in root_to_pixels:
                    root_to_pixels[root] = []
                    root_to_bbox[root] = [
                        rows,
                        0,
                        cols,
                        0,
                    ]  # min_row, max_row, min_col, max_col

                # Append the current pixel to the component list
                root_to_pixels[root].append((row, col))

                # Update the bounding box for this component
                min_row, max_row, min_col, max_col = root_to_bbox[root]
                root_to_bbox[root] = [
                    min(min_row, row),
                    max(max_row, row),
                    min(min_col, col),
                    max(max_col, col),
                ]

    # Prepare the final lists ensuring matched order
    component_pixels_list = []
    bbox_list = []
    for root in root_to_pixels:
        component_pixels_list.append(root_to_pixels[root])
        min_row, max_row, min_col, max_col = root_to_bbox[root]
        bbox_list.append(
            [
                (min_col, min_row),
                (max_col, min_row),
                (max_col, max_row),
                (min_col, max_row),
            ]
        )

    return component_pixels_list, bbox_list
