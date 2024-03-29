class BoundingBox:
    def __init__(self, bounds):
        self.min_row, self.max_row, self.min_col, self.max_col = bounds

    def merge(self, box: "BoundingBox"):
        self.min_row = min(self.min_row, box.min_row)
        self.max_row = max(self.max_row, box.max_row)
        self.min_col = min(self.min_col, box.min_col)
        self.max_col = max(self.max_col, box.max_col)

    @property
    def area(self) -> int:
        return (self.max_row - self.min_row) * (self.max_col - self.min_col)

    @property
    def center(self) -> tuple[float, float]:
        centor_row = (self.min_row + self.max_row) / 2
        centor_col = (self.min_col + self.max_col) / 2
        return centor_row, centor_col

    def center_distance(self, other: "BoundingBox") -> float:
        return (
            (self.center[0] - other.center[0]) ** 2
            + (self.center[1] - other.center[1]) ** 2
        ) ** 0.5

    def min_distance(self, other: "BoundingBox") -> float:
        return min(
            abs(self.min_row - other.min_row),
            abs(self.max_row - other.max_row),
            abs(self.min_col - other.min_col),
            abs(self.max_col - other.max_col),
        )

    def copy(self):
        return type(self)((self.min_row, self.max_row, self.min_col, self.max_col))
