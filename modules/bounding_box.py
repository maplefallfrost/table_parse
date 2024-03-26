class BoundingBox:
    def __init__(self, bounds):
        self.min_row, self.max_row, self.min_col, self.max_col = bounds

    @property
    def area(self) -> int:
        return (self.max_row - self.min_row) * (self.max_col - self.min_col)

    @property
    def center(self) -> tuple[float, float]:
        centor_row = (self.min_row + self.max_row) / 2
        centor_col = (self.min_col + self.max_col) / 2
        return centor_row, centor_col

    def distance(self, other: "BoundingBox") -> float:
        return (
            (self.center[0] - other.center[0]) ** 2
            + (self.center[1] - other.center[1]) ** 2
        ) ** 0.5
