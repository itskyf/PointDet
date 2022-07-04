from typing import Callable

from ..typing import PointCloud


class Compose:
    def __init__(self, transforms: list[Callable[[PointCloud], PointCloud]]):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data
