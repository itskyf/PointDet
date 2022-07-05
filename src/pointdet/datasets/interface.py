import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from torch.utils.data import Dataset

from ..transforms import Compose
from ..typing import PointCloud


class IDataset(ABC, Dataset):
    CLASSES: tuple[str, ...]

    def __init__(
        self,
        info_path: Path,
        rng: np.random.Generator,
        transforms: Optional[Callable[[PointCloud], PointCloud]],
        training: bool,
    ):
        with info_path.open("rb") as info_file:
            self.infos = pickle.load(info_file)
        self.rng = rng
        self.training = training
        self.transforms = transforms if transforms is not None else Compose([])

    def __getitem__(self, index: int) -> PointCloud:
        sample = self.get_sample(index)
        if sample.annos is not None and not np.any(sample.annos.labels != -1):
            # No interesting labels
            return self[self.rng.integers(len(self))]
        return self.transforms(sample)

    def __len__(self):
        return len(self.infos)

    @abstractmethod
    def _get_sample(self, info) -> PointCloud:
        ...

    def get_sample(self, index: int) -> PointCloud:
        """Get a sample without losing infomation when sample doesn't have GT object"""
        info = self.infos[index]
        return self._get_sample(info)
