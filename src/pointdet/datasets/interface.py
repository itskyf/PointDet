import abc
import pickle
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from torch.utils.data import Dataset

from ..typing import PointCloud


class IDataset(Dataset):
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
        self.transforms = transforms

    def __getitem__(self, index: int) -> PointCloud:
        info = self.infos[index]
        sample = self._getitem_impl(info)
        if sample.annos is not None and not np.any(sample.annos.labels != -1):
            # No interesting labels
            return self[self.rng.integers(len(self))]
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.infos)

    @abc.abstractmethod
    def _getitem_impl(self, info) -> PointCloud:
        ...
