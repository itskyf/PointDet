from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray

from .core.bbox.structures import LiDARBoxes3D


@dataclass
class BoxAnnotation:
    bboxes_3d: LiDARBoxes3D
    bboxes: torch.Tensor
    labels: NDArray[np.int32]
    names: NDArray[np.str_]
    difficulty: NDArray[np.int32]
    group_ids: NDArray[np.int32]
    # TODO plane_lidar


@dataclass
class PointCloud:
    sample_idx: int
    lidar2img: NDArray[np.float32]
    points: torch.Tensor
    annos: Optional[BoxAnnotation]


@dataclass
class DBInfo:
    name: str
    box3d_lidar: torch.Tensor
    path: Path
    difficulty: int
    gid: int
    gt_idx: int
    img_idx: int
    num_gt_points: int
