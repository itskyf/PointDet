import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray

from .core.bbox.structures import LiDARBoxes3D


@dataclass
class BoxAnnotation:
    bboxes_3d: LiDARBoxes3D
    labels: NDArray[np.int32]
    bboxes: NDArray[np.float32]
    difficulty: NDArray[np.int32]
    group_ids: NDArray[np.int32]
    names: NDArray[np.str_]
    # TODO plane_lidar


class FlipDirection(enum.Enum):
    DIAGONAL = enum.auto()
    HORIZONTAL = enum.auto()
    VERTICAL = enum.auto()


@dataclass
class PointCloud:
    sample_idx: int
    lidar2img: NDArray[np.float32]
    points: NDArray[np.float32]
    annos: Optional[BoxAnnotation]

    bboxes_3d: LiDARBoxes3D = field(init=False)
    labels_3d: NDArray[np.int32] = field(init=False)

    flip_direction: Optional[FlipDirection] = field(init=False)
    pcd_h_flip: bool = field(init=False)
    pcd_v_flip: bool = field(init=False)

    def __post_init__(self):
        if self.annos is not None:
            self.bboxes_3d = self.annos.bboxes_3d
            self.labels_3d = self.annos.labels


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
