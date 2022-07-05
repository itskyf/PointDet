from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .core.bbox.structures import LiDARBoxes3D
    from .core.points import LiDARPoints


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
class PCDAugmentation:
    # Flipping
    flip_dir: Optional[FlipDirection] = field(init=False)
    h_flip: bool = field(init=False)
    v_flip: bool = field(init=False)
    # Affine transform
    scale_factor: float = field(init=False)
    rotation_angle: float = field(init=False)
    # TODO rotation angle
    trans_factor: NDArray[np.float32] = field(init=False)


@dataclass
class PointCloud:
    sample_idx: int
    lidar2img: NDArray[np.float32]
    points: LiDARPoints
    annos: Optional[BoxAnnotation] = None

    gt_bboxes_3d: LiDARBoxes3D = field(init=False)
    gt_labels_3d: NDArray[np.int32] = field(init=False)

    aug_info: PCDAugmentation = field(default_factory=PCDAugmentation)

    def __post_init__(self):
        """Using this hook to ensure throwing error when augment sample with no annos"""
        if self.annos is not None:
            self.gt_bboxes_3d = self.annos.bboxes_3d
            self.gt_labels_3d = self.annos.labels


@dataclass
class DBInfo:
    name: str
    box3d_lidar: NDArray[np.float32]
    path: Path
    difficulty: int
    gid: int
    gt_idx: int
    img_idx: int
    num_gt_points: int
