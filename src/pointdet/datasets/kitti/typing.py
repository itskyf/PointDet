from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class KittiCalib:
    P0: NDArray[np.float32]
    P1: NDArray[np.float32]
    P2: NDArray[np.float32]
    P3: NDArray[np.float32]
    r0_rect: NDArray[np.float32]
    trv2c: NDArray[np.float32]
    tri2v: NDArray[np.float32]


@dataclass
class KittiAnnotation:
    index: NDArray[np.int32]
    names: NDArray[np.str_]
    difficulty: NDArray[np.int32]
    group_ids: NDArray[np.int32]
    truncated: NDArray[np.float32]
    occluded: NDArray[np.int32]
    alpha: NDArray[np.float32]
    bbox: NDArray[np.float32]
    dimensions: NDArray[np.float32]
    location: NDArray[np.float32]
    rotation_y: NDArray[np.float32]
    score: NDArray[np.float32]
    num_gt_points: NDArray[np.int32]


@dataclass
class KittiInfo:
    sample_idx: int
    calib: KittiCalib
    img_shape: NDArray[np.int32]
    num_features: int
    annos: Optional[KittiAnnotation]
