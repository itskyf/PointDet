import dataclasses
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from numpy.typing import NDArray

from ...core.bbox.structures import CameraBoxes3D, LiDARBoxes3D
from ...typing import BoxAnnotation, PointCloud
from ..interface import IDataset
from .typing import KittiAnnotation, KittiInfo


class KittiDataset(IDataset):
    CLASSES = ("Car", "Pedestrian", "Cyclist")

    def __init__(
        self,
        root_split: Path,
        info_path: Path,
        pts_prefix: str,
        rng: np.random.Generator,
        transforms: Optional[Callable[[PointCloud], PointCloud]] = None,
    ):
        super().__init__(info_path, rng, transforms, training="train" in info_path.name)
        self.root_split = root_split
        self.pts_prefix = pts_prefix

    def _getitem_impl(self, info: KittiInfo) -> PointCloud:
        calib = info.calib
        rect = calib.r0_rect
        trv2c = calib.trv2c

        lidar2img = calib.P2 @ rect @ trv2c
        annos = None
        if self.training:
            assert info.annos is not None, "Training sample does not incluce annotation"
            annos = self._get_annotation(info.annos, rect, trv2c) if self.training else None

        v_path = self.root_split / self.pts_prefix / f"{info.sample_idx:06d}.bin"
        points = torch.from_numpy(np.fromfile(v_path, dtype=np.float32))
        points = points.reshape(-1, 4)
        return PointCloud(info.sample_idx, lidar2img, points, annos)

    def _get_annotation(
        self, annos: KittiAnnotation, rect: NDArray[np.float32], trv2c: NDArray[np.float32]
    ) -> BoxAnnotation:
        """Get annotation info according to the given index.

        Args:
            index (int): index of the annotation data to get.

        Returns:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): 3D ground truth bboxes.
            gt_labels_3d (np.ndarray): labels of ground truths.
            gt_bboxes (np.ndarray): 2D ground truth bboxes.
            gt_labels (np.ndarray): labels of ground truths.
            gt_names (list[str]): class names of ground truths.
            difficulty (int): difficulty defined by KITTI. 0, 1, 2 represent xxxxx respectively.
        """
        # TODO maybe use index to get the annos, thus the evalhook could also use this api
        annos = _remove_annos_names(annos, "DontCare")
        rots = annos.rotation_y[..., np.newaxis]
        cam_bboxes_3d = CameraBoxes3D(
            torch.from_numpy(np.concatenate((annos.location, annos.dimensions, rots), axis=1))
        )
        bboxes_3d = LiDARBoxes3D.from_camera_box3d(cam_bboxes_3d, rect, trv2c)
        bboxes = torch.from_numpy(annos.bbox)
        # TODO process when there is plane

        labels = [self.CLASSES.index(name) if name in self.CLASSES else -1 for name in annos.names]
        labels = np.array(labels, dtype=np.int32)
        labels_3d = labels.copy()
        return BoxAnnotation(
            bboxes_3d, labels_3d, bboxes, labels, annos.names, annos.difficulty, annos.group_ids
        )


def get_data_path(idx: int, root: Path, info_type: str, suffix: str, training: bool):
    file_name = f"{idx:06d}.{suffix}"
    file_path = Path("training" if training else "testing") / info_type / file_name
    check_path = root / file_path
    if not check_path.exists():
        raise FileNotFoundError(check_path)
    return file_path


def _remove_annos_names(annos: KittiAnnotation, del_name: str) -> KittiAnnotation:
    """Remove annotations that do not need to be cared."""
    indices = [i for i, name in enumerate(annos.names) if name != del_name]
    replace_dict = {
        field.name: getattr(annos, field.name)[indices]
        for field in dataclasses.fields(KittiAnnotation)
    }
    return dataclasses.replace(annos, **replace_dict)
