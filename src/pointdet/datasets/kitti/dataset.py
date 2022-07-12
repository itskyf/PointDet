import dataclasses
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from numpy.typing import NDArray

from ...core.bbox.structures import CameraBoxes3D, LiDARBoxes3D
from ...core.points import LiDARPoints
from ...typing import BoxAnnotation, PointCloud
from ..interface import IDataset
from .typing import KittiAnnotation, KittiInfo


class KittiDataset(IDataset):
    CLASSES = ("Car", "Pedestrian", "Cyclist")

    def __init__(
        self,
        root: Path,
        split: str,
        pts_prefix: str,
        info_prefix: str = "kitti",
        *,
        transforms: Optional[Callable[[PointCloud], PointCloud]] = None,
        seed: Optional[int] = None,
    ):
        assert split in ("train", "val", "test")
        info_path = root / f"{info_prefix}_infos_{split}.pkl"
        super().__init__(info_path, transforms, training=split == "train", seed=seed)

        self.path = root / ("testing" if split == "test" else "training")
        self.pts_prefix = pts_prefix

    def _get_sample(self, info: KittiInfo) -> PointCloud:
        calib = info.calib
        rect = calib.r0_rect
        trv2c = calib.trv2c
        lidar2img = calib.P2 @ rect @ trv2c

        annos = None
        if self.training:
            assert info.annos is not None
            annos = self._get_annotation(info.annos, rect, trv2c)

        v_path = self.path / self.pts_prefix / f"{info.index:06d}.pt"
        points = torch.load(v_path)
        points = LiDARPoints(points, points_dim=points.size(-1))
        return PointCloud(info.index, lidar2img, points, annos)

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
        # TODO maybe use index to get the annos, thus the evalhook? could also use this api
        annos = _remove_annos_names(annos, "DontCare")
        rots = annos.rotation_y[..., np.newaxis]
        cam_bboxes3d = np.concatenate(
            [annos.location, annos.dimensions, rots], axis=1, dtype=np.float32
        )
        cam_bboxes3d = CameraBoxes3D(cam_bboxes3d)

        bboxes_3d = LiDARBoxes3D.from_camera_box3d(cam_bboxes3d, rect, trv2c)
        # TODO process when there is plane

        labels = [self.CLASSES.index(name) if name in self.CLASSES else -1 for name in annos.names]
        labels = np.array(labels, dtype=np.int32)
        return BoxAnnotation(
            bboxes_3d, labels, annos.bboxes, annos.difficulty, annos.group_ids, annos.names
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
    indices = annos.names != del_name
    replace_dict = {
        field.name: getattr(annos, field.name)[indices]
        for field in dataclasses.fields(KittiAnnotation)
    }
    return dataclasses.replace(annos, **replace_dict)
