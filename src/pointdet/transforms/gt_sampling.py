import copy
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from numpy.typing import NDArray

from ..core.bbox import box_np_ops
from ..core.points import LiDARPoints
from ..typing import DBInfo, PointCloud
from .utils import box_collision_test

DBInfos = dict[str, list[DBInfo]]
cs = ConfigStore.instance()


class _DBSamples(NamedTuple):
    bboxes_3d: NDArray[np.float32]
    labels_3d: NDArray[np.int32]
    group_ids: NDArray[np.int32]
    points: torch.Tensor


class BatchSampler:
    """Class for sampling specific category of ground truths.

    Args:
        sample_list (list[dict]): List of samples.
        name (str, optional): The category of samples. Default: None.
        epoch (int, optional): Sampling epoch. Default: None.
        shuffle (bool, optional): Whether to shuffle indices. Default: False.
    """

    def __init__(
        self, cls_infos: list[DBInfo], name: str, rng: np.random.Generator, shuffle: bool = True
    ):
        self._cls_infos = cls_infos
        self._rng = rng
        self._len_dbinfos = len(cls_infos)
        self._indices = list(range(self._len_dbinfos))
        if shuffle:
            self._rng.shuffle(self._indices)
        self._idx = 0
        self._name = name
        self._shuffle = shuffle

    def __call__(self, num: int) -> list[DBInfo]:
        """Sample specific number of ground truths.

        Args:
            num (int): Sampled number.

        Returns:
            list[dict]: Sampled ground truths.
        """
        indices = self._sample(num)
        return [self._cls_infos[i] for i in indices]

    def _sample(self, num: int) -> list[int]:
        """Sample specific number of ground truths and return indices.

        Args:
            num (int): Sampled number.

        Returns:
            list[int]: Indices of sampled ground truths.
        """
        if self._idx + num >= self._len_dbinfos:
            # Copy to prevent _reset_idx changing the result
            ret = self._indices[self._idx :].copy()
            self._reset_idx()
        else:
            ret = self._indices[self._idx : self._idx + num]
            self._idx += num
        return ret

    def _reset_idx(self):
        """Reset the index of batchsampler to zero."""
        assert self._name is not None
        # print("reset", self._name)
        if self._shuffle:
            self._rng.shuffle(self._indices)
        self._idx = 0


@dataclass
class DBSamplerConf:
    root: Path
    info_name: str
    rate: float
    classes: list[str]
    min_points: dict[str, int]  # TODO better typing
    sample_groups: dict[str, int]


cs.store(name="db_sampler", node=DBSamplerConf)


class DBSampler:
    """Class for sampling data from the ground truth database.

    Args:
        info_path (str): Path of groundtruth database info.
        data_root (str): Path of groundtruth database.
        rate (float): Rate of actual sampled over maximum sampled number.
        prepare (dict): Name of preparation functions and the input value.
        sample_groups (dict): Sampled classes and numbers.
        classes (list[str]): list of model's classes (not the whole dataset)
    """

    def __init__(
        self,
        root: Path,
        info_name: Path,
        rate: float,
        classes: list[str],
        min_points: dict[str, int],  # TODO repalce with callable instance using hydra
        sample_groups: dict[str, int],
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.data_root = root
        self.info_path = root / info_name
        self.rate = rate
        self.name2label = {name: i for i, name in enumerate(classes)}
        self.label2name = dict(enumerate(classes))

        # TODO logging
        # Filter database infos
        with self.info_path.open("rb") as info_file:
            db_infos: DBInfos = pickle.load(info_file)
        # TODO medium remove hardcode preparation
        db_infos = self.filter_by_difficulty(db_infos, [-1])
        db_infos = self.filter_by_min_points(db_infos, min_points)

        self.sample_classes = list(sample_groups.keys())
        self.sample_max_nums = list(sample_groups.values())
        rng = np.random.default_rng(seed)
        self._sampler_dict = {
            cls_name: BatchSampler(cls_infos, cls_name, rng)
            for cls_name, cls_infos in db_infos.items()
        }

    def __call__(
        self, gt_bboxes: NDArray[np.float32], gt_labels: NDArray[np.int32]
    ) -> Optional[_DBSamples]:
        """Sampling all categories of bboxes.

        Args:
            gt_bboxes (np.ndarray): Ground truth bounding boxes.
            gt_labels (np.ndarray): Ground truth labels of boxes.

        Returns:
            dict: Dict of sampled 'pseudo ground truths'.

                - gt_labels_3d (np.ndarray): ground truths labels
                    of sampled objects.
                - gt_bboxes_3d (:obj:`BaseInstance3DBoxes`):
                    sampled ground truth 3D bounding boxes
                - points (np.ndarray): sampled points
                - group_ids (np.ndarray): ids of sampled ground truths
        """
        num_sample_dict: dict[str, int] = {}
        for cls_name, sample_max in zip(self.sample_classes, self.sample_max_nums):
            cls_label = self.name2label[cls_name]
            num_sample = sample_max - np.count_nonzero(cls_label == gt_labels)
            if (num_sample := round(self.rate * num_sample)) > 0:
                num_sample_dict[cls_name] = num_sample

        sampled: list[DBInfo] = []
        sampled_bboxes = []
        avoid_coll_boxes = gt_bboxes

        for cls_name, num_sample in num_sample_dict.items():
            sampled_cls = self._sample_class_v2(cls_name, num_sample, avoid_coll_boxes)
            sampled += sampled_cls
            if len(sampled_cls) > 0:
                sampled_box = (
                    sampled_cls[0].box3d_lidar[np.newaxis, ...]
                    if len(sampled_cls) == 1
                    else np.stack([db_info.box3d_lidar for db_info in sampled_cls])
                )
                sampled_bboxes.append(sampled_box)
                avoid_coll_boxes = np.concatenate([avoid_coll_boxes, sampled_box])

        if len(sampled) == 0:
            return None

        sampled_pts_tensors = []
        for info in sampled:
            points = np.load(self.data_root / info.path)
            points[:, :3] += info.box3d_lidar[:3]
            sampled_pts_tensors.append(points)
        # TODO ground_plane
        sampled_bboxes = np.concatenate(sampled_bboxes)
        sampled_labels = np.array([self.name2label[info.name] for info in sampled], dtype=np.int32)
        group_ids = np.arange(gt_bboxes.shape[0], gt_bboxes.shape[0] + len(sampled))
        points = torch.cat(sampled_pts_tensors)
        return _DBSamples(sampled_bboxes, sampled_labels, group_ids, points)

    @staticmethod
    def filter_by_difficulty(db_infos: DBInfos, del_difficulties: list[int]) -> DBInfos:
        """Filter ground truths by difficulties.

        Args:
            db_infos (dict): Info of groundtruth database.
            removed_difficulty (list): Difficulties that are not qualified.

        Returns:
            dict: Info of database after filtering.
        """
        return {
            key: [info for info in cls_infos if info.difficulty not in del_difficulties]
            for key, cls_infos in db_infos.items()
        }

    @staticmethod
    def filter_by_min_points(db_infos: DBInfos, min_gt_pts: dict[str, int]) -> DBInfos:
        """Filter ground truths by number of points in the bbox.

        Args:
            db_infos (dict): Info of groundtruth database.
            min_gt_points_dict (dict): Different number of minimum points
                needed for different categories of ground truths.

        Returns:
            dict: Info of database after filtering.
        """
        for name, min_num in min_gt_pts.items():
            if min_num > 0:
                db_infos[name] = [info for info in db_infos[name] if info.num_gt_points >= min_num]
        return db_infos

    def _sample_class_v2(self, name: str, num: int, gt_bboxes: NDArray[np.float32]) -> list[DBInfo]:
        """Sampling specific categories of bounding boxes.

        Args:
            name (str): Class of objects to be sampled.
            num (int): Number of sampled bboxes.
            gt_bboxes (np.ndarray): Ground truth boxes.

        Returns:
            list[dict]: Valid samples after collision test.
        """
        sampled: list[DBInfo] = copy.deepcopy(self._sampler_dict[name](num))  # TODO copy necessary?

        num_gt = gt_bboxes.shape[0]
        num_sampled = len(sampled)
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6]
        )

        sp_boxes = np.stack([db_info.box3d_lidar for db_info in sampled])
        boxes = np.concatenate([gt_bboxes, sp_boxes])

        sp_boxes_new = boxes[gt_bboxes.shape[0] :]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6]
        )

        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv])
        coll_mat = box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if np.any(coll_mat[i]):
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])
        return valid_samples


class GTSampler:
    """Sample GT objects to the data.

    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
        use_ground_plane (bool): Whether to use gound plane to adjust the
            3D labels.
    """

    def __init__(self, db_sampler: DBSampler):
        self.db_sampler = db_sampler

    def __call__(self, pcd: PointCloud) -> PointCloud:
        """Call function to sample ground truth objects to the data.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after object sampling augmentation,
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated
                in the result dict.
        """
        gt_bboxes_3d = pcd.gt_bboxes_3d.tensor.numpy()
        gt_labels_3d = pcd.gt_labels_3d

        # TODO use ground_plane
        # change to float for blending operation
        # TODO sample 2D
        sampled: Optional[_DBSamples] = self.db_sampler(gt_bboxes_3d, gt_labels_3d)
        if sampled is not None:
            # Add sampled boxes to scene
            pcd.gt_bboxes_3d = pcd.gt_bboxes_3d.new_boxes(
                np.concatenate([gt_bboxes_3d, sampled.bboxes_3d])
            )
            pcd.gt_labels_3d = np.concatenate([gt_labels_3d, sampled.labels_3d])
            points = _remove_points_in_boxes(pcd.points, sampled.bboxes_3d)
            attr_dims = points.attr_dims
            points = torch.cat([sampled.points, points.tensor])
            pcd.points = LiDARPoints(points, points_dim=points.size(1), attr_dims=attr_dims)
        return pcd


def _remove_points_in_boxes(points: LiDARPoints, boxes: NDArray[np.float32]) -> LiDARPoints:
    """Remove the points in the sampled bounding boxes.

    Args:
        points (:obj:`BasePoints`): Input point cloud array.
        boxes (np.ndarray): Sampled ground truth boxes.

    Returns:
        np.ndarray: Points with those in the boxes removed.
    """
    masks = box_np_ops.points_in_rbbox(points.coor.numpy(), boxes)
    return points[np.logical_not(masks.any(-1))]
