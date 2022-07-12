# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..core.bbox.structures import CameraBoxes3D, LiDARBoxes3D
from ..typing import FlipDirection, PointCloud


class Compose:
    def __init__(self, transforms: list[Callable[[PointCloud], PointCloud]]):
        self.transforms = transforms

    def __call__(self, data: PointCloud) -> PointCloud:
        for transform in self.transforms:
            data = transform(data)
        return data


class GlobalRotScaleTrans:
    """Apply global rotation, scaling and translation to a 3D scene.
    Args:
        rot_range (list[float], optional): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float], optional): Range of scale ratio.
            Defaults to [0.95, 1.05].
        translation_std (list[float], optional): The standard deviation of
            translation noise applied to a scene, which
            is sampled from a gaussian distribution whose standard deviation
            is set by ``translation_std``. Defaults to [0, 0, 0]
        shift_height (bool, optional): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(
        self,
        rot_range: tuple[float, float] = (-np.pi / 4, np.pi / 4),
        scale_ratio_range: tuple[float, float] = (0.95, 1.05),
        translation_std: tuple[float, float, float] = (0, 0, 0),
        seed: Optional[int] = None,
    ):
        assert all(std >= 0 for std in translation_std), "translation_std should be positive"
        self.rng = np.random.default_rng(seed)
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = np.array(translation_std, dtype=np.float32)

    def __call__(self, pcd: PointCloud) -> PointCloud:
        """Private function to rotate, scale and translate bounding boxes and
        points.
        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
                'pcd_scale_factor', 'pcd_trans' and keys in
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        self._rot_bbox_points(pcd)
        self._scale_bbox_points(pcd)
        self._trans_bbox_points(pcd)
        return pcd

    def _rot_bbox_points(self, pcd: PointCloud):
        """Private function to rotate bounding boxes and points.
        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation'
                and keys in input_dict['bbox3d_fields'] are updated
                in the result dict.
        """
        noise_rotation = np.random.uniform(self.rot_range[0], self.rot_range[1])
        pcd.gt_bboxes_3d.rotate(noise_rotation)
        pcd.points.rotate(noise_rotation)
        # TODO this function is different with mmdet3d's
        # mmdet3d's version rotate points using matrix returned after rotate gt_boxes_3d

    def _scale_bbox_points(self, pcd: PointCloud):
        """Private function to scale bounding boxes and points.
        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after scaling, 'points'and keys in
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        try:
            scale_factor = pcd.aug_info.scale_factor
        except AttributeError:
            scale_factor = self.rng.uniform(self.scale_ratio_range[0], self.scale_ratio_range[1])
            pcd.aug_info.scale_factor = scale_factor
        pcd.gt_bboxes_3d.scale(scale_factor)
        pcd.points.scale(scale_factor)

    def _trans_bbox_points(self, pcd: PointCloud):
        """Private function to translate bounding boxes and points.
        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after translation, 'points', 'pcd_trans'
                and keys in input_dict['bbox3d_fields'] are updated
                in the result dict.
        """
        trans_vec = self.rng.normal(scale=self.translation_std, size=3).astype(np.float32)
        pcd.aug_info.trans_factor = trans_vec
        pcd.gt_bboxes_3d.translate(trans_vec)
        pcd.points.translate(trans_vec)


class ObjectRangeFilter:
    """Filter objects by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range: list[float]):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, pcd: PointCloud) -> PointCloud:
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d'
                keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range
        if isinstance(pcd.gt_bboxes_3d, LiDARBoxes3D):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(pcd.gt_bboxes_3d, CameraBoxes3D):
            bev_range = self.pcd_range[[0, 2, 3, 5]]
        else:
            raise TypeError

        gt_bboxes_3d = pcd.gt_bboxes_3d
        bev = pcd.gt_bboxes_3d.bev

        xmin_mask = bev[:, 0] > bev_range[0]
        ymin_mask = bev[:, 1] > bev_range[1]
        xmax_mask = bev[:, 0] < bev_range[2]
        ymax_mask = bev[:, 1] < bev_range[3]
        mask = xmin_mask & ymin_mask & xmax_mask & ymax_mask

        gt_bboxes_3d = gt_bboxes_3d.new_boxes(gt_bboxes_3d.tensor[mask])
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        pcd.gt_bboxes_3d = gt_bboxes_3d
        pcd.gt_labels_3d = pcd.gt_labels_3d[mask.bool().numpy()]
        return pcd


class PointsRangeFilter:
    """Filter points by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range: list[float]):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, pcd: PointCloud) -> PointCloud:
        """Call function to filter points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask'
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        tensor = pcd.points.tensor
        x_mask = (tensor[:, 0] > self.pcd_range[0]) & (tensor[:, 0] < self.pcd_range[3])
        y_mask = (tensor[:, 1] > self.pcd_range[1]) & (tensor[:, 1] < self.pcd_range[4])
        z_mask = (tensor[:, 2] > self.pcd_range[2]) & (tensor[:, 2] < self.pcd_range[5])
        pcd.points = pcd.points.new_points(tensor[x_mask & y_mask & z_mask])
        return pcd


class PointsSample:
    """Point sample.

    Sampling data to a certain number.

    Args:
        num_points (int): Number of points to be sampled.
        sample_range (float, optional): The range where to sample points.
            If not None, the points with depth larger than `sample_range` are
            prior to be sampled. Defaults to None.
        replace (bool, optional): Whether the sampling is with or without
            replacement. Defaults to False.
    """

    def __init__(self, num_points: int, seed: Optional[int] = None):
        self.num_samples = num_points
        self.rng = np.random.default_rng(seed)

    def _points_random_sampling(self, points, num_samples):
        """Points random sampling.

        Sample points to a certain number.

        Args:
            points (np.ndarray | :obj:`BasePoints`): 3D Points.
            num_samples (int): Number of samples to be sampled.
            sample_range (float, optional): Indicating the range where the
                points will be sampled. Defaults to None.
            replace (bool, optional): Sampling with or without replacement.
                Defaults to None.
            return_choices (bool, optional): Whether return choice.
                Defaults to False.
        Returns:
            tuple[np.ndarray] | np.ndarray:
                - points (np.ndarray | :obj:`BasePoints`): 3D Points.
                - choices (np.ndarray, optional): The generated random samples.
        """
        if not replace:
            replace = points.shape[0] < num_samples
        point_range = range(len(points))
        choices = np.random.choice(point_range, num_samples, replace=replace)
        return points[choices]

    def __call__(self, pcd: PointCloud) -> PointCloud:
        """Call function to sample points to in indoor scenes.

        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask'
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        tensor = pcd.points.tensor
        num_points = tensor.size(0)
        choices = self.rng.choice(
            range(num_points), self.num_samples, replace=num_points < self.num_samples
        )
        pcd.points = pcd.points.new_points(tensor[choices])
        return pcd


class PointsShuffle:
    """Shuffle input points."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, pcd: PointCloud) -> PointCloud:
        """Call function to shuffle points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask'
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        src_pts = pcd.points
        src_tensor = src_pts.tensor
        pcd.points = src_pts.new_points(src_tensor[self.rng.permutation(src_tensor.size(0))])
        return pcd


class RandomFlip:
    """Flip the image & bbox & mask.
    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.
    When random flip is enabled, ``flip_ratio``/``direction`` can either be a
    float/string or tuple of float/string. There are 3 flip modes:
    - ``flip_ratio`` is float, ``direction`` is string: the image will be
        ``direction``ly flipped with probability of ``flip_ratio`` .
        E.g., ``flip_ratio=0.5``, ``direction='horizontal'``,
        then image will be horizontally flipped with probability of 0.5.
    - ``flip_ratio`` is float, ``direction`` is list of string: the image will
        be ``direction[i]``ly flipped with probability of
        ``flip_ratio/len(direction)``.
        E.g., ``flip_ratio=0.5``, ``direction=['horizontal', 'vertical']``,
        then image will be horizontally flipped with probability of 0.25,
        vertically with probability of 0.25.
    - ``flip_ratio`` is list of float, ``direction`` is list of string:
        given ``len(flip_ratio) == len(direction)``, the image will
        be ``direction[i]``ly flipped with probability of ``flip_ratio[i]``.
        E.g., ``flip_ratio=[0.3, 0.5]``, ``direction=['horizontal',
        'vertical']``, then image will be horizontally flipped with probability
        of 0.3, vertically with probability of 0.5.
    Args:
        flip_ratio (float | list[float], optional): The flipping probability.
            Default: None.
        direction(str | list[str], optional): The flipping direction. Options
            are 'horizontal', 'vertical', 'diagonal'. Default: 'horizontal'.
            If input is a list, the length must equal ``flip_ratio``. Each
            element in ``flip_ratio`` indicates the flip probability of
            corresponding direction.
    """

    def __init__(
        self,
        flip_ratio: Union[float, list[float]],
        direction: Union[FlipDirection, list[FlipDirection]],
        seed: Optional[int] = None,
    ):
        if isinstance(flip_ratio, list):
            assert isinstance(direction, list) and len(flip_ratio) == len(direction)
            assert 0 <= sum(flip_ratio) <= 1
        elif isinstance(flip_ratio, float):
            assert 0 <= flip_ratio <= 1
        else:
            raise TypeError("flip_ratios must be float or list of float")

        self.directions = [*direction, None] if isinstance(direction, list) else [direction, None]
        self.num_direction = len(self.directions) - 1
        self.flip_ratio = flip_ratio
        self.rng = np.random.default_rng(seed)

    def bbox_flip(
        self, bboxes: NDArray[np.float32], img_shape: tuple[int, int], direction: FlipDirection
    ) -> NDArray[np.float32]:
        """Flip bboxes horizontally.
        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.
        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """

        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if direction is FlipDirection.HORIZONTAL:
            width = img_shape[1]
            flipped[..., 0::4] = width - bboxes[..., 2::4]
            flipped[..., 2::4] = width - bboxes[..., 0::4]
        elif direction is FlipDirection.VERTICAL:
            height = img_shape[0]
            flipped[..., 1::4] = height - bboxes[..., 3::4]
            flipped[..., 3::4] = height - bboxes[..., 1::4]
        elif direction is FlipDirection.DIAGONAL:
            height, width = img_shape
            flipped[..., 0::4] = width - bboxes[..., 2::4]
            flipped[..., 1::4] = height - bboxes[..., 3::4]
            flipped[..., 2::4] = width - bboxes[..., 0::4]
            flipped[..., 3::4] = height - bboxes[..., 1::4]
        else:
            raise NotImplementedError(f"Invalid flipping direction '{direction}'")
        return flipped

    def __call__(self, pcd: PointCloud) -> PointCloud:
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added \
                into result dict.
        """
        try:
            flip_dir = pcd.aug_info.flip_dir
        except AttributeError:
            # Randomly select a direction
            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratios = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / self.num_direction
                flip_ratios = [single_ratio] * self.num_direction + [non_flip_ratio]
            flip_dir = self.rng.choice(self.directions, p=flip_ratios)
            pcd.aug_info.flip_dir = flip_dir
        # TODO flip image
        return pcd


class RandomFlip3D(RandomFlip):
    """Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio_bev_horizontal (float, optional): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float, optional): The flipping probability
            in vertical direction. Defaults to 0.0.
        sync_2d (bool, optional): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images. Defaults to True.
    """

    def __init__(
        self,
        h_bev_flip_ratio: float = 0.0,
        v_bev_flip_ratio: float = 0.0,
        sync_2d: bool = True,
        seed: Optional[int] = None,
    ):
        assert 0 <= h_bev_flip_ratio <= 1 and 0 <= v_bev_flip_ratio <= 1
        super().__init__(h_bev_flip_ratio, FlipDirection.HORIZONTAL, seed)
        self.sync_2d = sync_2d
        self.v_bev_flip_ratio = v_bev_flip_ratio

    def _random_flip_data_3d(self, pcd: PointCloud, direction: FlipDirection):
        """Flip 3D data randomly.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str, optional): Flip direction.
                Default: 'horizontal'.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are
                updated in the result dict.
        """
        # TODO test mode, segmentation task, centers2d
        pcd.points.flip(direction)
        pcd.gt_bboxes_3d.flip(direction)

    def __call__(self, pcd: PointCloud) -> PointCloud:
        """Call function to flip points, values in the ``bbox3d_fields`` and
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction',
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added
                into result dict.
        """
        # flip 2D image and its annotations
        super().__call__(pcd)
        if self.sync_2d:
            pcd.aug_info.h_flip = pcd.aug_info.flip_dir is not None
            pcd.aug_info.v_flip = False
        else:
            rand_ratio = self.rng.random()
            try:
                pcd.aug_info.h_flip
            except AttributeError:
                assert isinstance(self.flip_ratio, float)
                pcd.aug_info.h_flip = rand_ratio < self.flip_ratio
            try:
                pcd.aug_info.v_flip
            except AttributeError:
                pcd.aug_info.v_flip = rand_ratio < self.v_bev_flip_ratio

        if pcd.aug_info.h_flip:
            self._random_flip_data_3d(pcd, FlipDirection.HORIZONTAL)
        if pcd.aug_info.v_flip:
            self._random_flip_data_3d(pcd, FlipDirection.VERTICAL)
        return pcd
