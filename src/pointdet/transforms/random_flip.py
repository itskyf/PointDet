# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..typing import FlipDirection, PointCloud


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
        rng: np.random.Generator,
    ):
        if isinstance(flip_ratio, list):
            assert isinstance(direction, list) and len(flip_ratio) == len(direction)
            assert 0 <= sum(flip_ratio) <= 1
        elif isinstance(flip_ratio, float):
            assert 0 <= flip_ratio <= 1
        else:
            raise TypeError("flip_ratios must be float or list of float")

        self.direction_list = (
            [*direction, None] if isinstance(direction, list) else [direction, None]
        )
        self.num_direction = len(self.direction_list) - 1
        self.flip_ratio = flip_ratio
        self.rng = rng

    def bbox_flip(self, bboxes, img_shape: tuple[int, int], direction):
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

    def __call__(self, pt_cloud: PointCloud) -> PointCloud:
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added \
                into result dict.
        """
        if not hasattr(pt_cloud, "flip_direction"):
            # Randomly select a direction
            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / self.num_direction
                flip_ratio_list = [single_ratio] * self.num_direction + [non_flip_ratio]
            pt_cloud.flip_direction = self.rng.choice(self.direction_list, p=flip_ratio_list)
        # TODO flip image
        return pt_cloud


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
        h_bev_flip_ratio: float,
        v_bev_flip_ratio: float,
        sync_2d: bool,
        rng: np.random.Generator,
    ):
        assert 0 <= h_bev_flip_ratio <= 1 and 0 <= v_bev_flip_ratio <= 1
        super().__init__(h_bev_flip_ratio, FlipDirection.HORIZONTAL, rng)
        self.sync_2d = sync_2d
        self.v_bev_flip_ratio = v_bev_flip_ratio

    def _random_flip_points(self, points: NDArray[np.float32], direction: FlipDirection):
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
        if direction is FlipDirection.HORIZONTAL:
            points[:, 1] = -points[:, 1]
        elif direction is FlipDirection.VERTICAL:
            points[:, 0] = -points[:, 0]
        else:
            raise NotImplementedError

    def __call__(self, pt_cloud: PointCloud) -> PointCloud:
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
        super().__call__(pt_cloud)
        if self.sync_2d:
            pt_cloud.pcd_h_flip = pt_cloud.flip_direction is not None
            pt_cloud.pcd_v_flip = False
        else:
            rand_ratio = self.rng.random()
            if not hasattr(pt_cloud, "pcd_h_flip"):
                assert isinstance(self.flip_ratio, float)
                pt_cloud.pcd_h_flip = rand_ratio < self.flip_ratio
            if not hasattr(pt_cloud, "pcd_v_flip"):
                pt_cloud.pcd_v_flip = rand_ratio < self.v_bev_flip_ratio

        if pt_cloud.pcd_h_flip:
            self._random_flip_points(pt_cloud.points, FlipDirection.HORIZONTAL)
        if pt_cloud.pcd_v_flip:
            self._random_flip_points(pt_cloud.points, FlipDirection.VERTICAL)
        return pt_cloud
