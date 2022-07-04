from typing import Optional

import numpy as np

from ..core.bbox.structures.utils import rotation_3d_in_axis
from ..typing import PointCloud


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
        rng: np.random.Generator,
        rot_range: Optional[tuple[float, float]] = None,
        scale_ratio_range: Optional[tuple[float, float]] = None,
        translation_std: Optional[tuple[float, float, float]] = None,
    ):
        if translation_std is None:
            translation_std = (0, 0, 0)
        assert all(std >= 0 for std in translation_std), "translation_std should be positive"
        if rot_range is None:
            rot_range = (-0.78539816, 0.78539816)
        if scale_ratio_range is None:
            scale_ratio_range = (0.95, 1.05)

        self.rng = rng
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

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
        noise_rotation = self.rng.uniform(self.rot_range[0], self.rot_range[1])
        rotated_points, rot_mat_t = rotation_3d_in_axis(
            pcd.points[:, :3][None], noise_rotation, return_mat=True
        )

        pcd.points[:, :3] = rotated_points.squeeze(0)
        pcd.aug_info.rotation = np.array(rot_mat_t.squeeze(0))
        pcd.aug_info.rotation_angle = noise_rotation

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

        pcd.points[:, :3] *= scale_factor
        pcd.gt_bboxes_3d.scale(scale_factor)

    def _trans_bbox_points(self, pcd: PointCloud):
        """Private function to translate bounding boxes and points.
        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after translation, 'points', 'pcd_trans'
                and keys in input_dict['bbox3d_fields'] are updated
                in the result dict.
        """
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_vec = self.rng.normal(scale=translation_std, size=3).astype(np.float32).T
        pcd.aug_info.trans_factor = trans_vec
        pcd.points[:, :3] += trans_vec
        pcd.gt_bboxes_3d.translate(trans_vec)
