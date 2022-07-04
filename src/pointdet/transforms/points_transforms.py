import numpy as np

from ..core.bbox.structures import CameraBoxes3D, LiDARBoxes3D
from ..typing import PointCloud


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
        mask = gt_bboxes_3d.in_range_bev(bev_range)

        gt_labels_3d = pcd.gt_labels_3d
        # mask is a torch tensor but gt_labels_3d is still numpy array, using mask to index
        # gt_labels_3d will cause bug when len(gt_labels_3d) == 1, where mask=1 will be
        # interpreted as gt_labels_3d[1] and cause out of index error
        pcd.gt_labels_3d = gt_labels_3d[mask.bool().numpy()]

        gt_bboxes_3d = gt_bboxes_3d.new_box(gt_bboxes_3d.tensor[mask])
        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        pcd.gt_bboxes_3d = gt_bboxes_3d
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
        points = pcd.points
        x_mask = (points[:, 0] > self.pcd_range[0]) & (points[:, 0] < self.pcd_range[3])
        y_mask = (points[:, 1] > self.pcd_range[1]) & (points[:, 1] < self.pcd_range[4])
        z_mask = (points[:, 2] > self.pcd_range[2]) & (points[:, 2] < self.pcd_range[5])
        pcd.points = points[x_mask & y_mask & z_mask]
        return pcd


class PointShuffle:
    """Shuffle input points."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def __call__(self, pcd: PointCloud) -> PointCloud:
        """Call function to shuffle points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask'
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        pcd.points = pcd.points[self.rng.permutation(pcd.points.shape[0])]
        return pcd
