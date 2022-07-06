from typing import Optional

import numba
import numpy as np
from numpy.typing import NDArray

from ..core.bbox import box_np_ops
from ..typing import PointCloud
from .utils import box_collision_test


class ObjectNoise:
    """Apply noise to each GT objects in the scene.
    Args:
        translation_std (list[float], optional): Standard deviation of the
            distribution where translation noise are sampled from.
            Defaults to [0.25, 0.25, 0.25].
        global_rot_range (list[float], optional): Global rotation to the scene.
            Defaults to [0.0, 0.0].
        rot_range (list[float], optional): Object rotation range.
            Defaults to [-0.15707963267, 0.15707963267].
        num_try (int, optional): Number of times to try if the noise applied is
            invalid. Defaults to 100.
    """

    def __init__(
        self,
        translation_std: tuple[float, float, float] = (0.25, 0.25, 0.25),
        global_rot_range: tuple[float, float] = (0.0, 0.0),
        rot_range: tuple[float, float] = (-0.15707963267, 0.15707963267),
        num_try: int = 100,
        seed: Optional[int] = None,
    ):
        self.translation_std = translation_std
        self.global_rot_range = global_rot_range
        self.rot_range = rot_range
        self.num_try = num_try
        self.rng = np.random.default_rng(seed)

    def __call__(self, pcd: PointCloud) -> PointCloud:
        """Call function to apply noise to each ground truth in the scene.
        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after adding noise to each object,
                'points', 'gt_bboxes_3d' keys are updated in the result dict.
        """
        # TODO: remove inplace operation
        gt_bboxes_3d = pcd.gt_bboxes_3d
        points = pcd.points

        np_bbox3d = gt_bboxes_3d.tensor.numpy()
        np_points = points.tensor.numpy()
        _noise_per_object_v3(
            self.rng,
            np_bbox3d,
            np_points,
            self.global_rot_range,
            rotation_perturb=self.rot_range,
            center_noise_std=self.translation_std,
            num_try=self.num_try,
        )

        pcd.gt_bboxes_3d = gt_bboxes_3d.new_boxes(np_bbox3d)
        pcd.points = points.new_points(np_points)
        return pcd


@numba.njit
def _box3d_transform(boxes, loc_transform, rot_transform, valid_mask):
    """Transform 3D boxes.
    Args:
        boxes (np.ndarray): 3D boxes to be transformed.
        loc_transform (np.ndarray): Location transform to be applied.
        rot_transform (np.ndarray): Rotation transform to be applied.
        valid_mask (np.ndarray): Mask to indicate which boxes are valid.
    """
    num_box = boxes.shape[0]
    for i in range(num_box):
        if valid_mask[i]:
            boxes[i, :3] += loc_transform[i]
            boxes[i, 6] += rot_transform[i]


@numba.njit
def _noise_per_box(boxes, valid_mask, loc_noises, rot_noises):
    """Add noise to every box (only on the horizontal plane).
    Args:
        boxes (np.ndarray): Input boxes with shape (N, 5).
        valid_mask (np.ndarray): Mask to indicate which boxes are valid
            with shape (N).
        loc_noises (np.ndarray): Location noises with shape (N, M, 3).
        rot_noises (np.ndarray): Rotation noises with shape (N, M).
    Returns:
        np.ndarray: Mask to indicate whether the noise is
            added successfully (pass the collision test).
    """
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    rot_mat_t = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones(num_boxes, dtype=np.int32)
    # print(valid_mask)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_corners[:] = box_corners[i]
                current_corners -= boxes[i, :2]
                _rotation_box2d_jit(current_corners, rot_noises[i, j], rot_mat_t)
                current_corners += boxes[i, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(current_corners.reshape(1, 4, 2), box_corners)
                coll_mat[0, i] = False
                # print(coll_mat)
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    break
    return success_mask


@numba.njit
def _noise_per_box_v2(boxes, valid_mask, loc_noises, rot_noises, global_rot_noises):
    """Add noise to every box (only on the horizontal plane). Version 2 used
    when enable global rotations.
    Args:
        boxes (np.ndarray): Input boxes with shape (N, 5).
        valid_mask (np.ndarray): Mask to indicate which boxes are valid
            with shape (N).
        loc_noises (np.ndarray): Location noises with shape (N, M, 3).
        rot_noises (np.ndarray): Rotation noises with shape (N, M).
    Returns:
        np.ndarray: Mask to indicate whether the noise is
            added successfully (pass the collision test).
    """
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    current_box = np.zeros((1, 5), dtype=boxes.dtype)
    rot_mat_t = np.zeros((2, 2), dtype=boxes.dtype)
    dst_pos = np.zeros((2,), dtype=boxes.dtype)
    success_mask = -np.ones(num_boxes, dtype=np.int32)
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners_norm = corners_norm.reshape(4, 2)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_box[0, :] = boxes[i]
                current_radius = np.sqrt(boxes[i, 0] ** 2 + boxes[i, 1] ** 2)
                current_grot = np.arctan2(boxes[i, 0], boxes[i, 1])
                dst_grot = current_grot + global_rot_noises[i, j]
                dst_pos[0] = current_radius * np.sin(dst_grot)
                dst_pos[1] = current_radius * np.cos(dst_grot)
                current_box[0, :2] = dst_pos
                current_box[0, -1] += dst_grot - current_grot

                rot_sin = np.sin(current_box[0, -1])
                rot_cos = np.cos(current_box[0, -1])
                rot_mat_t[0, 0] = rot_cos
                rot_mat_t[0, 1] = rot_sin
                rot_mat_t[1, 0] = -rot_sin
                rot_mat_t[1, 1] = rot_cos
                current_corners[:] = (
                    current_box[0, 2:4] * corners_norm @ rot_mat_t + current_box[0, :2]
                )
                current_corners -= current_box[0, :2]
                _rotation_box2d_jit(current_corners, rot_noises[i, j], rot_mat_t)
                current_corners += current_box[0, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(current_corners.reshape(1, 4, 2), box_corners)
                coll_mat[0, i] = False
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    loc_noises[i, j, :2] += dst_pos - boxes[i, :2]
                    rot_noises[i, j] += dst_grot - current_grot
                    break
    return success_mask


def _noise_per_object_v3(
    rng: np.random.Generator,
    gt_boxes: NDArray[np.float32],
    points: NDArray[np.float32],
    global_random_rot_range: tuple[float, float],
    rotation_perturb: tuple[float, float],
    center_noise_std: tuple[float, float, float],
    num_try: int,
    valid_mask: Optional[NDArray[np.bool_]] = None,
):
    """Random rotate or remove each groundtruth independently. use kitti viewer
    to test this function points_transform_
    Args:
        gt_boxes (np.ndarray): Ground truth boxes with shape (N, 7).
        points (np.ndarray, optional): Input point cloud with
            shape (M, 4). Default: None.
        valid_mask (np.ndarray, optional): Mask to indicate which
            boxes are valid. Default: None.
        rotation_perturb (float, optional): Rotation perturbation.
            Default: pi / 4.
        center_noise_std (float, optional): Center noise standard deviation.
            Default: 1.0.
        global_random_rot_range (float, optional): Global random rotation
            range. Default: pi/4.
        num_try (int, optional): Number of try. Default: 100.
    """
    num_boxes = gt_boxes.shape[0]
    enable_grot = np.abs(global_random_rot_range[0] - global_random_rot_range[1]) >= 1e-3
    if valid_mask is None:
        valid_mask = np.ones(num_boxes, dtype=np.bool_)

    loc_noises = rng.normal(scale=center_noise_std, size=[num_boxes, num_try, 3])
    rot_noises = rng.uniform(rotation_perturb[0], rotation_perturb[1], size=[num_boxes, num_try])
    gt_grots = np.arctan2(gt_boxes[:, 0], gt_boxes[:, 1])
    grot_lowers = global_random_rot_range[0] - gt_grots
    grot_uppers = global_random_rot_range[1] - gt_grots
    global_rot_noises = rng.uniform(
        grot_lowers[..., np.newaxis], grot_uppers[..., np.newaxis], size=[num_boxes, num_try]
    )
    gt_box_corners = box_np_ops.center_to_corner_box3d(
        gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, 6], origin=(0.5, 0.5, 0), axis=2
    )

    # TODO: rewrite this noise box function?
    in_gt_boxes = gt_boxes[:, [0, 1, 3, 4, 6]]
    selected_noise = (
        _noise_per_box_v2(in_gt_boxes, valid_mask, loc_noises, rot_noises, global_rot_noises)
        if enable_grot
        else _noise_per_box(in_gt_boxes, valid_mask, loc_noises, rot_noises)
    )

    loc_transforms = _select_transform(loc_noises, selected_noise)
    rot_transforms = _select_transform(rot_noises, selected_noise)
    surfaces = box_np_ops.corner_to_surfaces_3d_jit(gt_box_corners)
    if points is not None:
        # TODO: replace this points_in_convex function by my tools?
        point_masks = box_np_ops.points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
        _points_transform(
            points, gt_boxes[:, :3], point_masks, loc_transforms, rot_transforms, valid_mask
        )

    _box3d_transform(gt_boxes, loc_transforms, rot_transforms, valid_mask)


@numba.njit
def _points_transform(points, centers, point_masks, loc_transform, rot_transform, valid_mask):
    """Apply transforms to points and box centers.
    Args:
        points (np.ndarray): Input points.
        centers (np.ndarray): Input box centers.
        point_masks (np.ndarray): Mask to indicate which points need
            to be transformed.
        loc_transform (np.ndarray): Location transform to be applied.
        rot_transform (np.ndarray): Rotation transform to be applied.
        valid_mask (np.ndarray): Mask to indicate which boxes are valid.
    """
    num_box = centers.shape[0]
    num_points = points.shape[0]
    rot_mat_T = np.zeros((num_box, 3, 3), dtype=points.dtype)
    for i in range(num_box):
        _rotation_matrix_3d(rot_mat_T[i], rot_transform[i], 2)
    for i in range(num_points):
        for j in range(num_box):
            if valid_mask[j]:
                if point_masks[i, j] == 1:
                    points[i, :3] -= centers[j, :3]
                    points[i : i + 1, :3] = (
                        np.ascontiguousarray(points[i : i + 1, :3]) @ rot_mat_T[j]
                    )
                    points[i, :3] += centers[j, :3]
                    points[i, :3] += loc_transform[j]
                    break  # only apply first box's transform


@numba.njit
def _rotation_box2d_jit(corners, angle, rot_mat_t):
    """Rotate 2D boxes.
    Args:
        corners (np.ndarray): Corners of boxes.
        angle (float): Rotation angle.
        rot_mat_T (np.ndarray): Transposed rotation matrix.
    """
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_t[0, 0] = rot_cos
    rot_mat_t[0, 1] = rot_sin
    rot_mat_t[1, 0] = -rot_sin
    rot_mat_t[1, 1] = rot_cos
    corners[:] = corners @ rot_mat_t


@numba.njit
def _rotation_matrix_3d(rot_mat_T, angle, axis):
    """Get the 3D rotation matrix.
    Args:
        rot_mat_T (np.ndarray): Transposed rotation matrix.
        angle (float): Rotation angle.
        axis (int): Rotation axis.
    """
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[:] = np.eye(3)
    if axis == 1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 2] = rot_sin
        rot_mat_T[2, 0] = -rot_sin
        rot_mat_T[2, 2] = rot_cos
    elif axis == 2 or axis == -1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = rot_sin
        rot_mat_T[1, 0] = -rot_sin
        rot_mat_T[1, 1] = rot_cos
    elif axis == 0:
        rot_mat_T[1, 1] = rot_cos
        rot_mat_T[1, 2] = rot_sin
        rot_mat_T[2, 1] = -rot_sin
        rot_mat_T[2, 2] = rot_cos


def _select_transform(transform, indices):
    """Select transform.
    Args:
        transform (np.ndarray): Transforms to select from.
        indices (np.ndarray): Mask to indicate which transform to select.
    Returns:
        np.ndarray: Selected transforms.
    """
    result = np.zeros((transform.shape[0], *transform.shape[2:]), dtype=transform.dtype)
    for i in range(transform.shape[0]):
        if indices[i] != -1:
            result[i] = transform[i, indices[i]]
    return result
