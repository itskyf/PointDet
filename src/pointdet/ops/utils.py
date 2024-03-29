# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch

from ..core.bbox.structures.utils import rotation_3d_in_axis


def boxes_to_corners_3d(boxes3d: torch.Tensor):
    # TODO convert this function to LiDARBoxes3D's method
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    template = [
        [1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
        [-1, 1, -1],
        [1, 1, 1],
        [1, -1, 1],
        [-1, -1, 1],
        [-1, 1, 1],
    ]
    template = boxes3d.new_tensor(template) / 2

    corners3d = boxes3d[:, None, 3:6].expand(-1, 8, -1) * template.unsqueeze(0)
    corners3d = rotation_3d_in_axis(corners3d, boxes3d[:, 6], axis=2)
    return corners3d + boxes3d[:, None, 0:3]


def masked_gather(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Helper function for torch.gather to collect the points at
    the given indices in idx where some of the indices might be -1 to
    indicate padding. These indices are first replaced with 0.
    Then the points are gathered after which the padded values
    are set to 0.0.

    Args:
        points: (B, N, D) float32 tensor of points
        indices: (B, K) or (B, N, K) long tensor of indices into points,
            where some indices are -1 to indicate padding

    Returns:
        selected_points: (B, K, D) float32 tensor of points
            at the given indices
    """

    if points.size(0) != indices.size(0):
        raise ValueError("points and idx must have the same batch dimension")
    point_dims = points.size(2)

    if indices.ndim == 3:
        # Case: KNN, Ball Query where idx is of shape (N, P', K)
        # where P' is not necessarily the same as P as the
        # points may be gathered from a different pointcloud.
        num_gathers = indices.size(2)
        # Match dimensions for points and indices
        idx_expanded = indices[..., None].expand(-1, -1, -1, point_dims)
        points = points[:, :, None, :].expand(-1, -1, num_gathers, -1)
    elif indices.ndim == 2:
        # Farthest point sampling where idx is of shape (N, K)
        idx_expanded = indices[..., None].expand(-1, -1, point_dims)
    else:
        raise ValueError(f"idx format is not supported {indices.shape}")

    idx_expanded_mask = idx_expanded.eq(-1)
    idx_expanded = idx_expanded.clone()
    # Replace -1 values with 0 for gather
    idx_expanded[idx_expanded_mask] = 0
    # Gather points
    selected_points = points.gather(dim=1, index=idx_expanded)
    # Replace padded values
    selected_points[idx_expanded_mask] = 0.0
    return selected_points


def split_point_feats(points: torch.Tensor):
    points_xyz = points[..., :3].contiguous()
    features = points[..., 3:].transpose(1, 2).contiguous()
    return points_xyz, features
