from typing import Optional

import torch
from torch import nn
from torch.autograd import Function

from pointdet import _C

from .ball_query import ball_query


class QueryAndGroup(nn.Module):
    """Groups points with a ball query of radius.
    Args:
        radius (float): The maximum radius of the balls.
        num_neighbors (int): Maximum number of features to gather in the ball.
        use_xyz (bool): Whether to use xyz. Default: True.
        return_grouped_xyz (bool): Whether to return grouped xyz. Default: False.
        normalize_xyz (bool): Whether to normalize xyz. Default: False.
        uniform_sample (bool): Whether to sample uniformly. Default: False
        return_grouped_idx (bool): Whether to return grouped idx. Default: False.
    """

    def __init__(
        self,
        num_neighbors: int,
        radius: float,
        relative_xyz: bool = True,
        normalize_xyz: bool = False,
    ):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.radius = radius
        self.relative_xyz = relative_xyz
        self.normalize_xyz = normalize_xyz

    def forward(
        self,
        points_xyz: torch.Tensor,
        center_xyz: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            points_xyz (torch.Tensor): (B, N, 3) xyz coordinates of the points.
            center_xyz (torch.Tensor): (B, num_groups, 3) coordinates of the centriods.
            features (torch.Tensor): (B, C, N) The features of grouped points.
        Returns:
            grouped_xyz: (B, 3, num_groups, num_neighbors)
            grouped_features: (B, C, num_groups, num_neighbors)
            Grouped coordinates and features of points.
        """
        # (B, num_groups, num_neighbors)
        indices = ball_query(center_xyz, points_xyz, self.num_neighbors, self.radius)
        xyz_trans = points_xyz.transpose(1, 2).contiguous()
        # (B, 3, num_groups, num_neighbors)
        grouped_xyz = _grouping_operation.apply(xyz_trans, indices)
        # Relative offsets
        if self.relative_xyz:
            grouped_xyz -= center_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            grouped_features = _grouping_operation.apply(features, indices)
            return grouped_xyz, grouped_features
        return grouped_xyz, None


class _grouping_operation(Function):
    """Group feature with given index."""

    @staticmethod
    def forward(ctx, features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (Tensor): (B, feat_dims, num_feats) tensor of features to group.
            indices (Tensor): (B, P1, K) the indices of features to group with.
        Returns:
            Tensor: (B, C, num_groups, num_neighbors) Grouped features.
        """
        grouped_feats = _C.group_points(features, indices)
        ctx.for_backwards = (indices, features.size(2))
        return grouped_feats

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            grad_out (Tensor): (B, C, num_groups, num_neighbors)
            tensor of the gradients of the output from forward.
        Returns:
            Tensor: (B, C, N) gradient of the features.
        """
        indices, num_feats = ctx.for_backwards
        grad_features = _C.group_points_backward(grad_out.data, indices, num_feats)
        return grad_features, None
