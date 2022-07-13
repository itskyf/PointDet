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
        points: torch.Tensor,
        centroids: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            points (torch.Tensor): (B, N, D) coordinates of the points.
            centroids (torch.Tensor): (B, K, D) coordinates of the centriods.
            features (torch.Tensor): (B, C, N) The features of grouped points.
        Returns:
            grouped_xyz: (B, D, K, num_neighbors)
            grouped_features: (B, C, K, num_neighbors)
        """
        # (B, K, num_neighbors)
        indices = ball_query(centroids, points, self.num_neighbors, self.radius)
        # Transpose since last dim of points is its features
        # (B, 3, K, num_neighbors)
        grouped_xyz = grouping_ops(points.transpose(1, 2), indices)
        # Relative offsets
        if self.relative_xyz:
            grouped_xyz -= centroids.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            grouped_features = grouping_ops(features, indices)
            return grouped_xyz, grouped_features
        return grouped_xyz, None

    def extra_repr(self):
        return ", ".join(
            [
                f"num_neighbors={self.num_neighbors}",
                f"radius={self.radius}",
                f"relative_xyz={self.relative_xyz}",
                f"normalize_xyz={self.normalize_xyz}",
            ]
        )


class GroupingOperation(Function):
    """Group feature with given index."""

    @staticmethod
    def forward(ctx, features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (Tensor): (B, feat_dims, num_feats) tensor of features to group.
            indices (Tensor): (B, P1, K) the indices of features to group with.
        Returns:
            (B, C, K, num_neighbors) Grouped features.
        """
        grouped_feats = _C.group_points(features, indices)
        ctx.for_backwards = (indices, features.size(2))
        return grouped_feats

    @staticmethod
    def backward(ctx, grad_grouped: torch.Tensor):
        """
        Args:
            grad_out (Tensor): (B, C, K, num_neighbors)
            tensor of the gradients of the output from forward.
        Returns:
            (B, C, N) gradient of the features.
        """
        indices, num_feats = ctx.for_backwards
        grad_feats = _C.group_points_backward(grad_grouped.data, indices, num_feats)
        return grad_feats, None


grouping_ops = GroupingOperation.apply
