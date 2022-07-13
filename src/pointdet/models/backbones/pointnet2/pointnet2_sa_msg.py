from typing import Optional

import torch
from torch import nn

from ....ops import masked_gather
from ...modules import SamplingLayer
from .points_aggregation import PointsAggregation


class PointNet2SAMSG(nn.Module):
    def __init__(
        self,
        num_points: int,
        sampling_method: str,  # TODO convert to enum
        num_neighbors: tuple[int, ...],
        radii: tuple[float, ...],
        in_channels: int,
        mlps_channels: tuple[tuple[int, ...], ...],
        aggregation_channels: int,
    ):
        super().__init__()
        self.sampling_layer = SamplingLayer(num_points, sampling_method)
        self.group_agg_layer = PointsAggregation(
            num_neighbors, radii, in_channels, mlps_channels, aggregation_channels
        )

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        cls_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points (B, N, D) xyz coordinates of the features (generally D = 3).
            features (B, C, N) features of each point.
            cls_features (B, N, num_classes)

        Returns:
            sampled_points (B, K, D) where K is the number of points.
            features (B, agg_channels, K) new feature descriptors.
        """
        indices = self.sampling_layer(points, cls_features)
        sampled_points = masked_gather(points, indices)
        features = self.group_agg_layer(points, features, sampled_points)
        return sampled_points, features
