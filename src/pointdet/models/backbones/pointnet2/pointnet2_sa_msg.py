import torch
from torch import nn

from ....ops import masked_gather, sampling
from ...modules import build_normal_mlps
from .points_aggregation import PointsAggregation


class PointNet2SAMSGSampling(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_points: int,
        sampling_method: str,  # TODO convert to enum
        num_neighbors: tuple[int, ...],
        radii: tuple[float, ...],
        in_channels: int,
        mlps_channels: tuple[tuple[int, ...], ...],
        aggregation_channels: int,
    ):
        assert num_points > 0
        assert sampling_method in ("D-FPS", "CentroidAware"), sampling_method
        super().__init__()
        self.confidence_layer = (
            nn.Sequential(
                *build_normal_mlps(in_channels, in_channels, dims=1),
                nn.Conv1d(in_channels, num_classes, kernel_size=1, bias=False),
            )
            if sampling_method == "CentroidAware"
            else None
        )
        self.num_points = num_points
        self.group_agg_layer = PointsAggregation(
            num_neighbors, radii, in_channels, mlps_channels, aggregation_channels
        )

    def forward(self, points: torch.Tensor, features: torch.Tensor):
        """
        Args:
            points (B, N, D) xyz coordinates of the features (generally D = 3).
            features (B, C, N) features of each point.

        Returns:
            sampled_points (B, K, D) where K is the number of points.
            features (B, agg_channels, K) new feature descriptors.
        """
        assert points.size(1) > self.num_points, points.size(1)
        cls_preds = self.confidence_layer(features) if self.confidence_layer is not None else None
        indices = (
            sampling.centroid_aware(cls_preds, self.num_points)
            if cls_preds is not None
            else sampling.sample_farthest_points(points, self.num_points)
        )
        sampled_points = masked_gather(points, indices)
        features = self.group_agg_layer(points, features, sampled_points)
        return sampled_points, features, cls_preds
