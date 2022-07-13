from typing import Optional

import torch
from torch import nn

from ...ops import sampling


class SamplingLayer(nn.Module):
    def __init__(
        self,
        num_points: int,
        sampling_method: str,  # TODO convert to enum
    ):
        assert num_points > 0
        assert sampling_method in ("D-FPS", "CentroidAware"), sampling_method
        super().__init__()
        self.num_points = num_points  # K
        self.method = sampling_method

    def forward(self, points: torch.Tensor, cls_features: Optional[torch.Tensor] = None):
        """
        Args:
            points (B, N, D) xyz coordinates of the features (generally D = 3).
            features (B, C, N) features of each point.
            cls_features (B, N, num_classes)

        Returns:
            sampled_points (B, M, D) where M is the number of points
            new_feats (B, agg_channels, M) new feature descriptors.
        """
        total_points = points.size(1)
        # Downsampling, indices: [B, num_points]
        if total_points <= self.num_points:  # No downsampling
            device = points.device
            batch_size = points.size(0)
            indices = torch.arange(total_points, dtype=torch.long, device=device)
            indices *= torch.ones(batch_size, total_points, dtype=torch.long, device=device)
        elif self.method == "D-FPS":
            indices = sampling.sample_farthest_points(points, self.num_points)
        elif self.method == "CentroidAware":
            assert cls_features is not None
            indices = sampling.centroid_aware(cls_features, self.num_points)
        else:
            raise NotImplementedError
        return indices

    def extra_repr(self):
        return f"num_points={self.num_points}, method={self.method}"
