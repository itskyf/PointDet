import torch
from torch import nn

from ....ops import masked_gather, sampling
from ...modules import build_normal_mlps


class ContextualCentroidPerception(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_points: int,
        in_channels: int,
        mid_channels: int,
        max_translate_range: tuple[float, float, float],
    ):
        # TODO medium remove mid_channels?
        super().__init__()
        self.confidence_layer = nn.Sequential(
            *build_normal_mlps(in_channels, in_channels, dims=1),
            nn.Conv1d(in_channels, num_classes, kernel_size=1, bias=False),
        )
        self.num_points = num_points
        self.centroid_reg = nn.Sequential(
            *build_normal_mlps(in_channels, mid_channels, dims=1),
            nn.Conv1d(mid_channels, 3, kernel_size=1, bias=False),
        )
        self.register_buffer(
            "max_offset_limit", torch.tensor(max_translate_range).view(1, 1, 3), persistent=False
        )

    def forward(self, points: torch.Tensor, features: torch.Tensor):
        """
        Args:
            points (B, N, D) xyz coordinates of the features (generally D = 3).
            features (B, C, N) features of each point.
            cls_features (B, N, num_classes)
        Returns:
            ctr_preds (B, K, D)
            ctr_offsets (B, K, D)
        """
        assert points.size(1) > self.num_points, points.size(1)
        cls_preds = self.confidence_layer(features)
        indices = sampling.centroid_aware(cls_preds, self.num_points)

        ctr_origins = masked_gather(points, indices)
        features = masked_gather(features.transpose(1, 2), indices)
        features = features.transpose(1, 2)

        # Centroid prediction
        max_offset_limit = self.get_buffer("max_offset_limit")
        max_offset_limit = max_offset_limit.expand(ctr_origins.size(0), ctr_origins.size(1), -1)
        ctr_offsets = self.centroid_reg(features).transpose(1, 2)

        limited_ctr_offsets = torch.where(
            ctr_offsets > max_offset_limit, max_offset_limit, ctr_offsets
        )
        min_offset_limit = torch.neg(max_offset_limit)
        limited_ctr_offsets = torch.where(
            limited_ctr_offsets < min_offset_limit, min_offset_limit, limited_ctr_offsets
        )
        ctr_preds = ctr_origins + limited_ctr_offsets
        return ctr_preds, ctr_origins, ctr_offsets, cls_preds
