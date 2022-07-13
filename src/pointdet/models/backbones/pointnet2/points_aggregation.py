import torch
from torch import nn

from ....ops.group_points import QueryAndGroup
from ...modules import build_normal_mlps


class PointsAggregation(nn.Module):
    def __init__(
        self,
        num_neighbors: tuple[int, ...],
        radii: tuple[float, ...],
        in_channels: int,
        mlps_channels: tuple[tuple[int, ...], ...],
        aggregation_channels: int,
        reduction: str = "max",  # TODO convert to enum
        use_xyz: bool = True,
    ):
        assert len(mlps_channels) == len(num_neighbors) == len(radii)
        assert reduction in ("max", "mean", "sum"), reduction
        super().__init__()

        self.groupers = nn.ModuleList(
            [QueryAndGroup(k, radius) for k, radius in zip(num_neighbors, radii)]
        )
        self.use_xyz = use_xyz
        if self.use_xyz:
            in_channels += 3
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(*build_normal_mlps(in_channels, mlp_channels, dims=2))
                for mlp_channels in mlps_channels
            ]
        )

        self.aggreation_layer = nn.Sequential(
            *build_normal_mlps(
                in_channels=sum(mlp_channels[-1] for mlp_channels in mlps_channels),
                channels=aggregation_channels,
                dims=1,
            )
        )

        if reduction == "max":
            self.reduction_layer = lambda x: torch.max(x, dim=-1, keepdim=False).values
        elif reduction == "mean":
            self.reduction_layer = lambda x: torch.mean(x, dim=-1, keepdim=False)
        elif reduction == "sum":
            self.reduction_layer = lambda x: torch.sum(x, dim=-1, keepdim=False)
        else:
            raise ValueError

    def forward(self, points: torch.Tensor, features: torch.Tensor, centroids: torch.Tensor):
        """
        Args:
            points (B, N, D) xyz coordinates of the features (generally D = 3).
            features (B, C, N) features of each point.
            centroids (B, M, num_classes)

        Returns:
            new_feats (B, agg_channels, M) new feature descriptors.
        """
        new_feats_list = []
        for grouper, mlp in zip(self.groupers, self.mlps):
            grouped_xyz, new_feats = grouper(points, centroids, features)
            if self.use_xyz:
                new_feats = torch.cat([grouped_xyz, new_feats], dim=1)
            new_feats = mlp(new_feats)  # B, mlp_channels[-1], K, num_neighbors
            new_feats = self.reduction_layer(new_feats)  # B, mlp_channels[-1], K
            new_feats_list.append(new_feats)
        features = torch.cat(new_feats_list, dim=1)
        return self.aggreation_layer(features)
