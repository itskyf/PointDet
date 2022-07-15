import torch
from torch import nn

from ..pointnet2 import PointNet2SAMSGSampling


class IASSDEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_points_list: tuple[int],
        sampling_methods: tuple[str],  # TODO convert to enum
        num_neighbors_list: tuple[tuple[int, ...]],
        radii_list: tuple[tuple[float, ...]],
        mlps_channels_list: tuple[tuple[tuple[int, ...], ...]],
        aggregation_channels_list: tuple[int, ...],
    ):
        len_aggregation = len(aggregation_channels_list)
        assert (
            len(num_points_list)
            == len(sampling_methods)
            == len(num_neighbors_list)
            == len(radii_list)
            == len(mlps_channels_list)
            == len_aggregation
        )
        super().__init__()

        in_channels_list = [in_channels, *aggregation_channels_list[:-1]]
        sa_modules = [
            PointNet2SAMSGSampling(
                num_classes, n_points, s_method, n_neighbors, radii, in_cs, mlps_cs, agg_cs
            )
            for n_points, s_method, n_neighbors, radii, in_cs, mlps_cs, agg_cs in zip(
                num_points_list,
                sampling_methods,
                num_neighbors_list,
                radii_list,
                in_channels_list,
                mlps_channels_list,
                aggregation_channels_list,
            )
        ]
        self.sa_modules = nn.ModuleList(sa_modules)

    def forward(self, points: torch.Tensor, features: torch.Tensor):
        """
        Args:
            points (B, N, D) xyz coordinates of the features (generally D = 3).
            features (B, C, N) features of each point.

        Returns:
            points (B, K, D) sampled points
            features (B, agg_channels, K) new feature descriptors.
        """
        cls_preds_list = []
        points_list = []
        for sa_module in self.sa_modules:
            in_points = points
            points, features, cls_preds = sa_module(points, features)
            if cls_preds is not None:
                cls_preds_list.append(cls_preds)
                points_list.append(in_points)
        return points, features, cls_preds_list, points_list
