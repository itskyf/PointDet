import torch
from torch import nn

from ...modules import build_normal_mlps
from ..pointnet2 import PointNet2SAMSG


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
        confidence_pos: tuple[int, ...],
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
        assert list(confidence_pos) == sorted(set(confidence_pos))
        assert len(confidence_pos) <= len_aggregation and max(confidence_pos) <= len_aggregation
        super().__init__()

        in_channels_list = [in_channels, *aggregation_channels_list[:-1]]
        sa_modules = [
            PointNet2SAMSG(n_points, s_method, n_neighbors, radii, in_cs, mlps_cs, agg_cs)
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

        confidence_layers = {
            str(i): nn.Sequential(
                *build_normal_mlps(c_cs := aggregation_channels_list[i], c_cs, dims=1),
                nn.Conv1d(c_cs, num_classes, kernel_size=1, bias=False),
            )
            for i in confidence_pos
        }
        self.confidence_layers = nn.ModuleDict(confidence_layers)

    def forward(self, points: torch.Tensor, features: torch.Tensor):
        """
        Args:
            points (B, N, D) xyz coordinates of the features (generally D = 3).
            features (B, C, N) features of each point.

        Returns:
            points (B, K, D) sampled points
            features (B, agg_channels, K) new feature descriptors.
            cls_preds_list: list[(B, K, num_classes)]
        """
        cls_preds = None
        cls_preds_list = []
        # TODO critical return cls_preds_list?
        points_list = []
        for i, sa_module in enumerate(self.sa_modules):
            points, features = sa_module(points, features, cls_preds)
            points_list.append(points)
            pos = str(i)
            cls_preds = (
                self.confidence_layers[pos](features).transpose(1, 2)
                if pos in self.confidence_layers
                else None
            )
            if cls_preds is not None:
                cls_preds_list.append(cls_preds)
        return features, cls_preds_list, points_list
