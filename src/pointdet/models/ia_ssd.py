import torch
from torch import nn

from ..datasets import PCDBatch
from .backbones import ia_ssd, pointnet2
from .head import IASSDHead


class IASSDNet(nn.Module):
    def __init__(
        self,
        encoder: ia_ssd.IASSDEncoder,
        vote_layer: ia_ssd.ContextualCentroidPerception,
        centroid_agg_layer: pointnet2.PointsAggregation,
        head: IASSDHead,
    ):
        super().__init__()
        self.encoder = encoder
        self.vote_layer = vote_layer
        self.centroid_agg_layer = centroid_agg_layer
        self.head = head

    def forward(self, pcd_batch: PCDBatch):
        points_and_feats = torch.stack(pcd_batch.points_list)
        points, features = _split_point_feats(points_and_feats)

        features, cls_preds_list, points_list = self.encoder(points, features)
        ctr_preds, ctr_offsets, points = self.vote_layer(
            points_list[-1], features, cls_preds_list[-1]
        )
        # ctr_preds is centers, points output of vote_layer is centers_origin
        ctr_feats = self.centroid_agg_layer(points_list[-1], features, ctr_preds)
        points_list.append(points)
        self.head(
            ctr_preds, ctr_feats, pcd_batch.gt_boxes_list, pcd_batch.gt_labels_list, points_list
        )


def _split_point_feats(points: torch.Tensor):
    points_xyz = points[..., :3].contiguous()
    features = points[..., 3:].transpose(1, 2).contiguous()
    return points_xyz, features
