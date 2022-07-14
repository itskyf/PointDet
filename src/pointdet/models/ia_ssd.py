import torch
from torch import nn
from torch.nn import functional

from ..core.bbox.structures.utils import rotation_3d_in_axis
from ..datasets import PCDBatch
from .backbones import ia_ssd, pointnet2
from .head.ia_ssd import IASSDHead, TrainTargets


class IASSDNet(nn.Module):
    def __init__(
        self,
        encoder: ia_ssd.IASSDEncoder,
        vote_layer: ia_ssd.ContextualCentroidPerception,
        centroid_agg_layer: pointnet2.PointsAggregation,
        head: IASSDHead,
        loss_weights: dict[str, float],
    ):
        super().__init__()
        self.encoder = encoder
        self.vote_layer = vote_layer
        self.centroid_agg_layer = centroid_agg_layer
        self.head = head
        self.loss_weights = loss_weights

    def forward(self, pcd_batch: PCDBatch):
        points_and_feats = torch.stack(pcd_batch.points_list)
        points, features = _split_point_feats(points_and_feats)

        features, cls_preds_list, points_list = self.encoder(points, features)
        # Unlimited ctr_offsets is used for _contextual_vote_loss
        ctr_preds, ctr_origins, ctr_offsets = self.vote_layer(
            points_list[-1], features, cls_preds_list[-1]
        )
        ctr_feats = self.centroid_agg_layer(points_list[-1], features, ctr_preds)

        points_list.append(ctr_origins)
        pt_cls_preds, pt_box_preds, targets = self.head(
            ctr_preds, ctr_feats, pcd_batch.gt_boxes_list, pcd_batch.gt_labels_list, points_list
        )

        targets: TrainTargets
        if targets is not None:
            # Classification loss
            ctr = targets.ctr
            ctr_cls_loss = self._classification_loss(
                ctr_preds, pt_cls_preds, ctr.pts_cls_label, ctr.fg_pts_gt_box
            )
            # Corner loss
            corner_loss = self.get_corner_layer_loss()
            # Regression loss
            center_loss_box = self._center_box_binori_layer_loss()
            # Semantic loss in SA layers
            sa_loss_cls = self._sa_ins_layer_loss()
            # Voting loss
            center_loss_reg = self._contextual_vote_loss()
            return {
                "centroids_classification": ctr_cls_loss,
            }

    def get_corner_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict["center_cls_labels"] > 0
        gt_boxes = self.forward_ret_dict["center_gt_box_of_fg_points"]
        pred_boxes = self.forward_ret_dict["point_box_preds"]
        pred_boxes = pred_boxes[pos_mask]
        loss_corner = loss_utils.get_corner_loss_lidar(pred_boxes[:, 0:7], gt_boxes[:, 0:7])
        loss_corner = loss_corner.mean()
        loss_corner = loss_corner * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS["corner_weight"]
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({"corner_loss_reg": loss_corner.item()})
        return loss_corner, tb_dict

    def _classification_loss(
        self,
        ctr_preds: torch.Tensor,
        pt_cls_preds: torch.Tensor,
        ctr_cls_label: torch.Tensor,
        ctr_fg_pts_gt_box: torch.Tensor,
    ):
        """
        Args:
            pt_cls_preds [B, N, num_classes]
            ctr_cls_labels [B, N] category labels of each points
        """
        pos_mask = ctr_cls_label > 0
        neg_mask = ctr_cls_label == 0
        num_classes = pt_cls_preds.size(-1)

        cls_weights = neg_mask + pos_mask.float()
        cls_weights /= torch.clamp(pos_mask.sum(), min=1.0)
        cls_weights = cls_weights.unsqueeze(-1).expand(-1, -1, num_classes)

        fg_cls_labels = ctr_cls_label.clone()
        fg_cls_labels[fg_cls_labels < 0] = 0
        one_hot_targets = functional.one_hot(fg_cls_labels, num_classes + 1)
        # Remove first column since 0 is label of background
        one_hot_targets = one_hot_targets[..., 1:].float()

        # Centerness regularization
        centerness_mask = _generate_centerness_mask(ctr_preds, pos_mask, ctr_fg_pts_gt_box)
        one_hot_targets *= centerness_mask.unsqueeze(-1).expand(-1, -1, num_classes)

        pts_cls_loss = functional.binary_cross_entropy_with_logits(
            pt_cls_preds, one_hot_targets, cls_weights, reduction="none"
        )
        pts_cls_loss = pts_cls_loss.mean(dim=-1).sum()
        return pts_cls_loss * self.loss_weights["classification"]


def _split_point_feats(points: torch.Tensor):
    points_xyz = points[..., :3].contiguous()
    features = points[..., 3:].transpose(1, 2).contiguous()
    return points_xyz, features


def _generate_centerness_mask(
    ctr_preds: torch.Tensor, pos_mask: torch.Tensor, gt_boxes: torch.Tensor
):
    """
    Args:
        ctr_preds: FloatTensor [B, N, 3]
        pos_mask: BoolTensor [B, N]
        gt_boxes: FloatTensor [N_FG, 7]
    Returns:
        centerness_mask: FloatTensor [B, N, 1]
    """
    ctr_preds = ctr_preds[pos_mask].detach()
    offset_xyz = ctr_preds[:, 0:3] - gt_boxes[:, 0:3]

    offset_xyz_canical = rotation_3d_in_axis(offset_xyz.unsqueeze(1), -gt_boxes[:, 6], axis=2)
    offset_xyz_canical = offset_xyz_canical.squeeze(1)

    template = gt_boxes.new_tensor([[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]])
    margin = gt_boxes[:, None, 3:6].repeat(1, 2, 1) * template.unsqueeze(0)
    distance = margin - offset_xyz_canical[:, None, :].repeat(1, 2, 1)
    distance[:, 1, :] = -1 * distance[:, 1, :]
    distance_min = torch.where(
        distance[:, 0, :] < distance[:, 1, :], distance[:, 0, :], distance[:, 1, :]
    )
    distance_max = torch.where(
        distance[:, 0, :] > distance[:, 1, :], distance[:, 0, :], distance[:, 1, :]
    )

    centerness = distance_min / distance_max
    centerness = centerness[:, 0] * centerness[:, 1] * centerness[:, 2]
    centerness = torch.clamp(centerness, min=1e-6)
    centerness = torch.pow(centerness, 1 / 3)

    centerness_mask = torch.zeros_like(pos_mask, dtype=torch.float32)
    centerness_mask[pos_mask] = centerness
    return centerness_mask
