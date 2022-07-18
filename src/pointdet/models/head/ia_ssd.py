import enum
from typing import NamedTuple, Optional

import torch
from torch import nn

from pointdet import _C

from ...core.bbox.coder import PointResBinOriCoder
from ..modules.mlps import build_normal_mlps


class AssignType(enum.Enum):
    EXTEND_GT = enum.auto()
    IGNORE_FLAG = enum.auto()


class Target(NamedTuple):
    # Tensors have shape [B, N]
    pt_box_indices: torch.Tensor  # -1: background
    pt_cls_labels: torch.Tensor  # 0: background, -1: ignored
    fg_pt_gt_boxes: torch.Tensor  # [sum_nfg, 7]
    fg_pt_box_labels: Optional[torch.Tensor]


class TrainTargets(NamedTuple):
    ctr: Target
    ctr_org: Target
    sa_ins: list[Target]


class IASSDHead(nn.Module):
    def __init__(
        self,
        box_coder: PointResBinOriCoder,
        in_channels: int,
        mid_channels: int,
        num_classes: int,
        ext_dims: list[list[float]],
        mean_size: list[list[float]],
        extra_method: str,
    ):
        # TODO remove mid_channels?
        assert len(ext_dims) == 3 and all(len(dims) == 3 for dims in ext_dims)
        _ext_dims = torch.tensor(ext_dims, dtype=torch.float32)
        _mean_size = torch.tensor(mean_size, dtype=torch.float32)
        assert _mean_size.min() > 0 and _mean_size.shape == (3, 3)
        assert extra_method in ("extend_gt", "extend_query")
        super().__init__()
        self.box_coder = box_coder
        self.box_center_layer = nn.Sequential(
            *build_normal_mlps(in_channels, mid_channels, dims=1),
            nn.Conv1d(mid_channels, self.box_coder.code_size, kernel_size=1)
        )
        self.cls_center_layer = nn.Sequential(
            *build_normal_mlps(in_channels, mid_channels, dims=1),
            nn.Conv1d(mid_channels, num_classes, kernel_size=1)
        )

        self.register_buffer("gt_ext_dims", _ext_dims[0], persistent=False)
        self.register_buffer("sa_ext_dims", _ext_dims[1], persistent=False)
        self.register_buffer("ctr_org_ext_dims", _ext_dims[2], persistent=False)
        self.register_buffer("mean_size", _mean_size, persistent=False)
        self.extra_method = extra_method

    def forward(
        self,
        ctr_feats: torch.Tensor,
        ctr_preds: torch.Tensor,
        ctr_origins: torch.Tensor,
        sa_pts_list: list[torch.Tensor],
        gt_boxes_list: list[torch.Tensor],
        gt_labels_list: list[torch.Tensor],
    ):
        """
        Args:
        Returns:
            ctr_box_preds [N, B, bin_size]
            pt_box_preds [N, B, 7]
        """
        ctr_box_preds = self.box_center_layer(ctr_feats).transpose(1, 2)
        pt_cls_preds = self.cls_center_layer(ctr_feats).transpose(1, 2)

        pred_classes = pt_cls_preds.max(dim=-1).indices
        pt_box_preds = self.box_coder.decode_torch(
            ctr_box_preds, ctr_preds, pred_classes, self.get_buffer("mean_size")
        )

        targets = (
            self._assign_targets(ctr_preds, ctr_origins, sa_pts_list, gt_boxes_list, gt_labels_list)
            if self.training
            else None
        )
        return ctr_box_preds, pt_cls_preds, pt_box_preds, targets

    def _assign_targets(
        self,
        ctr_preds: torch.Tensor,
        ctr_origins: torch.Tensor,
        sa_pts_list: list[torch.Tensor],
        gt_boxes_list: list[torch.Tensor],
        gt_labels_list: list[torch.Tensor],
    ):
        """
        Args:
        Returns:
        """
        ctr_targets = self._assign_targets_stack(
            ctr_preds,
            gt_boxes_list,
            gt_labels_list,
            self.get_buffer("gt_ext_dims"),
            AssignType.IGNORE_FLAG,
            return_box_labels=True,
        )
        if self.extra_method == "extend_gt":
            ctr_org_targets = self._assign_targets_stack(
                ctr_origins,
                gt_boxes_list,
                gt_labels_list,
                self.get_buffer("ctr_org_ext_dims"),
                AssignType.EXTEND_GT,
                return_box_labels=True,
            )
        else:
            raise NotImplementedError
        sa_targets = [
            self._assign_targets_stack(
                points,
                gt_boxes_list,
                gt_labels_list,
                self.get_buffer("sa_ext_dims"),
                AssignType.IGNORE_FLAG if i == 0 else AssignType.EXTEND_GT,
            )
            for i, points in enumerate(sa_pts_list)
        ]
        return TrainTargets(ctr_targets, ctr_org_targets, sa_targets)

    def _assign_targets_stack(
        self,
        points_batch: torch.Tensor,
        gt_boxes_list: list[torch.Tensor],
        gt_labels_list: list[torch.Tensor],
        ext_dims: torch.Tensor,
        assign_type: AssignType,
        return_box_labels: bool = False,
    ):
        """
        Args:
        Returns:
        """
        points_batch = points_batch.detach()
        num_points = points_batch.size(1)

        pt_box_indices_list = []
        pt_cls_labels_list = []
        fg_pt_gt_boxes_list = []
        fg_pt_box_labels_list = [] if return_box_labels else None

        for points, gt_boxes, gt_labels in zip(points_batch, gt_boxes_list, gt_labels_list):
            u_boxes = gt_boxes.unsqueeze(0)
            u_points = points.unsqueeze(0)

            # 1, num_points
            pt_box_indices: torch.Tensor = _C.points_in_boxes_part(u_boxes, u_points)
            pt_box_indices = pt_box_indices.squeeze(0)
            box_fg_flag = pt_box_indices != -1

            # clone to not affect gt_boxes indexing below
            ext_gt_boxes = u_boxes.clone()
            ext_gt_boxes[..., 3:6] += ext_dims
            pt_ext_box_indices: torch.Tensor = _C.points_in_boxes_part(ext_gt_boxes, u_points)
            pt_ext_box_indices = pt_ext_box_indices.squeeze(0)
            ext_fg_flag = pt_ext_box_indices != -1

            pt_cls_labels = torch.zeros_like(pt_box_indices)
            if assign_type is AssignType.IGNORE_FLAG:
                fg_flag = box_fg_flag
                ignore_flag = torch.logical_xor(box_fg_flag, ext_fg_flag)
                pt_cls_labels[ignore_flag] = -1
            elif assign_type is AssignType.EXTEND_GT:
                pt_ext_box_indices[box_fg_flag] = pt_box_indices[box_fg_flag]
                fg_flag = ext_fg_flag
                pt_box_indices = pt_ext_box_indices
            else:
                raise NotImplementedError
            pt_box_indices_list.append(pt_box_indices)

            pt_cls_labels[fg_flag] = gt_labels[pt_box_indices[fg_flag]]
            pt_cls_labels_list.append(pt_cls_labels)

            bg_flag = pt_cls_labels == 0
            fg_flag = torch.logical_xor(fg_flag, fg_flag & bg_flag)

            fg_pt_box_indices = pt_box_indices[fg_flag]
            fg_pt_gt_boxes = gt_boxes[fg_pt_box_indices]
            fg_pt_gt_boxes_list.append(fg_pt_gt_boxes)

            if fg_pt_box_labels_list is not None:
                fg_pt_box_labels = u_boxes.new_zeros((num_points, 8))
                if fg_pt_gt_boxes.size(0) > 0:
                    fg_pt_box_labels[fg_flag] = self.box_coder.encode_torch(
                        points[fg_flag],
                        fg_pt_gt_boxes,
                        gt_labels[fg_pt_box_indices],
                        self.get_buffer("mean_size"),
                    )
                fg_pt_box_labels_list.append(fg_pt_box_labels)

        pt_box_indices = torch.stack(pt_box_indices_list)
        pt_cls_labels = torch.stack(pt_cls_labels_list)
        fg_pt_gt_boxes = torch.cat(fg_pt_gt_boxes_list)
        fg_pt_box_labels = (
            torch.stack(fg_pt_box_labels_list) if fg_pt_box_labels_list is not None else None
        )
        return Target(pt_box_indices, pt_cls_labels, fg_pt_gt_boxes, fg_pt_box_labels)
