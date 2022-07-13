from dataclasses import dataclass

import torch

from ..typing import PointCloud


@dataclass
class PCDBatch:
    points_list: list[torch.Tensor]
    gt_boxes_list: list[torch.Tensor]
    gt_labels_list: list[torch.Tensor]

    def to(self, device: torch.device):
        points_list = [points.to(device) for points in self.points_list]
        boxes_list = [boxes.to(device) for boxes in self.gt_boxes_list]
        labels_list = [labels.to(device) for labels in self.gt_labels_list]
        return PCDBatch(points_list, boxes_list, labels_list)


def form_pcd_batch(pcd_list: list[PointCloud]) -> PCDBatch:
    points_list = [pcd.points.tensor for pcd in pcd_list]
    boxes_list = [pcd.gt_bboxes_3d.tensor for pcd in pcd_list]
    labels_list = [torch.from_numpy(pcd.gt_labels_3d) for pcd in pcd_list]
    return PCDBatch(points_list, boxes_list, labels_list)
