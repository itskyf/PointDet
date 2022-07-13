import torch
from torch.autograd import Function
from torch.autograd.function import FunctionCtx

from pointdet import _C


def centroid_aware(cls_features: torch.Tensor, num_points: int) -> torch.Tensor:
    """
    Args:
        cls_features (B, N, num_classes)
    Returns:
        indices (B, num_points)
    """
    cls_features_max = cls_features.max(dim=-1)[0]
    score_pred = torch.sigmoid(cls_features_max)
    out = torch.topk(score_pred, num_points, dim=-1)
    return out.indices


class FurthestPointSampling(Function):
    """Uses iterative furthest point sampling to select a set of features whose
    corresponding points have the furthest distance."""

    @staticmethod
    def forward(ctx: FunctionCtx, points: torch.Tensor, num_points: int):
        """
        Args:
            points_xyz (torch.Tensor): (B, N, 3) where N > num_points.
            num_points (int): Number of points in the sampled set.

        Returns:
            torch.Tensor: (B, num_points) indices of the sampled points.
        """
        indices = _C.sample_farthest_points(points, num_points)
        ctx.mark_non_differentiable(indices)
        return indices

    @staticmethod
    def backward(ctx, grad_indices):
        return None, None


sample_farthest_points = FurthestPointSampling.apply
