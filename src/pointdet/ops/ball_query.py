# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from pointdet import _C


class _ball_query(Function):
    """
    Torch autograd Function wrapper for Ball Query C++/CUDA implementations.
    """

    @staticmethod
    def forward(
        ctx,
        centers: torch.Tensor,
        points: torch.Tensor,
        num_samples: int,
        radius: float,
        lengths1: torch.Tensor,
        lengths2: torch.Tensor,
    ):
        """
        Arguments defintions the same as in the ball_query function
        """
        idx = _C.ball_query(centers, points, lengths1, lengths2, num_samples, radius)
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_idx):
        return None, None, None, None


def ball_query(
    centers: torch.Tensor,
    points: torch.Tensor,
    num_samples: int = 500,
    radius: float = 0.2,
    lengths1: Optional[torch.Tensor] = None,
    lengths2: Optional[torch.Tensor] = None,
):
    """
    Ball Query is an alternative to KNN. It can be
    used to find all points in p2 that are within a specified radius
    to the query point in p1 (with an upper limit of K neighbors).

    The neighbors returned are not necssarily the *nearest* to the
    point in p1, just the first K values in p2 which are within the
    specified radius.

    This method is faster than kNN when there are large numbers of points
    in p2 and the ordering of neighbors is not important compared to the
    distance being within the radius threshold.

    "Ball query’s local neighborhood guarantees a fixed region scale thus
    making local region features more generalizable across space, which is
    preferred for tasks requiring local pattern recognition
    (e.g. semantic point labeling)" [1].

    [1] Charles R. Qi et al, "PointNet++: Deep Hierarchical Feature Learning
        on Point Sets in a Metric Space", NeurIPS 2017.

    Args:
        p1: Tensor of shape (N, P1, D) giving a batch of N point clouds, each containing
            up to P1 points of dimension D. These represent the centers of the ball queries.
        p2: Tensor of shape (N, P2, D) giving a batch of N point clouds, each containing
            up to P2 points of dimension D.
        num_samples: upper bound on the number of samples to take within the radius
        radius: the radius around each point within which the neighbors need to be located
        lengths1: LongTensor of shape (N,) of values in the range [0, P1], giving the length
            of each pointcloud in p1. Or None to indicate that every cloud has length P1.
        lengths2: LongTensor of shape (N,) of values in the range [0, P2], giving the length
            of each pointcloud in p2. Or None to indicate that every cloud has length P2.

    Returns:
        dists: Tensor of shape (N, P1, K) giving the squared distances to
            the neighbors. This is padded with zeros both where a cloud in p2
            has fewer than S points and where a cloud in p1 has fewer than P1 points
            and also if there are fewer than K points which satisfy the radius threshold.

        idx: LongTensor (N, P1, K) giving the indices of the S neighbors in p2 for points in p1.
            Concretely, if `p1_idx[n, i, k] = j` then `p2[n, j]` is the k-th
            neighbor to `p1[n, i]` in `p2[n]`. This is padded with -1 both where a cloud
            in p2 has fewer than S points and where a cloud in p1 has fewer than P1 points
            and also if there are fewer than K points which satisfy the radius threshold.
    """
    if centers.size(0) != points.size(0):
        raise ValueError("pts1 and pts2 must have the same batch dimension")
    if centers.size(2) != points.size(2):
        raise ValueError("pts1 and pts2 must have the same point dimension")

    num_centers = centers.size(1)
    num_points = points.size(1)
    N = centers.size(0)

    device = centers.device
    if lengths1 is None:
        lengths1 = torch.full((N,), num_centers, dtype=torch.int64, device=device)
    if lengths2 is None:
        lengths2 = torch.full((N,), num_points, dtype=torch.int64, device=device)

    # pyre-fixme[16]: `_ball_query` has no attribute `apply`.
    idx = _ball_query.apply(centers, points, num_samples, radius, lengths1, lengths2)
    return idx
