# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from random import randint
from typing import Optional, Union

import torch

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


def sample_farthest_points(
    points: torch.Tensor,
    num_points: Union[int, list[int], torch.Tensor] = 50,
    lengths: Optional[torch.Tensor] = None,
    random_start: bool = False,
) -> torch.Tensor:
    """
    Iterative farthest point sampling algorithm [1] to subsample a set of
    K points from a given pointcloud. At each iteration, a point is selected
    which has the largest nearest neighbor distance to any of the
    already selected points.

    Farthest point sampling provides more uniform coverage of the input
    point cloud compared to uniform random sampling.

    [1] Charles R. Qi et al, "PointNet++: Deep Hierarchical Feature Learning
        on Point Sets in a Metric Space", NeurIPS 2017.

    Args:
        points: (B, P, D) array containing the batch of pointclouds
        lengths: (B,) number of points in each pointcloud (to support heterogeneous
            batches of pointclouds)
        num_points: samples required in each sampled point cloud (this is typically << P). If
            K is an int then the same number of samples are selected for each
            pointcloud in the batch. If K is a tensor is should be length (N,)
            giving the number of samples to select for each element in the batch
        random_start: bool, if True, a random point is selected as the starting point.

    Returns:
        indices: (B, K) array of selected indices. If the input K is a tensor, then the shape
            will be (B, max(K), D), and padded with -1 for batch elements where k_i < max(K).
    """
    batch_size, total_points = points.shape[:2]
    device = points.device
    # Validate inputs
    if lengths is None:
        lengths = torch.full((batch_size,), total_points, dtype=torch.int64, device=device)
    # TODO: support providing K as a ratio of the total number of points instead of as an int
    if isinstance(num_points, int):
        num_points = torch.full((batch_size,), num_points, dtype=torch.int64, device=device)
    elif isinstance(num_points, list):
        num_points = torch.tensor(num_points, dtype=torch.int64, device=device)

    # Check dtypes are correct and convert if necessary
    if points.dtype != torch.float32:
        points = points.float()
    if lengths.dtype != torch.int64:
        lengths = lengths.long()
    if num_points.dtype != torch.int64:
        num_points = num_points.long()

    # Generate the starting indices for sampling
    start_idxs = torch.zeros_like(lengths)
    if random_start:
        for b_idx in range(batch_size):
            # pyre-fixme[6]: For 1st param expected `int` but got `Tensor`.
            start_idxs[b_idx] = torch.randint(high=lengths[b_idx], size=(1,)).item()

    with torch.no_grad():
        indices = _C.sample_farthest_points(points, lengths, num_points, start_idxs)
    return indices


def sample_farthest_points_naive(
    points: torch.Tensor,
    num_points: Union[int, list[int], torch.Tensor] = 50,
    lengths: Optional[torch.Tensor] = None,
    random_start: bool = False,
) -> torch.Tensor:
    """
    Same Args/Returns as sample_farthest_points
    """
    batch_size, total_points = points.shape[:2]
    device = points.device

    # Validate inputs
    if lengths is None:
        lengths = torch.full((batch_size,), total_points, dtype=torch.int64, device=device)

    if lengths.shape[0] != batch_size:
        raise ValueError("points and lengths must have same batch dimension.")

    # TODO: support providing K as a ratio of the total number of points instead of as an int
    if isinstance(num_points, int):
        num_points = torch.full((batch_size,), num_points, dtype=torch.int64, device=device)
    elif isinstance(num_points, list):
        num_points = torch.tensor(num_points, dtype=torch.int64, device=device)
    if num_points.size(0) != batch_size:
        raise ValueError("K and points must have the same batch dimension")

    # Find max value of K
    max_k = torch.max(num_points)
    # List of selected indices from each batch element
    all_sampled_indices = []

    for n in range(batch_size):
        # Initialize an array for the sampled indices, shape: (max_K,)
        # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
        #  typing.Tuple[int, ...]]` but got `Tuple[Tensor]`.
        sample_idx_batch = torch.full((max_k,), fill_value=-1, dtype=torch.int64, device=device)

        # Initialize closest distances to inf, shape: (P,)
        # This will be updated at each iteration to track the closest distance of the
        # remaining points to any of the selected points
        # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
        #  typing.Tuple[int, ...]]` but got `Tuple[Tensor]`.
        closest_dists = points.new_full((lengths[n],), float("inf"), dtype=torch.float32)

        # Select a random point index and save it as the starting point
        selected_idx = randint(0, lengths[n] - 1) if random_start else 0
        sample_idx_batch[0] = selected_idx

        # If the pointcloud has fewer than K points then only iterate over the min
        # pyre-fixme[6]: For 1st param expected `SupportsRichComparisonT` but got
        #  `Tensor`.
        # pyre-fixme[6]: For 2nd param expected `SupportsRichComparisonT` but got
        #  `Tensor`.
        k_n = min(lengths[n], num_points[n])

        # Iteratively select points for a maximum of k_n
        for i in range(1, k_n):
            # Find the distance between the last selected point
            # and all the other points. If a point has already been selected
            # it's distance will be 0.0 so it will not be selected again as the max.
            dist = points[n, selected_idx, :] - points[n, : lengths[n], :]
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            dist_to_last_selected = (dist**2).sum(-1)  # (P - i)

            # If closer than currently saved distance to one of the selected
            # points, then updated closest_dists
            closest_dists = torch.min(dist_to_last_selected, closest_dists)  # (P - i)

            # The aim is to pick the point that has the largest
            # nearest neighbour distance to any of the already selected points
            selected_idx = torch.argmax(closest_dists)
            sample_idx_batch[i] = selected_idx

        # Add the list of points for this batch to the final list
        all_sampled_indices.append(sample_idx_batch)

    all_sampled_indices = torch.stack(all_sampled_indices)
    # Gather the points
    # Return (N, max_K, D) subsampled points and indices
    return all_sampled_indices
