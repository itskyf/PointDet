#pragma once
#include <ATen/ATen.h>

at::Tensor GroupPointsCuda(const at::Tensor points, const at::Tensor indices, const int batch_size,
                           const int feat_dims, const int num_feats, const int num_groups,
                           const int num_neighbors);

inline at::Tensor GroupPoints(const at::Tensor points, const at::Tensor indices) {
  const int batch_size = points.size(0);
  TORCH_CHECK(batch_size == indices.size(0),
              "points and indices must have the same batch dimension");
  const int feat_dims = points.size(1), num_feats = points.size(2);
  const int num_groups = indices.size(1), num_neighbors = indices.size(2);

  if (points.is_cuda() || indices.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    CHECK_CUDA(indices);
    return GroupPointsCuda(points.contiguous(), indices.contiguous(), batch_size, feat_dims,
                           num_feats, num_groups, num_neighbors);
#endif
  }
  AT_ERROR("Not compiled with GPU support");
}

at::Tensor GroupPointsBackwardCuda(const at::Tensor grad_grouped, const at::Tensor indices,
                                   const int batch_size, const int feat_dims, const int num_feats,
                                   const int num_groups, const int num_neighbors);

inline at::Tensor GroupPointsBackward(const at::Tensor grad_grouped, const at::Tensor indices,
                                      const int num_feats) {
  const int batch_size = grad_grouped.size(0), feat_dims = grad_grouped.size(1);
  const int num_groups = grad_grouped.size(2), num_neighbors = grad_grouped.size(3);
  if (grad_grouped.is_cuda() || indices.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(grad_grouped);
    CHECK_CUDA(indices);
    return GroupPointsBackwardCuda(grad_grouped.contiguous(), indices.contiguous(), batch_size,
                                   feat_dims, num_feats, num_groups, num_neighbors);
#endif
  }
  AT_ERROR("Not compiled with GPU support");
}
