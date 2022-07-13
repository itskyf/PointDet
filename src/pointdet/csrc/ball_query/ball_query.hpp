#pragma once
#include <ATen/ATen.h>

#include "utils/cuda_utils.hpp"

// CUDA implementation
at::Tensor BallQueryCuda(const at::Tensor& centroids, const at::Tensor& points,
                         const int batch_size, const int num_groups, const int num_neighbors,
                         const int total_points, const float radius);

// Implementation which is exposed
// Note: the backward pass reuses the KNearestNeighborBackward kernel
inline at::Tensor BallQuery(const at::Tensor& centroids, const at::Tensor& points,
                            const int num_neighbors, const float radius) {
  const int batch_size = centroids.size(0), num_groups = centroids.size(1);
  const int total_points = points.size(1);

  TORCH_CHECK(batch_size == points.size(0),
              "centroids and points must have the same batch dimention");
  TORCH_CHECK(centroids.size(2) == points.size(2) && centroids.size(2) == 3,
              "centroids and points must have the same 3 channels");

  const float radius2 = radius * radius;

  if (centroids.is_cuda() || points.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(centroids);
    CHECK_CUDA(points);
    return BallQueryCuda(centroids, points, batch_size, num_groups, num_neighbors, total_points,
                         radius2);
#endif
  }
  AT_ERROR("Not compiled with GPU support");
}
