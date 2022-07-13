#pragma once
#include <ATen/ATen.h>

#include <tuple>

#include "utils/cuda_utils.hpp"

at::Tensor FarthestPointSamplingCuda(const at::Tensor& points, const int batch_size,
                                     const int total_points, const int num_points);

inline at::Tensor FarthestPointSampling(const at::Tensor& points, const int num_points) {
  const int batch_size = points.size(0), total_points = points.size(1);
  if (points.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    return FarthestPointSamplingCuda(points, batch_size, total_points, num_points);
#endif
  }
  AT_ERROR("Not compiled with GPU support");
}
