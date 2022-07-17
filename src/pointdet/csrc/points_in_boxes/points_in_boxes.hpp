#pragma once
#include <ATen/ATen.h>

// TODO CPU implementation
//
at::Tensor PointsInBoxesPartCuda(const at::Tensor& boxes, const at::Tensor& points,
                                 const int batch_size, const int num_boxes, const int num_points);

inline at::Tensor PointsInBoxesPart(const at::Tensor& boxes, const at::Tensor& points) {
  const int batch_size = boxes.size(0), num_boxes = boxes.size(1);
  const int num_points = points.size(1);

  TORCH_CHECK(batch_size == points.size(0), "points and boxes should have the same batch size");
  TORCH_CHECK(boxes.size(2) == 7, "boxes dimension should be 7");
  TORCH_CHECK(points.size(2) == 3, "points dimension should be 3");

  if (boxes.is_cuda() || points.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(boxes);
    CHECK_CUDA(points);
    return PointsInBoxesPartCuda(boxes.contiguous(), points.contiguous(), batch_size, num_boxes,
                                 num_points);
#endif
  }
  AT_ERROR("Not compiled with GPU support");
}
