/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>

// A chunk of work is blocksize-many points of P1.
// The number of potential chunks to do is N*(1+(P1-1)/blocksize)
// call (1+(P1-1)/blocksize) chunks_per_cloud
// These chunks are divided among the gridSize-many blocks.
// In block b, we work on chunks b, b+gridSize, b+2*gridSize etc .
// In chunk i, we work on cloud i/chunks_per_cloud on points starting from
// blocksize*(i%chunks_per_cloud).

template <typename scalar_t>
__global__ void BallQueryKernel(
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> p1,
    const at::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> p2,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lengths1,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lengths2,
    at::PackedTensorAccessor32<int32_t, 3, at::RestrictPtrTraits> idxs, const int K,
    const float radius2) {
  const int N = p1.size(0);
  const int chunks_per_cloud = (1 + (p1.size(1) - 1) / blockDim.x);
  const int chunks_to_do = N * chunks_per_cloud;
  const int D = p1.size(2);

  for (int chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    const int n = chunk / chunks_per_cloud;  // batch_index
    const int start_point = blockDim.x * (chunk % chunks_per_cloud);
    int i = start_point + threadIdx.x;

    // Check if point is valid in heterogeneous tensor
    if (i >= lengths1[n]) {
      continue;
    }

    // Iterate over points in p2 until desired count is reached or
    // all points have been considered
    for (int j = 0, count = 0; j < lengths2[n] && count < K; ++j) {
      // Calculate the distance between the points
      scalar_t dist2 = 0.0;
      for (int d = 0; d < D; ++d) {
        scalar_t diff = p1[n][i][d] - p2[n][j][d];
        dist2 += (diff * diff);
      }

      if (dist2 < radius2) {
        // If the point is within the radius
        // Set the value of the index to the point index
        idxs[n][i][count] = j;
        // increment the number of selected samples for the point i
        ++count;
      }
    }
  }
}

at::Tensor BallQueryCuda(const at::Tensor& p1,        // (N, P1, 3)
                         const at::Tensor& p2,        // (N, P2, 3)
                         const at::Tensor& lengths1,  // (N,)
                         const at::Tensor& lengths2,  // (N,)
                         int K, float radius) {
  // Check inputs are on the same device
  at::TensorArg p1_t{p1, "p1", 1}, p2_t{p2, "p2", 2};
  at::TensorArg lengths1_t{lengths1, "lengths1", 3}, lengths2_t{lengths2, "lengths2", 4};
  at::CheckedFrom c = "BallQueryCuda";
  at::checkAllSameGPU(c, {p1_t, p2_t, lengths1_t, lengths2_t});
  at::checkAllSameType(c, {p1_t, p2_t});
  TORCH_CHECK(p1.size(0) == p2.size(0), "Point sets must have the same batch dimension");
  TORCH_CHECK(p1.size(2) == p2.size(2), "Point sets must have the same last dimension");

  // Set the device for the kernel launch based on the device of p1
  at::cuda::CUDAGuard device_guard(p1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int N = p1.size(0);
  const int P1 = p1.size(1);
  const float radius2 = radius * radius;

  // Output tensor with indices of neighbors for each point in p1
  auto int_dtype = p1.options().dtype(at::kInt);
  auto idxs = at::full({N, P1, K}, -1, int_dtype);

  if (idxs.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return idxs;
  }

  constexpr size_t blocks = 256, threads = 256;

  AT_DISPATCH_FLOATING_TYPES(p1.scalar_type(), "ball_query_kernel_cuda", [&] {
    BallQueryKernel<<<blocks, threads, 0, stream>>>(
        p1.packed_accessor64<float, 3, at::RestrictPtrTraits>(),
        p2.packed_accessor64<float, 3, at::RestrictPtrTraits>(),
        lengths1.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
        lengths2.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
        idxs.packed_accessor32<int32_t, 3, at::RestrictPtrTraits>(), K, radius2);
  });
  AT_CUDA_CHECK(cudaGetLastError());
  return idxs;
}
