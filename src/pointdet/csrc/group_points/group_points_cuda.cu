#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>

#include "utils/cuda_utils.hpp"

template <typename T>
__global__ void group_points_kernel(const T *points, const int *__restrict__ idx, T *out,
                                    const int batch_size, const int feat_dims, const int num_feats,
                                    const int num_groups, const int num_neighbors) {
  // points: (B, C, N)
  // idx: (B, num_groups, num_neighbors)
  // output:
  //      out: (B, C, num_groups, num_neighbors)
  const int totals = num_groups * num_neighbors;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < totals; i += blockDim.x * gridDim.x) {
    if (blockIdx.z >= batch_size || blockIdx.y >= feat_dims) {
      return;
    }
    const int pt_idx = i / num_neighbors;
    const int idk_idx = pt_idx * num_neighbors + i % num_neighbors;  // TODO better naming
    const int idk2_idx = blockIdx.z * feat_dims + blockIdx.y;        // TODO better naming

    idx += blockIdx.z * totals + idk_idx;
    const int in_idx = idk2_idx * num_feats + idx[0];
    const int out_idx = idk2_idx * totals + idk_idx;
    out[out_idx] = points[in_idx];
  }
}

at::Tensor GroupPointsCuda(const at::Tensor points, const at::Tensor indices, const int batch_size,
                           const int feat_dims, const int num_feats, const int num_groups,
                           const int num_neighbors) {
  // points: (B, feat_dims, num_feats)
  // indices: (B, num_groups, num_neighbors)
  // output: (B, C, num_groups, num_neighbors)
  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(GetBlockSize(num_groups * num_neighbors), feat_dims, batch_size);
  dim3 threads(kThreadsPerBlock);
  auto output = at::empty({batch_size, feat_dims, num_groups, num_neighbors}, points.options());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(points.scalar_type(), "group_points_kernel", [&] {
    group_points_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        points.data_ptr<scalar_t>(), indices.data_ptr<int>(), output.data_ptr<scalar_t>(),
        batch_size, feat_dims, num_feats, num_groups, num_neighbors);
  });
  AT_CUDA_CHECK(cudaGetLastError());

  return output;
}

template <typename T>
__global__ void group_points_backward_kernel(const T *grad_out, const int *__restrict__ idx,
                                             T *grad_points, const int batch_size,
                                             const int feat_dims, const int num_feats,
                                             const int num_groups, const int num_neighbors) {
  // grad_out: (B, C, npoints, nsample)
  // idx: (B, npoints, nsample)
  // output:
  //      grad_points: (B, C, N)
  const int totals = num_groups * num_neighbors;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < totals; i += blockDim.x * gridDim.x) {
    if (blockIdx.z >= batch_size || blockIdx.y >= feat_dims) {
      return;
    }
    const int pt_idx = i / num_neighbors;
    const int idk_idx = pt_idx * num_neighbors + i % num_neighbors;  // TOOD better naming
    const int idk2_idx = blockIdx.z * feat_dims + blockIdx.y;        // TODO better naming

    grad_out += idk2_idx * totals + idk_idx;
    idx += blockIdx.z * totals + idk_idx;
    atomicAdd(grad_points + idk2_idx * num_feats + idx[0], grad_out[0]);
  }
}

at::Tensor GroupPointsBackwardCuda(const at::Tensor grad_out, const at::Tensor indices,
                                   const int batch_size, const int feat_dims, const int num_feats,
                                   const int num_groups, const int num_neighbors) {
  // grad_out: (B, C, npoints, nsample)
  // idx: (B, npoints, nsample)
  // output:
  //      grad_points: (B, C, N)

  at::cuda::CUDAGuard device_guard(grad_out.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(GetBlockSize(num_groups * num_neighbors), feat_dims, batch_size);
  dim3 threads(kThreadsPerBlock);
  auto grad_points = at::zeros({batch_size, feat_dims, num_feats}, grad_out.options());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_out.scalar_type(), "group_points_backward_kernel", [&] {
    group_points_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        grad_out.data_ptr<scalar_t>(), indices.data_ptr<int>(), grad_points.data_ptr<scalar_t>(),
        batch_size, feat_dims, num_feats, num_groups, num_neighbors);
  });
  AT_CUDA_CHECK(cudaGetLastError());

  return grad_points;
}
