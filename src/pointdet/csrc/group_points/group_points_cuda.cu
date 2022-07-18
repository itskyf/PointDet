#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>

#include "utils/cuda_utils.hpp"

template <typename T>
__global__ void group_points_kernel(const T *p_points, const int *__restrict__ p_indices, T *p_out,
                                    int batch_size, int n_channels, int num_feats, int num_groups,
                                    int num_neighbors) {
  const int bs_idx = blockIdx.z, c_idx = blockIdx.y;
  const int totals = num_groups * num_neighbors;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < totals; i += blockDim.x * gridDim.x) {
    if (bs_idx >= batch_size || c_idx >= n_channels) {
      return;
    }

    const int pt_idx = i / num_neighbors, sample_idx = i % num_neighbors;
    p_indices += bs_idx * totals + pt_idx * num_neighbors + sample_idx;

    const int in_idx = bs_idx * n_channels * num_feats + c_idx * num_feats + p_indices[0];
    const int out_idx =
        (bs_idx * n_channels + c_idx) * totals + pt_idx * num_neighbors + sample_idx;
    p_out[out_idx] = p_points[in_idx];
  }
}

at::Tensor GroupPointsCuda(const at::Tensor points, const at::Tensor indices, const int batch_size,
                           const int n_channels, const int num_feats, const int num_groups,
                           const int num_neighbors) {
  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 blocks(GetBlockSize(num_groups * num_neighbors), n_channels, batch_size);
  dim3 threads(kThreadsPerBlock);
  auto output = at::empty({batch_size, n_channels, num_groups, num_neighbors}, points.options());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(points.scalar_type(), "group_points_kernel", [&] {
    group_points_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        points.data_ptr<scalar_t>(), indices.data_ptr<int>(), output.data_ptr<scalar_t>(),
        batch_size, n_channels, num_feats, num_groups, num_neighbors);
  });
  AT_CUDA_CHECK(cudaGetLastError());

  return output;
}

template <typename T>
__global__ void group_points_backward_kernel(const T *p_grad_grouped,
                                             const int *__restrict__ p_indices, T *p_grad_feats,
                                             const int batch_size, const int n_channels,
                                             const int num_feats, const int num_groups,
                                             const int num_neighbors) {
  const int bs_idx = blockIdx.z, c_idx = blockIdx.y;
  const int totals = num_groups * num_neighbors;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < totals; i += blockDim.x * gridDim.x) {
    if (bs_idx >= batch_size || c_idx >= n_channels) {
      return;
    }
    const int pt_idx = i / num_neighbors, sample_idx = i % num_neighbors;
    p_grad_grouped += (bs_idx * n_channels + c_idx) * totals + pt_idx * num_neighbors + sample_idx;
    p_indices += bs_idx * totals + pt_idx * num_neighbors + sample_idx;
    atomicAdd(p_grad_feats + (bs_idx * n_channels + c_idx) * num_feats + p_indices[0],
              p_grad_grouped[0]);
  }
}

at::Tensor GroupPointsBackwardCuda(const at::Tensor grad_grouped, const at::Tensor indices,
                                   const int batch_size, const int n_channels, const int num_feats,
                                   const int num_groups, const int num_neighbors) {
  at::cuda::CUDAGuard device_guard(grad_grouped.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(GetBlockSize(num_groups * num_neighbors), n_channels, batch_size);
  dim3 threads(kThreadsPerBlock);
  auto grad_feats = at::zeros({batch_size, n_channels, num_feats}, grad_grouped.options());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_grouped.scalar_type(), "group_points_backward_kernel", [&] {
        group_points_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            grad_grouped.data_ptr<scalar_t>(), indices.data_ptr<int>(),
            grad_feats.data_ptr<scalar_t>(), batch_size, n_channels, num_feats, num_groups,
            num_neighbors);
      });
  AT_CUDA_CHECK(cudaGetLastError());

  return grad_feats;
}
