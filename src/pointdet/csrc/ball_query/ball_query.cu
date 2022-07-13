#include <ATen/ATen.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>

#include "utils/cuda_utils.hpp"

template <typename T>
__global__ void ball_query_kernel(const T* p_centroids, const T* p_points, int* p_indices,
                                  const int batch_size, const int num_groups,
                                  const int num_neighbors, const int total_points,
                                  const float radius2) {
  const int bs_idx = blockIdx.y;
  for (int pt_idx = blockIdx.x * blockDim.x + threadIdx.x; pt_idx < num_groups;
       pt_idx += blockDim.x * gridDim.x) {
    if (bs_idx >= batch_size) {
      return;
    }

    p_centroids += bs_idx * num_groups * 3 + pt_idx * 3;
    p_points += bs_idx * total_points * 3;
    p_indices += bs_idx * num_groups * num_neighbors + pt_idx * num_neighbors;

    const T c_x = p_centroids[0], c_y = p_centroids[1], c_z = p_centroids[2];

    int cnt = 0;
    for (int k = 0; k < total_points; ++k) {
      const T x = p_points[k * 3 + 0], y = p_points[k * 3 + 1], z = p_points[k * 3 + 2];
      const T d2 = (c_x - x) * (c_x - x) + (c_y - y) * (c_y - y) + (c_z - z) * (c_z - z);
      if (d2 < radius2) {
        if (cnt == 0) {
          for (int l = 0; l < num_neighbors; ++l) {
            p_indices[l] = k;
          }
        }
        p_indices[cnt] = k;
        if (++cnt >= num_neighbors) {
          break;
        }
      }
    }
  }
}
at::Tensor BallQueryCuda(const at::Tensor& centroids, const at::Tensor& points,
                         const int batch_size, const int num_groups, const int num_neighbors,
                         const int total_points, const float radius2) {
  at::cuda::CUDAGuard device_guard(centroids.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(GetBlockSize(num_groups), batch_size);
  dim3 threads(kThreadsPerBlock);

  auto int_opts = points.options().dtype(at::kInt);
  at::Tensor indices = at::zeros({batch_size, num_groups, num_neighbors}, int_opts);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(centroids.scalar_type(), "ball_query_kernel", [&] {
    ball_query_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        centroids.data_ptr<scalar_t>(), points.data_ptr<scalar_t>(), indices.data_ptr<int>(),
        batch_size, num_groups, num_neighbors, total_points, radius2);
  });

  AT_CUDA_CHECK(cudaGetLastError());

  return indices;
}
