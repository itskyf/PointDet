#include <ATen/ATen.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>

__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i, const int idx1,
                         const int idx2) {
  const float v1 = dists[idx1], v2 = dists[idx2];
  const int i1 = dists_i[idx1], i2 = dists_i[idx2];
  dists[idx1] = std::max(v1, v2);
  dists_i[idx1] = v2 > v1 ? i2 : i1;
}

template <unsigned int block_size>
__global__ void farthest_point_sampling_kernel(const float *__restrict__ points,
                                               float *__restrict__ temp, int *__restrict__ indices,
                                               int batch_size, int total_points, int num_points) {
  if (num_points <= 0) {
    return;
  }
  __shared__ float dists[block_size];
  __shared__ int dists_i[block_size];

  const int batch_idx = blockIdx.x;
  points += batch_idx * total_points * 3;
  temp += batch_idx * total_points;
  indices += batch_idx * num_points;

  int tid = threadIdx.x;
  const int stride = block_size;

  int old = 0;
  if (threadIdx.x == 0) {
    indices[0] = old;
  }

  __syncthreads();
  for (int j = 1; j < num_points; ++j) {
    int besti = 0;
    float best = -1;
    float x1 = points[old * 3 + 0];
    float y1 = points[old * 3 + 1];
    float z1 = points[old * 3 + 2];
    for (int k = tid; k < total_points; k += stride) {
      float x2, y2, z2;
      x2 = points[k * 3 + 0];
      y2 = points[k * 3 + 1];
      z2 = points[k * 3 + 2];

      float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
      float d2 = fmin(d, temp[k]);
      temp[k] = d2;
      besti = d2 > best ? k : besti;
      best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

#pragma unroll
    for (int block_size_thres = 1024; block_size_thres >= 2; block_size_thres >>= 1) {
      const int tid_thres = block_size_thres / 2;
      if (block_size >= block_size_thres && tid < tid_thres) {
        __update(dists, dists_i, tid, tid + tid_thres);
      }
      __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0) {
      indices[j] = old;
    }
  }
}

at::Tensor FarthestPointSamplingCuda(const at::Tensor &points, const int batch_size,
                                     const int total_points, const int num_points) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto pt_ops = points.options().dtype(at::kInt);
  at::Tensor indices = at::zeros({batch_size, num_points}, pt_ops.dtype(at::kInt));
  at::Tensor tmp_tensor = at::full({batch_size, total_points}, 1e10, pt_ops);

  const float *p_points = points.data_ptr<float>();
  float *p_tmp = tmp_tensor.data_ptr<float>();
  int *p_indices = indices.data_ptr<int>();

  const int pow_2 = std::log(static_cast<double>(total_points)) / std::log(2.0);
  const int n_threads = std::max(std::min(1 << pow_2, 1024), 1);
  switch (n_threads) {
    case 1024:
      farthest_point_sampling_kernel<1024><<<batch_size, n_threads, 0, stream>>>(
          p_points, p_tmp, p_indices, batch_size, total_points, num_points);
      break;
    case 512:
      farthest_point_sampling_kernel<512><<<batch_size, n_threads, 0, stream>>>(
          p_points, p_tmp, p_indices, batch_size, total_points, num_points);
      break;
    case 256:
      farthest_point_sampling_kernel<256><<<batch_size, n_threads, 0, stream>>>(
          p_points, p_tmp, p_indices, batch_size, total_points, num_points);
      break;
    case 128:
      farthest_point_sampling_kernel<128><<<batch_size, n_threads, 0, stream>>>(
          p_points, p_tmp, p_indices, batch_size, total_points, num_points);
      break;
    case 64:
      farthest_point_sampling_kernel<64><<<batch_size, n_threads, 0, stream>>>(
          p_points, p_tmp, p_indices, batch_size, total_points, num_points);
      break;
    case 32:
      farthest_point_sampling_kernel<32><<<batch_size, n_threads, 0, stream>>>(
          p_points, p_tmp, p_indices, batch_size, total_points, num_points);
      break;
    case 16:
      farthest_point_sampling_kernel<16><<<batch_size, n_threads, 0, stream>>>(
          p_points, p_tmp, p_indices, batch_size, total_points, num_points);
      break;
    case 8:
      farthest_point_sampling_kernel<8><<<batch_size, n_threads, 0, stream>>>(
          p_points, p_tmp, p_indices, batch_size, total_points, num_points);
      break;
    case 4:
      farthest_point_sampling_kernel<4><<<batch_size, n_threads, 0, stream>>>(
          p_points, p_tmp, p_indices, batch_size, total_points, num_points);
      break;
    case 2:
      farthest_point_sampling_kernel<2><<<batch_size, n_threads, 0, stream>>>(
          p_points, p_tmp, p_indices, batch_size, total_points, num_points);
      break;
    case 1:
      farthest_point_sampling_kernel<1><<<batch_size, n_threads, 0, stream>>>(
          p_points, p_tmp, p_indices, batch_size, total_points, num_points);
      break;
    default:
      farthest_point_sampling_kernel<512><<<batch_size, n_threads, 0, stream>>>(
          p_points, p_tmp, p_indices, batch_size, total_points, num_points);
  }

  AT_CUDA_CHECK(cudaGetLastError());
  return indices;
}
