#include <ATen/ATen.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>

#include "utils/cuda_utils.hpp"
template <typename T>
__device__ inline void lidar_to_local_coords(T shift_x, T shift_y, T rz, T &local_x, T &local_y) {
  T cosa = cos(-rz), sina = sin(-rz);
  local_x = shift_x * cosa + shift_y * (-sina);
  local_y = shift_x * sina + shift_y * cosa;
}

template <typename T>
__device__ inline bool check_pt_in_box3d(const T *pt, const T *box3d, T &local_x, T &local_y) {
  // param pt: (x, y, z)
  // param box3d: (cx, cy, cz, x_size, y_size, z_size, rz) in LiDAR coordinate,
  // cz in the bottom center
  T x = pt[0], y = pt[1], z = pt[2];
  T cx = box3d[0], cy = box3d[1], cz = box3d[2];
  T x_size = box3d[3], y_size = box3d[4], z_size = box3d[5], rz = box3d[6];
  cz += z_size / 2.0;  // shift to the center since cz in box3d is the bottom center

  if (fabsf(z - cz) > z_size / 2.0) {
    return 0;
  }
  lidar_to_local_coords(x - cx, y - cy, rz, local_x, local_y);
  return (local_x > -x_size / 2.0) & (local_x < x_size / 2.0) & (local_y > -y_size / 2.0) &
         (local_y < y_size / 2.0);
}

template <typename T>
__global__ void points_in_boxes_part_kernel(const T *boxes, const T *pts,
                                            int64_t *box_idx_of_points, const int batch_size,
                                            const int num_boxes, const int num_points) {
  // params boxes: (B, N, 7) [x, y, z, x_size, y_size, z_size, rz] in LiDAR
  // coordinate, z is the bottom center, each box DO NOT overlaps params pts:
  // (B, npoints, 3) [x, y, z] in LiDAR coordinate params boxes_idx_of_points:
  // (B, npoints), default -1
  const int bs_idx = blockIdx.y;
  for (int pt_idx = blockIdx.x * blockDim.x + threadIdx.x; pt_idx < num_points;
       pt_idx += blockDim.x * gridDim.x) {
    if (bs_idx >= batch_size) return;

    boxes += bs_idx * num_boxes * 7;
    pts += bs_idx * num_points * 3 + pt_idx * 3;
    box_idx_of_points += bs_idx * num_points + pt_idx;

    T local_x = 0, local_y = 0;
    int cur_in_flag = 0;
    for (int k = 0; k < num_boxes; ++k) {
      cur_in_flag = check_pt_in_box3d(pts, boxes + k * 7, local_x, local_y);
      if (cur_in_flag) {
        box_idx_of_points[0] = k;
        break;
      }
    }
  }
}

at::Tensor PointsInBoxesPartCuda(const at::Tensor &boxes, const at::Tensor &points,
                                 const int batch_size, const int num_boxes, const int num_points) {
  // Find the box in which each point is, return box indices of points
  // params boxes: (B, N, 7) [x, y, z, x_size, y_size, z_size, rz] in LiDAR coordinate,
  // z is the bottom center, each box DO NOT overlaps params
  // pts: (B, npoints, 3) [x, y, z] in LiDAR coordinate params
  // boxes_idx_of_points: (B, npoints), default -1
  at::cuda::CUDAGuard device_guard(boxes.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto long_opts = points.options().dtype(at::kLong);
  auto box_idx_of_points = at::full({batch_size, num_points}, -1, long_opts);

  dim3 blocks(GetBlockSize(num_points), batch_size);
  dim3 threads(kThreadsPerBlock);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(boxes.scalar_type(), "points_in_boxes_part_kernel", [&] {
    points_in_boxes_part_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        boxes.data_ptr<scalar_t>(), points.data_ptr<scalar_t>(),
        box_idx_of_points.data_ptr<int64_t>(), batch_size, num_boxes, num_points);
  });
  AT_CUDA_CHECK(cudaGetLastError());

  return box_idx_of_points;
}
