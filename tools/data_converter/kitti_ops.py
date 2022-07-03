import numba
import numpy as np
from numpy.typing import NDArray

from pointdet.core.bbox import box_np_ops
from pointdet.core.bbox.structures.utils import limit_period


@numba.njit
def corner_to_surfaces_3d_jit(corners):
    """Convert 3d box corners from corner function above to surfaces that
    normal vectors all direct to internal.
    Args:
        corners (np.ndarray): 3d box corners with the shape of (N, 8, 3).
    Returns:
        np.ndarray: Surfaces with the shape of (N, 6, 4, 3).
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    num_boxes = corners.shape[0]
    surfaces = np.zeros((num_boxes, 6, 4, 3), dtype=corners.dtype)
    corner_idxes = np.array(
        [[0, 1, 2, 3], [7, 6, 5, 4], [0, 3, 7, 4], [1, 5, 6, 2], [0, 4, 5, 1], [3, 2, 6, 7]]
    )
    for i in range(num_boxes):
        for j in range(6):
            for k in range(4):
                surfaces[i, j, k] = corners[i, corner_idxes[j, k]]
    return surfaces


def box_camera_to_lidar(data, r_rect, velo2cam):
    """Convert boxes in camera coordinate to lidar coordinate.
    Note:
        This function is for KITTI only.
    Args:
        data (np.ndarray, shape=[N, 7]): Boxes in camera coordinate.
        r_rect (np.ndarray, shape=[4, 4]): Matrix to project points in
            specific camera coordinate (e.g. CAM2) to CAM0.
        velo2cam (np.ndarray, shape=[4, 4]): Matrix to project points in
            camera coordinate to lidar coordinate.
    Returns:
        np.ndarray, shape=[N, 3]: Boxes in lidar coordinate.
    """
    x_size, y_size, z_size = data[:, 3:4], data[:, 4:5], data[:, 5:6]
    xyz_lidar = camera_to_lidar(data[:, 0:3], r_rect, velo2cam)
    # yaw and dims also needs to be converted
    r_new = limit_period(-data[:, 6:7] - np.pi / 2, period=np.pi * 2)
    return np.concatenate((xyz_lidar, x_size, z_size, y_size, r_new), axis=1)


def camera_to_lidar(points, r_rect, velo2cam):
    """Convert points in camera coordinate to lidar coordinate.
    Args:
        points (np.ndarray, shape=[N, 3]): Points in camera coordinate.
        r_rect (np.ndarray, shape=[4, 4]): Matrix to project points in
            specific camera coordinate (e.g. CAM2) to CAM0.
        velo2cam (np.ndarray, shape=[4, 4]): Matrix to project points in
            camera coordinate to lidar coordinate.
    Returns:
        np.ndarray, shape=[N, 3]: Points in lidar coordinate.
    """
    points_shape = list(points.shape[0:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    lidar_points = points @ np.linalg.inv((r_rect @ velo2cam).T)
    return lidar_points[..., :3]


def get_frustum(
    bbox_img: tuple[int, int, int, int],
    cam_intrinsic: NDArray[np.float32],
    near_clip=0.001,
    far_clip=100.0,
):
    """Get frustum corners in camera coordinates.
    Args:
        bbox_image (list[int]): box in image coordinates.
        C (np.ndarray): Intrinsics.
        near_clip (float, optional): Nearest distance of frustum.
            Defaults to 0.001.
        far_clip (float, optional): Farthest distance of frustum.
            Defaults to 100.
    Returns:
        np.ndarray, shape=[8, 3]: coordinates of frustum corners.
    """
    fku = cam_intrinsic[0, 0]
    fkv = -cam_intrinsic[1, 1]
    u0v0 = cam_intrinsic[0:2, 2]
    z_points = np.array([near_clip] * 4 + [far_clip] * 4, dtype=cam_intrinsic.dtype)[:, np.newaxis]
    box_corners = np.array(
        [
            [bbox_img[0], bbox_img[1]],
            [bbox_img[0], bbox_img[3]],
            [bbox_img[2], bbox_img[3]],
            [bbox_img[2], bbox_img[1]],
        ],
        dtype=cam_intrinsic.dtype,
    )
    lhs = box_corners - u0v0
    near_box_corners = lhs / np.array(
        [fku / near_clip, -fkv / near_clip], dtype=cam_intrinsic.dtype
    )
    far_box_corners = lhs / np.array([fku / far_clip, -fkv / far_clip], dtype=cam_intrinsic.dtype)
    ret_xy = np.concatenate((near_box_corners, far_box_corners))  # [8, 2]
    return np.concatenate((ret_xy, z_points), axis=1)  # ret_xyz


def projection_matrix_to_crt_kitti(proj: NDArray[np.float32]):
    """Split projection matrix of KITTI.
    P = C @ [R|T]
    C is upper triangular matrix, so we need to inverse CR and use QR
    stable for all kitti camera projection matrix.
    Args:
        proj (p.array, shape=[4, 4]): Intrinsics of camera.
    Returns:
        tuple[np.ndarray]: Splited matrix of C, R and T.
    """

    CR = proj[0:3, 0:3]
    CT = proj[0:3, 3]
    RinvCinv = np.linalg.inv(CR)
    Rinv, Cinv = np.linalg.qr(RinvCinv)
    C = np.linalg.inv(Cinv)
    R = np.linalg.inv(Rinv)
    T = Cinv @ CT
    return C, R, T


def remove_outside_points(
    points: NDArray[np.float32],
    image_shape: NDArray[np.int32],
    rect: NDArray[np.float32],
    trv2c: NDArray[np.float32],
    P2: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Remove points which are outside of image.
    Args:
        points (np.ndarray, shape=[N, 3+dims]): Total points.
        rect (np.ndarray, shape=[4, 4]): Matrix to project points in
            specific camera coordinate (e.g. CAM2) to CAM0.
        trv2c (np.ndarray, shape=[4, 4]): Matrix to project points in
            camera coordinate to lidar coordinate.
        P2 (p.array, shape=[4, 4]): Intrinsics of Camera2.
        image_shape (list[int]): Shape of image.
    Returns:
        np.ndarray, shape=[N, 3+dims]: Filtered points.
    """
    # 5x faster than remove_outside_points_v1(2ms vs 10ms)
    C, R, T = projection_matrix_to_crt_kitti(P2)
    image_bbox = (0, 0, image_shape[1], image_shape[0])
    frustum = get_frustum(image_bbox, C) - T
    frustum = np.linalg.inv(R) @ frustum.T
    frustum = camera_to_lidar(frustum.T, rect, trv2c)
    frustum_surfaces = corner_to_surfaces_3d_jit(frustum[np.newaxis, ...])
    indices = box_np_ops.points_in_convex_polygon_3d_jit(points[:, :3], frustum_surfaces)
    return points[indices.reshape(-1)]
