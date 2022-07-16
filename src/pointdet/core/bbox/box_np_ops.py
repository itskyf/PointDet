from typing import Optional, Union

import numba
import numpy as np
from numpy.typing import NDArray

from .structures.utils import rotation_3d_in_axis


@numba.njit
def box2d_to_corner_jit(boxes: NDArray[np.float32]):
    """Convert box2d to corner.
    Args:
        boxes (np.ndarray, shape=[N, 5]): Boxes2d with rotation.
    Returns:
        box_corners (np.ndarray, shape=[N, 4, 2]): Box corners.
    """
    num_box = boxes.shape[0]
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners = boxes.reshape(num_box, 1, 5)[:, :, 2:4] * np.expand_dims(corners_norm, 0)
    rot_mat_t = np.zeros((2, 2), dtype=boxes.dtype)
    box_corners = np.zeros((num_box, 4, 2), dtype=boxes.dtype)
    for i in range(num_box):
        rot_sin = np.sin(boxes[i, -1])
        rot_cos = np.cos(boxes[i, -1])
        rot_mat_t[0, 0] = rot_cos
        rot_mat_t[0, 1] = rot_sin
        rot_mat_t[1, 0] = -rot_sin
        rot_mat_t[1, 1] = rot_cos
        box_corners[i] = corners[i] @ rot_mat_t + boxes[i, :2]
    return box_corners


def center_to_corner_box2d(
    centers: NDArray[np.float32],
    dims: NDArray[np.float32],
    angles: Optional[NDArray[np.float32]] = None,
    origin: float = 0.5,
):
    """Convert kitti locations, dimensions and angles to corners.
    format: center(xy), dims(xy), angles(counterclockwise when positive)
    Args:
        centers (np.ndarray): Locations in kitti label file with shape (N, 2).
        dims (np.ndarray): Dimensions in kitti label file with shape (N, 2).
        angles (np.ndarray, optional): Rotation_y in kitti label file with
            shape (N). Defaults to None.
        origin (list or array or float, optional): origin point relate to
            smallest point. Defaults to 0.5.
    Returns:
        np.ndarray: Corners with the shape of (N, 4, 2).
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles)
    return corners + centers.reshape(-1, 1, 2)


def center_to_corner_box3d(
    centers: NDArray[np.float32],
    dims: NDArray[np.float32],
    angles: Optional[NDArray[np.float32]] = None,
    origin: tuple[float, float, float] = (0.5, 1.0, 0.5),
    axis: int = 1,
):
    """Convert kitti locations, dimensions and angles to corners.
    Args:
        centers (np.ndarray): Locations in kitti label file with shape (N, 3).
        dims (np.ndarray): Dimensions in kitti label file with shape (N, 3).
        angles (np.ndarray, optional): Rotation_y in kitti label file with
            shape (N). Defaults to None.
        origin (list or array or float, optional): Origin point relate to
            smallest point. Use (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0)
            in lidar. Defaults to (0.5, 1.0, 0.5).
        axis (int, optional): Rotation axis. 1 for camera and 2 for lidar.
            Defaults to 1.
    Returns:
        np.ndarray: Corners with the shape of (N, 8, 3).
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(lwh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    return corners + centers.reshape(-1, 1, 3)


def corners_nd(
    dims: NDArray[np.float32], origin: Union[float, tuple[float, ...]]
) -> NDArray[np.float32]:
    """Generate relative box corners based on length per dim and origin point.
    Args:
        dims (np.ndarray, shape=[N, ndim]): Array of length per dim
        origin (list or array or float, optional): origin point relate to
            smallest point. Defaults to 0.5
    Returns:
        np.ndarray, shape=[N, 2 ** ndim, ndim]: Returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1.
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(np.unravel_index(np.arange(2**ndim), [2] * ndim), axis=1)
    corners_norm = corners_norm.astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm -= np.array(origin, dtype=dims.dtype)
    return dims.reshape(-1, 1, ndim) * corners_norm.reshape(1, 2**ndim, ndim)  # corners


@numba.njit
def corner_to_standup_nd_jit(boxes_corner: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert boxes_corner to aligned (min-max) boxes.

    Args:
        boxes_corner (np.ndarray, shape=[N, 2**dim, dim]): Boxes corners.

    Returns:
        np.ndarray, shape=[N, dim*2]: Aligned (min-max) boxes.
    """
    num_boxes = boxes_corner.shape[0]
    ndim = boxes_corner.shape[-1]
    result = np.zeros((num_boxes, ndim * 2), dtype=boxes_corner.dtype)
    for i in range(num_boxes):
        for j in range(ndim):
            result[i, j] = np.min(boxes_corner[i, :, j])
        for j in range(ndim):
            result[i, j + ndim] = np.max(boxes_corner[i, :, j])
    return result


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


def corner_to_surfaces_3d(corners):
    """convert 3d box corners from corner function above to surfaces that
    normal vectors all direct to internal.
    Args:
        corners (np.ndarray): 3D box corners with shape of (N, 8, 3).
    Returns:
        np.ndarray: Surfaces with the shape of (N, 6, 4, 3).
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    surfaces = np.array(
        [
            [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
            [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
            [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
            [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
            [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
            [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
        ]
    )
    return surfaces.transpose([2, 0, 1, 3])


@numba.njit
def _points_in_convex_polygon_3d_jit(points, polygon_surfaces, normal_vec, direction, num_surfaces):
    """
    Args:
        points (np.ndarray): Input points with shape of (num_points, 3).
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of
            (num_polygon, max_num_surfaces, max_num_points_of_surface, 3).
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.
        normal_vec (np.ndarray): Normal vector of polygon_surfaces.
        d (int): Directions of normal vector.
        num_surfaces (np.ndarray): Number of surfaces a polygon contains
            shape of (num_polygon).
    Returns:
        np.ndarray: Result matrix with the shape of [num_points, num_polygon].
    """
    max_num_surfaces = polygon_surfaces.shape[1]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = (
                    points[i, 0] * normal_vec[j, k, 0]
                    + points[i, 1] * normal_vec[j, k, 1]
                    + points[i, 2] * normal_vec[j, k, 2]
                    + direction[j, k]
                )
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret


def points_in_convex_polygon_3d_jit(points, polygon_surfaces, num_surfaces=None):
    """Check points is in 3d convex polygons.
    Args:
        points (np.ndarray): Input points with shape of (num_points, 3).
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of
            (num_polygon, max_num_surfaces, max_num_points_of_surface, 3).
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.
        num_surfaces (np.ndarray, optional): Number of surfaces a polygon
            contains shape of (num_polygon). Defaults to None.
    Returns:
        np.ndarray: Result matrix with the shape of [num_points, num_polygon].
    """
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, direction = surface_equ_3d(polygon_surfaces[:, :, :3, :])
    return _points_in_convex_polygon_3d_jit(
        points, polygon_surfaces, normal_vec, direction, num_surfaces
    )


def points_in_rbbox(
    points: NDArray[np.float32], rbbox: NDArray[np.float32], z_axis=2, origin=(0.5, 0.5, 0)
):
    """Check points in rotated bbox and return indices.
    Note:
        This function is for counterclockwise boxes.
    Args:
        points (np.ndarray, shape=[N, 3+dim]): Points to query.
        rbbox (np.ndarray, shape=[M, 7]): Boxes3d with rotation.
        z_axis (int, optional): Indicate which axis is height.
            Defaults to 2.
        origin (tuple[int], optional): Indicate the position of
            box center. Defaults to (0.5, 0.5, 0).
    Returns:
        np.ndarray, shape=[N, M]: Indices of points in each box.
    """
    # TODO this function is different from PointCloud3D, be careful
    # when start to use nuscene, check the input
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, 6], origin=origin, axis=z_axis
    )
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return indices


def surface_equ_3d(polygon_surfaces):
    """
    Args:
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of
            [num_polygon, max_num_surfaces, max_num_points_of_surface, 3].
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.
    Returns:
        tuple: normal vector and its direction.
    """
    # return [a, b, c], d in ax+by+cz+d=0
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    surface_vec = polygon_surfaces[:, :, :2, :] - polygon_surfaces[:, :, 1:3, :]
    # normal_vec: [..., 3]
    normal_vec = np.cross(surface_vec[:, :, 0, :], surface_vec[:, :, 1, :])
    # d = -np.inner(normal_vec, points[..., 0, :])
    direction = np.einsum("aij, aij->ai", normal_vec, polygon_surfaces[:, :, 0, :])
    return normal_vec, np.negative(direction)
