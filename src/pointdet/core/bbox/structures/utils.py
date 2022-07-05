import numpy as np
import torch

from ...utils import array_converter


@array_converter(("val",))
def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.
    Args:
        val (torch.Tensor | np.ndarray): The value to be converted.
        offset (float, optional): Offset to set the value range.
            Defaults to 0.5.
        period ([type], optional): Period of the value. Defaults to np.pi.
    Returns:
        (torch.Tensor | np.ndarray): Value in the range of
            [-offset * period, (1-offset) * period]
    """
    return val - torch.floor(val / period + offset) * period


@array_converter(("points", "angles"))
def rotation_3d_in_axis(points, angles, axis: int = 0, clockwise: bool = False):
    """Rotate points by angles according to axis.
    Args:
        points (np.ndarray | torch.Tensor | list | tuple ):
            Points of shape (N, M, 3).
        angles (np.ndarray | torch.Tensor | list | tuple | float):
            Vector of angles in shape (N,)
        axis (int, optional): The axis to be rotated. Defaults to 0.
        return_mat: Whether or not return the rotation matrix (transposed).
            Defaults to False.
        clockwise: Whether the rotation is clockwise. Defaults to False.
    Raises:
        ValueError: when the axis is not in range [0, 1, 2], it will
            raise value error.
    Returns:
        (torch.Tensor | np.ndarray): Rotated points in shape (N, M, 3).
    """
    batch_free = points.ndim == 2
    if batch_free:
        points = points[None]
    if isinstance(angles, float) or angles.ndim == 0:
        angles = torch.full(points.shape[:1], angles)

    p_a_shape = f"{points.shape}, {angles.shape}"
    assert points.ndim == 3 and angles.ndim == 1, f"Incorrect dims: {p_a_shape}"
    assert points.size(0) == angles.size(0), f"Incorrect shape: {p_a_shape}"
    assert points.size(-1) in [2, 3], f"Points size should be 2 or 3 instead of {points.size(-1)}"
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)

    if points.size(-1) == 3:
        if axis in (1, -2):
            rot_mat_t = torch.stack(
                [
                    torch.stack([rot_cos, zeros, -rot_sin]),
                    torch.stack([zeros, ones, zeros]),
                    torch.stack([rot_sin, zeros, rot_cos]),
                ]
            )
        elif axis in (2, -1):
            rot_mat_t = torch.stack(
                [
                    torch.stack([rot_cos, rot_sin, zeros]),
                    torch.stack([-rot_sin, rot_cos, zeros]),
                    torch.stack([zeros, zeros, ones]),
                ]
            )
        elif axis in (0, -3):
            rot_mat_t = torch.stack(
                [
                    torch.stack([ones, zeros, zeros]),
                    torch.stack([zeros, rot_cos, rot_sin]),
                    torch.stack([zeros, -rot_sin, rot_cos]),
                ]
            )
        else:
            raise ValueError(f"axis should in range " f"[-3, -2, -1, 0, 1, 2], got {axis}")
    else:
        rot_mat_t = torch.stack([torch.stack([rot_cos, rot_sin]), torch.stack([-rot_sin, rot_cos])])

    if clockwise:
        rot_mat_t = rot_mat_t.transpose(0, 1)
    points_new = points if points.size(0) == 0 else torch.einsum("aij,jka->aik", points, rot_mat_t)
    if batch_free:
        points_new = points_new.squeeze(0)
    return points_new
