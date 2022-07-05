from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
import torch

from ....typing import FlipDirection
from .interface import IBoxes3D
from .utils import limit_period, rotation_3d_in_axis

if TYPE_CHECKING:
    from .cam_box3d import CameraBoxes3D


class LiDARBoxes3D(IBoxes3D):
    """3D boxes in LIDAR coordinates.

    Coordinates in LiDAR:

    .. code-block:: none

                                up z    x front (yaw=0)
                                   ^   ^
                                   |  /
                                   | /
       (yaw=0.5*pi) left y <------ 0

    The relative coordinate of bottom center in a LiDAR box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    The yaw is 0 at the positive direction of x axis, and increases from
    the positive direction of x to the positive direction of y.

    A refactor is ongoing to make the three coordinate systems
    easier to understand and convert between each other.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicating the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    @classmethod
    def from_camera_box3d(
        cls,
        src: CameraBoxes3D,
        rect: Union[np.ndarray, torch.Tensor],
        trv2c: Union[np.ndarray, torch.Tensor],
    ):
        arr = src.tensor
        if not isinstance(rect, torch.Tensor):
            rect = arr.new_tensor(rect)
        if not isinstance(trv2c, torch.Tensor):
            trv2c = arr.new_tensor(trv2c)
        xyz = arr[..., :3]
        rt_mat = torch.linalg.inv(rect @ trv2c)
        if rt_mat.size(1) == 4:  # extend xyz
            xyz = torch.cat([xyz, arr.new_ones(arr.size(0), 1)], dim=-1)
        xyz = xyz @ rt_mat.t()

        x_size, y_size, z_size = arr[..., 3:4], arr[..., 4:5], arr[..., 5:6]
        xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
        yaw = limit_period(-arr[..., 6:7] - np.pi / 2, period=np.pi * 2) if src.with_yaw else None
        arr = (
            torch.cat([xyz[..., :3], xyz_size, yaw, arr[..., 7:]], dim=-1)
            if yaw is not None
            else torch.cat([xyz[..., :3], xyz_size, arr[..., 6:]], dim=-1)
        )
        return cls(arr, box_dim=arr.size(-1), with_yaw=src.with_yaw)

    def flip(self, bev_direction: FlipDirection):
        """Flip the boxes in BEV along given BEV direction.

        In LIDAR coordinates, it flips the y (horizontal) or x (vertical) axis.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
            points (torch.Tensor | np.ndarray | :obj:`BasePoints`, optional):
                Points to flip. Defaults to None.

        Returns:
            torch.Tensor, numpy.ndarray or None: Flipped points.
        """
        if bev_direction is FlipDirection.HORIZONTAL:
            self.tensor[:, 1::7] = -self.tensor[:, 1::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6]
        elif bev_direction is FlipDirection.VERTICAL:
            self.tensor[:, 0::7] = -self.tensor[:, 0::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6] + np.pi
        else:
            raise ValueError

    def rotate(self, angle: Union[float, np.ndarray, torch.Tensor]):
        """Rotate boxes with points (optional) with the given angle or rotation
        matrix.

        Args:
            angles (float | torch.Tensor | np.ndarray):
                Rotation angle or rotation matrix.
            points (torch.Tensor | np.ndarray | :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.

        Returns:
            tuple or None: When ``points`` is None, the function returns
                None, otherwise it returns the rotated points and the
                rotation matrix ``rot_mat_T``.
        """
        if not isinstance(angle, torch.Tensor):
            angle = self.tensor.new_tensor(angle)

        assert (
            angle.shape == (3, 3) or angle.numel() == 1
        ), f"invalid rotation angle shape {angle.shape}"

        if angle.numel() == 1:
            self.tensor[:, 0:3] = rotation_3d_in_axis(
                self.tensor[:, 0:3], angle, axis=2
            )  # yaw axis
        else:
            rot_mat_t = angle
            rot_sin = rot_mat_t[0, 1]
            rot_cos = rot_mat_t[0, 0]
            angle = np.arctan2(rot_sin, rot_cos)
            self.tensor[:, 0:3] = self.tensor[:, 0:3] @ rot_mat_t
        self.tensor[:, 6] += angle

        if self.tensor.size(1) == 9:
            # TODO support other dataset (tensor.shape[1] == 9)
            raise NotImplementedError
