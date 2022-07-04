from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray

from ....typing import FlipDirection
from .interface import IBoxes3D
from .utils import limit_period

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
        cls, src: CameraBoxes3D, rect: NDArray[np.float32], trv2c: NDArray[np.float32]
    ):
        arr = src.tensor
        xyz = arr[..., :3]
        rt_mat = torch.linalg.inv(torch.from_numpy(rect @ trv2c))
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
