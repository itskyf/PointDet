import numpy as np

from ....typing import FlipDirection
from .interface import IBoxes3D


class CameraBoxes3D(IBoxes3D):
    """3D boxes in CAM coordinates.
    Coordinates in camera:
    .. code-block:: none
                z front (yaw=-0.5*pi)
               /
              /
             0 ------> x right (yaw=0)
             |
             |
             v
        down y
    The relative coordinate of bottom center in a CAM box is (0.5, 1.0, 0.5),
    and the yaw is around the y axis, thus the rotation axis=1.
    The yaw is 0 at the positive direction of x axis, and decreases from
    the positive direction of x to the positive direction of z.
    Attributes:
        tensor (torch.Tensor): Float matrix in shape (N, box_dim).
        box_dim (int): Integer indicating the dimension of a box
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as
            axis-aligned boxes tightly enclosing the original boxes.
    """

    def flip(self, bev_direction: FlipDirection):
        """Flip the boxes in BEV along given BEV direction.

        In CAM coordinates, it flips the x (horizontal) or z (vertical) axis.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
            points (torch.Tensor | np.ndarray | :obj:`BasePoints`, optional):
                Points to flip. Defaults to None.

        Returns:
            torch.Tensor, numpy.ndarray or None: Flipped points.
        """
        if bev_direction is FlipDirection.HORIZONTAL:
            self.tensor[:, 0::7] = -self.tensor[:, 0::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6] + np.pi
        elif bev_direction is FlipDirection.VERTICAL:
            self.tensor[:, 2::7] = -self.tensor[:, 2::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6]
        else:
            raise ValueError
