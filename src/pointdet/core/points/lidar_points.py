from typing import Optional, Union

import numpy as np
import torch

from ...typing import FlipDirection
from .interface import IPoints


class LiDARPoints(IPoints):
    """Points of instances in LIDAR coordinates.

    Args:
        tensor (torch.Tensor | np.ndarray | list): a N x points_dim matrix.
        points_dim (int, optional): Number of the dimension of a point.
            Each row is (x, y, z). Defaults to 3.
        attribute_dims (dict, optional): Dictionary to indicate the
            meaning of extra dimension. Defaults to None.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x points_dim.
        points_dim (int): Integer indicating the dimension of a point.
            Each row is (x, y, z, ...).
        attribute_dims (bool): Dictionary to indicate the meaning of extra
            dimension. Defaults to None.
        rotation_axis (int): Default rotation axis for points rotation.
    """

    def __init__(
        self,
        tensor: Union[np.ndarray, torch.Tensor],
        points_dim: int = 3,
        attr_dims: Optional[dict[str, int]] = None,
    ):
        super().__init__(tensor, points_dim, attr_dims)
        self.rotation_axis = 2

    def flip(self, bev_direction: FlipDirection):
        """Flip the points along given BEV direction.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
        """
        if bev_direction is FlipDirection.HORIZONTAL:
            self.tensor[:, 1] = -self.tensor[:, 1]
        elif bev_direction is FlipDirection.VERTICAL:
            self.tensor[:, 0] = -self.tensor[:, 0]
