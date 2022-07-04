from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import torch
from numpy.typing import NDArray

from ....typing import FlipDirection


class IBoxes3D(ABC):
    """Base class for 3D Boxes.

    Note:
        The box is bottom centered, i.e. the relative position of origin in
        the box is (0.5, 0.5, 0).

    Args:
        tensor (torch.Tensor | np.ndarray | list): a N x box_dim matrix.
        box_dim (int): Number of the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw).
            Defaults to 7.
        with_yaw (bool): Whether the box is with yaw rotation.
            If False, the value of yaw will be set to 0 as minmax boxes.
            Defaults to True.
        origin (tuple[float], optional): Relative position of the box origin.
            Defaults to (0.5, 0.5, 0). This will guide the box be converted to
            (0.5, 0.5, 0) mode.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicating the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    def __init__(
        self,
        tensor: Union[torch.Tensor, NDArray[np.float32]],
        box_dim: int = 7,
        with_yaw: bool = True,
        origin: Optional[tuple[float, float, float]] = None,
    ):
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, tensor.shape

        self.box_dim = box_dim
        self.with_yaw = with_yaw
        if tensor.size(-1) == 6:
            # If the dimension of boxes is 6, we expand box_dim by padding
            # 0 as a fake yaw and set with_yaw to False.
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 1)
            tensor = torch.cat((tensor, fake_rot), dim=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        self.tensor = tensor

        if origin is not None and origin != (0.5, 0.5, 0):
            dst = self.tensor.new_tensor((0.5, 0.5, 0))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    @property
    def volume(self):
        """torch.Tensor: A vector with volume of each box."""
        return self.tensor[:, 3] * self.tensor[:, 4] * self.tensor[:, 5]

    @property
    def dims(self):
        """torch.Tensor: Size dimensions of each box in shape (N, 3)."""
        return self.tensor[:, 3:6]

    @property
    def yaw(self):
        """torch.Tensor: A vector with yaw of each box in shape (N, )."""
        return self.tensor[:, 6]

    @property
    def height(self):
        """torch.Tensor: A vector with height of each box in shape (N, )."""
        return self.tensor[:, 5]

    @property
    def top_height(self):
        """torch.Tensor:
        A vector with the top height of each box in shape (N, )."""
        return self.bottom_height + self.height

    @property
    def bottom_height(self):
        """torch.Tensor:
        A vector with bottom's height of each box in shape (N, )."""
        return self.tensor[:, 2]

    @property
    def center(self):
        """torch.Tensor: A tensor with center of each box in shape (N, 3)."""
        return self.tensor[:, :3]

    @abstractmethod
    def flip(self, bev_direction: FlipDirection):
        """Flip the boxes in BEV along given BEV direction.

        Args:
            bev_direction (str, optional): Direction by which to flip.
                Can be chosen from 'horizontal' and 'vertical'.
                Defaults to 'horizontal'.
        """

    def scale(self, scale_factor: float):
        """Scale the box with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the boxes.
        """
        self.tensor[:, :6] *= scale_factor
        self.tensor[:, 7:] *= scale_factor  # velocity

    def translate(self, trans_vector):
        """Translate boxes with the given translation vector.

        Args:
            trans_vector (torch.Tensor): Translation vector of size (1, 3).
        """
        if not isinstance(trans_vector, torch.Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        self.tensor[:, :3] += trans_vector
