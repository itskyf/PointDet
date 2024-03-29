from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch

from ....typing import FlipDirection
from .utils import limit_period

_DEFAULT_ORIGIN = (0.5, 0.5, 0)


class IBoxes3D(ABC):
    """Base class for 3D Boxes.

    Note:
        The box is bottom centered, i.e. the relative position of origin in
        the box is (0.5, 0.5, 0).

    Args:
        tensor (torch.Tensor | np.ndarray): a N x box_dim matrix.
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
        tensor: Union[np.ndarray, torch.Tensor],
        box_dim: int = 7,
        with_yaw: bool = True,
        origin: tuple[float, float, float] = _DEFAULT_ORIGIN,
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
            fake_rot = tensor.new_zeros(tensor.size(0), 1)
            tensor = torch.cat((tensor, fake_rot), dim=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        self.tensor = tensor.clone()

        if origin != _DEFAULT_ORIGIN:
            dst = self.tensor.new_tensor(_DEFAULT_ORIGIN)
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    @property
    def bev(self):
        """torch.Tensor: 2D BEV box of each box with rotation in XYWHR format, in shape (N, 5)."""
        return self.tensor[:, [0, 1, 3, 4, 6]]

    @abstractmethod
    def flip(self, bev_direction: FlipDirection):
        """Flip the boxes in BEV along given BEV direction.

        Args:
            bev_direction (str, optional): Direction by which to flip.
                Can be chosen from 'horizontal' and 'vertical'.
                Defaults to 'horizontal'.
        """

    @abstractmethod
    def rotate(self, angle: Union[float, np.ndarray, torch.Tensor]):
        """Rotate boxes with points (optional) with the given angle or rotation
        matrix.

        Args:
            angle (float | torch.Tensor | np.ndarray):
                Rotation angle or rotation matrix.
            points (torch.Tensor | numpy.ndarray |
                :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.
        """

    def limit_yaw(self, offset: float = 0.5, period: float = np.pi):
        """Limit the yaw to a given period and offset.

        Args:
            offset (float, optional): The offset of the yaw. Defaults to 0.5.
            period (float, optional): The expected period. Defaults to np.pi.
        """
        self.tensor[:, 6] = limit_period(self.tensor[:, 6], offset, period)

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

    def new_boxes(self, data: Union[np.ndarray, torch.Tensor]):
        """Create a new box object with data.

        The new box and its tensor has the similar properties
            as self and self.tensor, respectively.

        Args:
            data (torch.Tensor | numpy.array): Data to be copied.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new bbox object with ``data``,
                the object's other properties are similar to ``self``.
        """
        new_tensor = (
            self.tensor.new_tensor(data)
            if not isinstance(data, torch.Tensor)
            else data.to(self.tensor.device)
        )
        return type(self)(new_tensor, box_dim=self.box_dim, with_yaw=self.with_yaw)
