from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import torch

from ...typing import FlipDirection
from ..bbox.structures.utils import rotation_3d_in_axis


class IPoints(ABC):
    """Base class for Points.

    Args:
        tensor (torch.Tensor | np.ndarray): a N x points_dim matrix.
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
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that
            # does not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, points_dim)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == points_dim, tensor.size()
        self.tensor = tensor
        self.points_dim = points_dim
        self.attr_dims = attr_dims
        self.rotation_axis = 0

    def __getitem__(self, key: Union[int, slice, np.ndarray, torch.Tensor]):
        """
        Note:
            The following usage are allowed:
            1. `new_points = points[3]`:
                return a `Points` that contains only one point.
            2. `new_points = points[2:10]`:
                return a slice of points.
            3. `new_points = points[vector]`:
                where vector is a torch.BoolTensor with `length = len(points)`.
                Nonzero elements in the vector will be selected.
            4. `new_points = points[3:11, vector]`:
                return a slice of points and attribute dims.
            5. `new_points = points[4:12, 2]`:
                return a slice of points with single attribute.
            Note that the returned Points might share storage with this Points,
            subject to Pytorch's indexing semantics.

        Returns:
            :obj:`BasePoints`: A new object of
                :class:`BasePoints` after indexing.
        """
        original_type = type(self)
        if isinstance(key, int):
            return original_type(self.tensor[key].view(1, -1), self.points_dim, self.attr_dims)
        if isinstance(key, np.ndarray):
            new_pts = self.tensor[torch.from_numpy(key)]
        elif isinstance(key, (slice, torch.Tensor)):
            new_pts = self.tensor[key]
        else:
            raise NotImplementedError(f"Invalid slice {key}!")
        assert new_pts.dim() == 2, f"Indexing on points with {key} failed to return a matrix!"
        points_dims = new_pts.size(1)
        return original_type(new_pts, points_dims, self.attr_dims)

    @property
    def bev(self):
        """torch.Tensor: BEV of the points in shape (N, 2)."""
        return self.tensor[:, [0, 1]]

    @property
    def coord(self):
        """torch.Tensor: Coordinates of each point in shape (N, 3)."""
        return self.tensor[:, :3]

    @abstractmethod
    def flip(self, bev_direction: FlipDirection):
        """Flip the points along given BEV direction.
            Args:
        bev_direction (str): Flip direction (horizontal or vertical).
        """

    def rotate(self, rotation: Union[float, np.ndarray, torch.Tensor], axis: Optional[int] = None):
        """Rotate points with the given rotation matrix or angle.

        Args:
            rotation (float | np.ndarray | torch.Tensor): Rotation matrix
                or angle.
            axis (int, optional): Axis to rotate at. Defaults to None.
        """
        if not isinstance(rotation, torch.Tensor):
            rotation = self.tensor.new_tensor(rotation)
        assert (
            rotation.shape == torch.Size([3, 3]) or rotation.numel() == 1
        ), f"invalid rotation shape {rotation.shape}"

        if axis is None:
            axis = self.rotation_axis

        if rotation.numel() == 1:
            rotated_points = rotation_3d_in_axis(self.tensor[:, :3][None], rotation, axis)
            self.tensor[:, :3] = rotated_points.squeeze(0)
        else:
            # rotation.numel() == 9
            self.tensor[:, :3] = self.tensor[:, :3] @ rotation

    def scale(self, scale_factor: float):
        """Scale the points with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the points.
        """
        self.tensor[:, :3] *= scale_factor

    def translate(self, trans_vector):
        """Translate points with the given translation vector.

        Args:
            trans_vector (np.ndarray, torch.Tensor): Translation
                vector of size 3 or nx3.
        """
        if not isinstance(trans_vector, torch.Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        trans_vector = trans_vector.squeeze(0)
        if trans_vector.ndim == 1:
            assert trans_vector.size(0) == 3
        elif trans_vector.ndim == 2:
            assert trans_vector.size(0) == self.tensor.size(0) and trans_vector.size(1) == 3
        else:
            raise NotImplementedError(
                f"Unsupported translation vector of shape {trans_vector.shape}"
            )
        self.tensor[:, :3] += trans_vector

    def new_points(self, data: Union[np.ndarray, torch.Tensor]):
        """Create a new point object with data.

        The new point and its tensor has the similar properties
            as self and self.tensor, respectively.

        Args:
            data (torch.Tensor | numpy.array): Data to be copied.

        Returns:
            :obj:`BasePoints`: A new point object with ``data``,
                the object's other properties are similar to ``self``.
        """
        new_tensor = (
            self.tensor.new_tensor(data)
            if not isinstance(data, torch.Tensor)
            else data.to(self.tensor.device)
        )
        return type(self)(new_tensor, self.points_dim, self.attr_dims)
