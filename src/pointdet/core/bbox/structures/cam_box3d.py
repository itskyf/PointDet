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
