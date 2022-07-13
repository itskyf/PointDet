from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx, once_differentiable

from pointdet import _C


class BallQuery(Function):
    """
    Torch autograd Function wrapper for Ball Query C++/CUDA implementations.
    """

    @staticmethod
    def forward(
        ctx: FunctionCtx, centroids: Tensor, points: Tensor, num_neighbors: int, radius: float
    ):
        """
        Arguments defintions the same as in the ball_query function
        """
        indices = _C.ball_query(centroids, points, num_neighbors, radius)
        ctx.mark_non_differentiable(indices)
        return indices

    @staticmethod
    @once_differentiable
    def backward(ctx: FunctionCtx, grad_indices: Tensor):
        return None, None, None, None


ball_query = BallQuery.apply
