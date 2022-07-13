from itertools import chain, tee
from typing import Union

from torch import nn


def build_normal_mlps(
    in_channels: int,
    channels: Union[int, tuple[int]],
    dims: int,
):
    conv, bn = _get_conv_module(dims)
    if isinstance(channels, int):
        channels = (channels,)
    c_ins, c_outs = tee((in_channels, *channels))
    next(c_outs, None)
    return list(
        chain.from_iterable(
            [conv(c_in, c_out, kernel_size=1, bias=False), bn(c_out), nn.ReLU(inplace=True)]
            for c_in, c_out in zip(c_ins, c_outs)
        )
    )


def build_fewer_act_mlps(
    in_channels: int,
    channels: Union[int, tuple[int, ...]],
    dims: int,
):
    """Only activation before last Conv"""
    conv, bn = _get_conv_module(dims)
    if isinstance(channels, int):
        return nn.Sequential(
            conv(in_channels, channels, kernel_size=1, bias=False), bn(channels), nn.GELU()
        )

    assert len(channels) >= 2
    c_ins, c_outs = tee((in_channels, *channels))
    next(c_outs, None)

    modules = list(
        chain.from_iterable(
            [conv(c_in, c_out, kernel_size=1, bias=False), bn(c_out)]
            for c_in, c_out in zip(c_ins, c_outs)
        )
    )
    return [*modules[:-2], activation(), *modules[-2:]]


def _get_conv_module(dims: int):
    if dims == 1:
        return nn.Conv1d, nn.BatchNorm1d
    if dims == 2:
        return nn.Conv2d, nn.BatchNorm2d
    raise NotImplementedError
