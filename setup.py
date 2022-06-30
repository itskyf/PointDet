# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import torch.cuda
from setuptools import setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

MODULE_NAME = "pointdet"


def get_extensions():
    ext_dir = Path(__file__).parent / "src" / MODULE_NAME / "csrc"

    define_macros = None
    extension = CppExtension
    force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
    sources = list(ext_dir.rglob("*.cpp"))

    if (torch.cuda.is_available() and CUDA_HOME is not None) or force_cuda:
        extension = CUDAExtension
        sources += ext_dir.rglob("*.cu")
        define_macros = [("WITH_CUDA", None)]

    sources = list(map(str, sources))
    return extension(
        name=f"{MODULE_NAME}._C",
        sources=sources,
        include_dirs=[ext_dir],
        define_macros=define_macros,
        extra_compile_args={"cxx": ["-std=c++17"], "nvcc": ["-std=c++17"]},
    )


if __name__ == "__main__":
    setup(
        cmdclass={
            "build_ext": BuildExtension,
        },
        ext_modules=[get_extensions()],
    )
