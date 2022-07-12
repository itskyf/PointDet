/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <torch/extension.h>

#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor.")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous.")
#define CHECK_CONTIGUOUS_CUDA(x) \
  CHECK_CUDA(x);                 \
  CHECK_CONTIGUOUS(x)

constexpr int kThreadsPerBlock = 512;
inline int GetBlockSize(const int n, const int num_threads = kThreadsPerBlock) {
  const int optimal_block_num = (n + num_threads - 1) / num_threads;
  constexpr int max_block_num = 4096;
  return std::min(optimal_block_num, max_block_num);
}
