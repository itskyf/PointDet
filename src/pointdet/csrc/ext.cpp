/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/extension.h>

#include "ball_query/ball_query.hpp"
#include "group_points/group_points.hpp"
#include "points_in_boxes/points_in_boxes.hpp"
#include "sample_farthest_points/sample_farthest_points.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ball_query", &BallQuery);
  m.def("group_points", &GroupPoints);
  m.def("group_points_backward", &GroupPointsBackward);
  m.def("points_in_boxes_part", &PointsInBoxesPart);
  m.def("sample_farthest_points", &FarthestPointSampling);
}
