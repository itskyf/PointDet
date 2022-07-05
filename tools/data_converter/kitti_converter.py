# Copyright (c) OpenMMLab. All
import pickle
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from tqdm.contrib.concurrent import thread_map

from pointdet.core.bbox import box_np_ops
from pointdet.datasets.kitti import get_data_path
from pointdet.datasets.kitti.typing import KittiAnnotation, KittiCalib, KittiInfo

from . import kitti_ops


def gen_kitti_infos(root: Path, pkl_prefix: str, out_dir: Path):
    """Create info file of KITTI dataset.
    Given the raw data, generate its related info file in pkl format.
    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str, optional): Prefix of the info file to be generated.
            Default: 'kitti'.
        with_plane (bool, optional): Whether to use plane information.
            Default: False.
        save_path (str, optional): Path to save the info file.
            Default: None.
    """
    imageset_dir = root / "ImageSets"
    train_img_ids = _read_imageset_file(imageset_dir / "train.txt")
    val_img_ids = _read_imageset_file(imageset_dir / "val.txt")
    test_img_ids = _read_imageset_file(imageset_dir / "test.txt")

    print("Generating infos. This may take several minutes.")
    # Train set
    train_v_dir = out_dir / "training" / "velodyne_reduced"
    train_v_dir.mkdir()
    pt_v_path = train_v_dir.with_name("velodyne_pt")
    pt_v_path.mkdir()

    kitti_infos_train = _gen_kitti_infos(root, train_img_ids, out_reduce_path=train_v_dir)
    info_path = out_dir / f"{pkl_prefix}_infos_train.pkl"
    print(f"Kitti info train file is saved to {info_path}")
    with info_path.open("wb") as info_file:
        pickle.dump(kitti_infos_train, info_file, pickle.HIGHEST_PROTOCOL)

    # Val set
    kitti_infos_val = _gen_kitti_infos(root, val_img_ids, out_reduce_path=train_v_dir)
    info_path = out_dir / f"{pkl_prefix}_infos_val.pkl"
    print(f"Kitti info val file is saved to {info_path}")
    with info_path.open("wb") as info_file:
        pickle.dump(kitti_infos_val, info_file, pickle.HIGHEST_PROTOCOL)

    info_path = out_dir / f"{pkl_prefix}_infos_trainval.pkl"
    print(f"Kitti info trainval file is saved to {info_path}")
    with info_path.open("wb") as info_file:
        pickle.dump(kitti_infos_train + kitti_infos_val, info_file, pickle.HIGHEST_PROTOCOL)

    # Test set
    test_v_dir = out_dir / "testing" / "velodyne_reduced"
    test_v_dir.mkdir()
    pt_v_path = test_v_dir.with_name("velodyne_pt")
    pt_v_path.mkdir()

    kitti_infos_test = _gen_kitti_infos(
        root, test_img_ids, training=False, get_label=False, out_reduce_path=test_v_dir
    )
    info_path = out_dir / f"{pkl_prefix}_infos_test.pkl"
    print(f"Kitti info test file is saved to {info_path}")
    with info_path.open("wb") as info_file:
        pickle.dump(kitti_infos_test, info_file, pickle.HIGHEST_PROTOCOL)


def _extend_matrix(mat: np.ndarray):
    return np.concatenate([mat, np.array([[0.0, 0.0, 0.0, 1.0]])], dtype=mat.dtype)


def _gen_kitti_infos(
    root: Path,
    image_ids: list[int],
    out_reduce_path: Path,
    training: bool = True,
    get_label: bool = True,
    extend_matrix: bool = True,
):
    # TODO merge docs with kitti.typing
    """
    KITTI annotation format version 2:
    [optional]points: [N, 3+] point cloud
    [optional, for kitti]image:
        image_idx: ...
        image_path: ...
        image_shape: ...
    point_cloud:
        num_features: 4
        velodyne_path: ...
    [optional, for kitti]calib:
        R0_rect: ...
        Tr_velo_to_cam: ...
        P2: ...
    annos:
        location: [num_gt, 3] array
        dimensions: [num_gt, 3] array
        rotation_y: [num_gt] angle array
        name: [num_gt] ground truth name array
        [optional]difficulty: kitti difficulty
        [optional]group_ids: used for multi-part object
    """
    # TODO num_features variable instead of 4
    pt_v_path = out_reduce_path.with_name("velodyne_pt")  # TODO consistency with _gen_kitti_info

    def map_func(idx: int):
        calib = _get_calib(idx, root, extend_matrix, training)
        rect = calib.r0_rect
        trv2c = calib.trv2c

        img_path = get_data_path(idx, root, "image_2", "png", training)
        with Image.open(root / img_path) as img:
            img_w, img_h = img.size
        img_shape = np.array((img_h, img_w), dtype=np.int32)

        v_path = root / get_data_path(idx, root, "velodyne", "bin", training)
        points = np.fromfile(v_path, dtype=np.float32).reshape(-1, 4)
        pt_fname = f"{v_path.stem}.pt"
        v_path = pt_v_path / pt_fname
        torch.save(torch.from_numpy(points), v_path.with_suffix(".pt"))
        if out_reduce_path is not None:
            points = kitti_ops.remove_outside_points(points, img_shape, rect, trv2c, calib.P2)
            v_path = out_reduce_path / pt_fname
            torch.save(torch.from_numpy(points), v_path)

        annos = _get_label_anno(idx, root, points, rect, trv2c, training) if get_label else None
        # TODO output plane
        return KittiInfo(idx, calib, img_shape, 4, annos)

    return thread_map(map_func, image_ids)


def _get_anno_difficulty(
    bbox: NDArray[np.float32],
    dims: NDArray[np.float32],
    occlusion: NDArray[np.int32],
    truncation: NDArray[np.float32],
):
    min_height = [40, 25, 25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [0, 1, 2]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [0.15, 0.3, 0.5]  # maximum truncation level of the groundtruth used for evaluation
    height = bbox[:, 3] - bbox[:, 1]

    dims_len = len(dims)
    easy_mask = np.ones(dims_len, dtype=np.bool_)
    moderate_mask = np.ones(dims_len, dtype=np.bool_)
    hard_mask = np.ones(dims_len, dtype=np.bool_)

    for i, (height, occ, trunc) in enumerate(zip(height, occlusion, truncation)):
        if occ > max_occlusion[0] or height <= min_height[0] or trunc > max_trunc[0]:
            easy_mask[i] = False
        if occ > max_occlusion[1] or height <= min_height[1] or trunc > max_trunc[1]:
            moderate_mask[i] = False
        if occ > max_occlusion[2] or height <= min_height[2] or trunc > max_trunc[2]:
            hard_mask[i] = False
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    difficuties = []
    for i in range(dims_len):
        if is_easy[i]:
            difficuties.append(0)
        elif is_moderate[i]:
            difficuties.append(1)
        elif is_hard[i]:
            difficuties.append(2)
        else:
            difficuties.append(-1)
    return np.array(difficuties, np.int32)


def _get_calib(idx: int, root: Path, extend_matrix: bool, training: bool):
    calib_path = root / get_data_path(idx, root, "calib", "txt", training)
    with calib_path.open() as calib_file:
        lines = calib_file.readlines()
    P0 = np.array(list(map(float, lines[0].split(" ")[1:13])), dtype=np.float32).reshape(3, 4)
    P1 = np.array(list(map(float, lines[1].split(" ")[1:13])), dtype=np.float32).reshape(3, 4)
    P2 = np.array(list(map(float, lines[2].split(" ")[1:13])), dtype=np.float32).reshape(3, 4)
    P3 = np.array(list(map(float, lines[3].split(" ")[1:13])), dtype=np.float32).reshape(3, 4)
    if extend_matrix:
        P0 = _extend_matrix(P0)
        P1 = _extend_matrix(P1)
        P2 = _extend_matrix(P2)
        P3 = _extend_matrix(P3)

    r0_rect = np.array(list(map(float, lines[4].split(" ")[1:10])), dtype=np.float32).reshape(3, 3)
    if extend_matrix:
        rect_4x4 = np.identity(4, dtype=np.float32)
        rect_4x4[:3, :3] = r0_rect
    else:
        rect_4x4 = r0_rect

    tr_velo_2_cam = np.array(list(map(float, lines[5].split(" ")[1:13])), dtype=np.float32)
    tr_imu_2_velo = np.array(list(map(float, lines[6].split(" ")[1:13])), dtype=np.float32)
    tr_velo_2_cam = tr_velo_2_cam.reshape(3, 4)
    tr_imu_2_velo = tr_imu_2_velo.reshape(3, 4)
    if extend_matrix:
        tr_velo_2_cam = _extend_matrix(tr_velo_2_cam)
        tr_imu_2_velo = _extend_matrix(tr_imu_2_velo)

    return KittiCalib(P0, P1, P2, P3, rect_4x4, tr_velo_2_cam, tr_imu_2_velo)


def _get_label_anno(
    idx: int,
    root: Path,
    points: NDArray[np.float32],
    rect: NDArray[np.float32],
    trv2c: NDArray[np.float32],
    training: bool,
):
    label_path = root / get_data_path(idx, root, "label_2", "txt", training)
    with label_path.open() as label_file:
        content = [line.strip().split(" ") for line in label_file.readlines()]

    names = [x[0] for x in content]
    num_gt = len(names)
    num_obj = num_gt - names.count("DontCare")

    indices = np.concatenate([np.arange(num_obj), np.full(num_gt - num_obj, -1)], dtype=np.int32)
    group_ids = np.arange(num_gt, dtype=np.int32)
    truncated = np.array([float(x[1]) for x in content], dtype=np.float32)
    occluded = np.array([int(x[2]) for x in content], dtype=np.int32)
    alpha = np.array([float(x[3]) for x in content], dtype=np.float32)
    bboxes = np.array([[float(info) for info in x[4:8]] for x in content], dtype=np.float32)
    # dimensions will convert hwl format to standard lhw(camera) format.
    dimensions = np.array([[float(info) for info in x[8:11]] for x in content], dtype=np.float32)
    dimensions = dimensions[:, [2, 0, 1]]
    difficuties = _get_anno_difficulty(bboxes, dimensions, occluded, truncated)
    location = np.array([[float(info) for info in x[11:14]] for x in content], dtype=np.float32)
    # location = location.reshape(-1, 3)
    rotation_y = np.array([float(x[14]) for x in content], dtype=np.float32)
    # rotation_y = rotation_y.reshape(-1)
    # have score
    score = (
        np.array([float(x[15]) for x in content], dtype=np.float32)  # have score
        if len(content) != 0 and len(content[0]) == 16
        else np.zeros(bboxes.shape[0], dtype=np.float32)
    )

    # From MMDetection3D's _calculate_num_points_in_gt
    dims = dimensions[:num_obj]
    loc = location[:num_obj]
    rots = rotation_y[:num_obj][..., np.newaxis]
    gt_boxes_camera = np.concatenate([loc, dims, rots], axis=1)
    gt_boxes_lidar = kitti_ops.box_camera_to_lidar(gt_boxes_camera, rect, trv2c)
    gt_indices = box_np_ops.points_in_rbbox(points[:, :3], gt_boxes_lidar)
    ignored = np.full(len(dimensions) - num_obj, fill_value=-1, dtype=np.int32)
    num_gt_points = np.concatenate([gt_indices.sum(axis=0), ignored], dtype=np.int32)

    return KittiAnnotation(
        indices,
        np.array(names),
        difficuties,
        group_ids,
        truncated,
        occluded,
        alpha,
        bboxes,
        dimensions,
        location,
        rotation_y,
        score,
        num_gt_points,
    )


def _read_imageset_file(path: Path):
    with path.open() as imageset_file:
        lines = imageset_file.readlines()
    return list(map(int, lines))
