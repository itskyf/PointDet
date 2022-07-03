# Copyright (c) OpenMMLab. All rights reserved.
import enum
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm.auto import tqdm

from pointdet.core.bbox import box_np_ops
from pointdet.datasets import KittiDataset
from pointdet.typing import DBInfo, PointCloud


class DatasetEnum(enum.Enum):
    KITTI = enum.auto()


def create_groundtruth_database(
    dataset_enum: DatasetEnum,
    root: Path,
    info_prefix: str,
    info_path: Path,
    database_dir: Optional[Path] = None,
    db_info_save_path: Optional[Path] = None,
    # TODO use mask
    # add_rgb=False,
    # bev_only=False,
    # coors_range=None,
    # lidar_only=False,
    # mask_anno_path=None,
    # with_mask=False,
):
    """Given the raw data, generate the ground truth database.

    Args:
        dataset_class_name (str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str, optional): Path of the info file.
            Default: None.
        mask_anno_path (str, optional): Path of the mask_anno.
            Default: None.
        used_classes (list[str], optional): Classes have been used.
            Default: None.
        database_save_path (str, optional): Path to save database.
            Default: None.
        db_info_save_path (str, optional): Path to save db_info.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
        with_mask (bool, optional): Whether to use mask.
            Default: False.
    """
    if database_dir is None:
        database_dir = root / f"{info_prefix}_gt_database"
    database_dir.mkdir()

    if db_info_save_path is None:
        db_info_save_path = root / f"{info_prefix}_dbinfos_train.pkl"

    rng = np.random.default_rng()
    if dataset_enum is DatasetEnum.KITTI:
        dataset = KittiDataset(root / "training", info_path, "velodyne", rng)
    else:
        raise NotImplementedError

    group_counter = 0
    db_infos: defaultdict[str, list[DBInfo]] = defaultdict(list)
    for pt_cloud in tqdm(dataset, desc="GT database"):
        pt_cloud: PointCloud
        assert pt_cloud.annos is not None

        difficulty = pt_cloud.annos.difficulty
        group_ids = pt_cloud.annos.group_ids
        names = pt_cloud.annos.names

        bboxes_3d = pt_cloud.annos.bboxes_3d.tensor.numpy()
        points = pt_cloud.points.numpy()
        point_indices = box_np_ops.points_in_rbbox(points, bboxes_3d)

        group_dict = {}
        img_idx = pt_cloud.sample_idx
        for gt_idx, (bbox_3d, diff, name, local_gid) in enumerate(
            zip(bboxes_3d, difficulty, names, group_ids)
        ):
            filename = f"{img_idx}_{name}_{gt_idx}.npy"
            rel_path = Path(f"{info_prefix}_gt_database") / filename

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, gt_idx]]
            gt_points[:, :3] -= bboxes_3d[gt_idx, :3]
            np.save(database_dir / filename, gt_points)

            if local_gid not in group_dict:
                group_dict[local_gid] = group_counter
                group_counter += 1
            gid = group_dict[local_gid]
            num_gt_pts = gt_points.shape[0]
            db_infos[name].append(
                DBInfo(name, bbox_3d, rel_path, diff, gid, gt_idx, img_idx, num_gt_pts)
            )

    for name, cls_infos in db_infos.items():
        print(f"{name}: {len(cls_infos)} database infos")

    with db_info_save_path.open("wb") as db_info_file:
        # Convert defaultdict to normal dict
        pickle.dump(dict(db_infos), db_info_file, pickle.HIGHEST_PROTOCOL)
