# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from pathlib import Path
from typing import Optional

from data_converter import kitti_converter
from data_converter.create_gt_database import DatasetEnum, create_groundtruth_database


def kitti_data_prep(root: Path, info_prefix: str, out_dir: Optional[Path] = None):
    """Prepare data related to Kitti dataset.
    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.
    Args:
        root_path (Path): path of dataset root.
        info_prefix (str): the prefix of info filenames.
        version (str): dataset version.
        out_dir (str): Output directory of the groundtruth database info.
        with_plane (bool, optional): Whether to use plane information.
            Default: False.
    """
    if out_dir is None:  # TODO maybe move to parent scope
        out_dir = root
    kitti_converter.gen_kitti_infos(root, info_prefix, out_dir)

    # TODO export 2D annotations
    print("Create GT Database of KITTI dataset")
    create_groundtruth_database(DatasetEnum.KITTI, root, info_prefix)


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", choices=["kitti"])
    parser.add_argument("path", type=Path)
    parser.add_argument("--out_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.dataset == "kitti":
        kitti_data_prep(args.path, args.dataset)
    else:
        raise NotImplementedError
