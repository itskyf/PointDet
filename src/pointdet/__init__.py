import importlib.metadata
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf


def run_dir(default_dir: str, ckpt: Optional[str], experiment: Optional[str]):
    if ckpt is None:
        if experiment is None:
            experiment = "default"
        return Path(default_dir) / experiment

    ckpt_dir = Path(ckpt).parent
    assert ckpt_dir.is_dir()
    return ckpt_dir


__version__ = importlib.metadata.version(__package__)
OmegaConf.register_new_resolver("len", len)
OmegaConf.register_new_resolver("run_dir", run_dir)
OmegaConf.register_new_resolver("to_path", Path)  # TODO using structured config instead of resolved
