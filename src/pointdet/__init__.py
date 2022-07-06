import importlib.metadata
from pathlib import Path

from omegaconf import OmegaConf

OmegaConf.register_new_resolver("to_path", Path)  # TODO using structured config instead of resolved
__version__ = importlib.metadata.version(__package__)
