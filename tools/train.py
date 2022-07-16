import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

import hydra
import omegaconf
import torch
import torch.backends.cudnn
from hydra.core.hydra_config import HydraConfig
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange

from pointdet.datasets import IDataset, PCDBatch, form_pcd_batch

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_name="main", config_path=None)
def main(cfg: omegaconf.DictConfig):
    hydra_cfg = HydraConfig.get()
    out_dir = Path(hydra_cfg.run.dir)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if not cfg.disable_cuda and torch.cuda.is_available() else "cpu")

    transforms = hydra.utils.instantiate(cfg.model.augmentation)
    train_set: IDataset = hydra.utils.instantiate(cfg.dataset, split="train", transforms=transforms)
    train_loader = DataLoader(
        train_set,
        cfg.model.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=form_pcd_batch,
        pin_memory=True,
    )

    model: nn.Module = hydra.utils.instantiate(cfg.model.module)
    model.to(device)
    optimizer: optim.Optimizer = hydra.utils.instantiate(
        cfg.model.optimizer, params=model.parameters()
    )
    # TODO initialize lr_scheduler from config
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, total_steps=cfg.model.epochs * len(train_loader), **cfg.model.lr_scheduler
    )
    state = CheckpointState(model, optimizer, scheduler, out_dir)

    tb_writer = SummaryWriter(out_dir)
    if cfg.ckpt is not None:
        cur_epoch = state.load(cfg.ckpt) + 1
        logger.info("Loaded checkpoint %s", cfg.ckpt)
    else:
        cur_epoch = 0
        logger.info("Training from scratch")
    logger.info("Batch size: %d", cfg.model.batch_size)

    best_loss = float("inf")
    cur_step = cur_epoch * len(train_loader)
    model.train()
    for epoch in trange(cur_epoch, cfg.model.epochs, desc="Epoch", dynamic_ncols=True):
        loss = None
        losses_dict = {}

        for pcd_batch in tqdm(train_loader, desc="Step", dynamic_ncols=True, leave=False):
            cur_step += 1
            optimizer.zero_grad(set_to_none=True)

            pcd_batch: PCDBatch = pcd_batch.to(device)
            losses_dict: dict[str, torch.Tensor] = model(pcd_batch)
            loss = sum(losses_dict.values())

            if TYPE_CHECKING:
                assert isinstance(loss, torch.Tensor)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.model.clip_max_norm)
            optimizer.step()
            scheduler.step()

        assert loss is not None and losses_dict
        loss = loss.item()
        losses_dict_float = {key: val.item() for key, val in losses_dict.items()}

        tb_writer.add_scalar("train/loss", loss, cur_step)
        for key, val in losses_dict_float.items():
            tb_writer.add_scalar(f"train/{key}_loss", val, cur_step)
        if loss < best_loss:
            best_loss = loss
            state.save(epoch, losses_dict_float, "ckpt_best_loss.pt")
        if epoch % cfg.save_interval == cfg.save_interval - 1:
            state.save(epoch, losses_dict_float)

    tb_writer.close()


class Checkpoint(NamedTuple):
    epoch: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any]
    losses: dict[str, float]


class CheckpointState:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: lr_scheduler._LRScheduler,
        out_dir: Path,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.out_dir = out_dir

    def save(self, epoch: int, losses_dict: dict[str, float], ckpt_name: Optional[str] = None):
        if ckpt_name is None:
            ckpt_name = f"ckpt_{epoch}.pt"
        ckpt = Checkpoint(
            epoch,
            self.model.state_dict(),
            self.optimizer.state_dict(),
            self.scheduler.state_dict(),
            losses_dict,
        )
        torch.save(ckpt, self.out_dir / ckpt_name)

    def load(self, ckpt_name: str):
        ckpt: Checkpoint = torch.load(self.out_dir / ckpt_name)
        self.model.load_state_dict(ckpt.model_state)
        self.optimizer.load_state_dict(ckpt.optimizer_state)
        self.scheduler.load_state_dict(ckpt.scheduler_state)
        return ckpt.epoch


if __name__ == "__main__":
    main()
