"""Training entry point.

    python -m bevunify.train +experiment=gaussianlss data.platform=sedan
    python -m bevunify.train +experiment=cvt data.platform=suv

Defaults reproduce the recipe of the released checkpoints (2 GPUs x 16, SyncBN, seed 2025).
Checkpoints go to <save_dir>/<project>/<uuid>/checkpoints: best.ckpt (best validation
mIoU) and last.ckpt. Relaunching with the same experiment.uuid resumes from last.ckpt, so
set experiment.uuid explicitly for runs that may need a restart.
"""
import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.strategies.ddp import DDPStrategy

from bevunify import ROOT
from bevunify.common import build_data_module, build_model_module, setup_config

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path=str(ROOT / "config"), config_name="config")
def main(cfg):
    torch.set_float32_matmul_precision("high")
    setup_config(cfg)
    pl.seed_everything(cfg.experiment.seed, workers=True)

    model_module = build_model_module(cfg)
    data_module = build_data_module(cfg)

    project_dir = Path(cfg.experiment.save_dir) / cfg.experiment.project
    ckpt_dir = project_dir / str(cfg.experiment.uuid) / "checkpoints"
    resume = ckpt_dir / "last.ckpt"
    resume = resume if resume.exists() else None
    if resume is not None:
        log.info(f"Resuming from {resume}")

    # CSV logs: <save_dir>/<project>/<uuid>/version_<n>/metrics.csv, one version per launch.
    logger = pl.loggers.CSVLogger(save_dir=str(project_dir), name=str(cfg.experiment.uuid))
    callbacks = [
        ModelSummary(max_depth=2),
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(dirpath=str(ckpt_dir), filename="best", monitor="val/metrics/mIoU", mode="max",
                        save_last=True, save_weights_only=False),
    ]

    devices = cfg.trainer.devices
    # static_graph: CVT's EfficientNet uses reentrant gradient checkpointing, which DDP's
    # default reducer rejects ("marked ready twice").
    strategy = "auto" if devices == 1 else DDPStrategy(static_graph=True)
    trainer = pl.Trainer(logger=logger, callbacks=callbacks, strategy=strategy, **cfg.trainer)
    if trainer.global_rank == 0:
        log.info("\n" + OmegaConf.to_yaml(cfg, resolve=True))

    trainer.fit(model_module, datamodule=data_module, ckpt_path=resume)


if __name__ == "__main__":
    main()
