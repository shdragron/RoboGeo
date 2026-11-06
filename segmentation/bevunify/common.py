"""Config, model and checkpoint helpers shared by train.py and eval.py."""
import hashlib
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torchmetrics import MetricCollection

from GaussianLSS.losses import MultipleLoss

from .data_module import DataModule
from .model_module import UnifiedModelModule


def setup_config(cfg):
    OmegaConf.set_struct(cfg, False)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, True)
    Path(cfg.experiment.save_dir).mkdir(parents=True, exist_ok=True)


def build_model(cfg):
    return instantiate(cfg.model)


def build_model_module(cfg):
    loss_func = MultipleLoss(instantiate(cfg.loss))
    metrics = MetricCollection(dict(instantiate(cfg.metrics).items()), compute_groups=False)
    return UnifiedModelModule(build_model(cfg), loss_func, metrics, cfg.optimizer, cfg.scheduler,
                              cfg=cfg, val_only=cfg.val_only)


def build_data_module(cfg):
    return DataModule(cfg.data, cfg.loader)


def load_weights(model, path):
    """Load a checkpoint into ``model`` with strict key matching.

    Accepts the released weights-only files and checkpoints written by train.py; both hold
    'state_dict' with the Lightning-module prefix 'backbone.'.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    prefix = "backbone."
    state_dict = {k[len(prefix):]: v for k, v in checkpoint["state_dict"].items() if k.startswith(prefix)}
    model.load_state_dict(state_dict, strict=True)
    return model


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
