"""Lightning data module over the per-scene CARLA datasets."""
import pytorch_lightning as pl
import torch

from .carla_data import build_datasets


class DataModule(pl.LightningDataModule):
    def __init__(self, data_cfg, loader_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.loader_cfg = loader_cfg

    def _loader(self, split, shuffle):
        dataset = torch.utils.data.ConcatDataset(build_datasets(self.data_cfg, split))
        loader_cfg = dict(self.loader_cfg)
        batch_size = loader_cfg[f"{split}_batch_size"]
        del loader_cfg["train_batch_size"], loader_cfg["val_batch_size"]
        if loader_cfg["num_workers"] == 0:
            loader_cfg["prefetch_factor"] = None
        return torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, **loader_cfg)

    def train_dataloader(self):
        return self._loader("train", shuffle=True)

    def val_dataloader(self):
        return self._loader("val", shuffle=False)
