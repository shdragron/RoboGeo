"""Lightning module: the GaussianLSS ModelModule with an epoch-length OneCycle schedule."""
import torch
from omegaconf import OmegaConf

from GaussianLSS.model.model_module import ModelModule


class UnifiedModelModule(ModelModule):
    def configure_optimizers(self):
        params = [p for p in self.backbone.parameters() if p.requires_grad]
        params += [p for p in self.loss_func.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, **OmegaConf.to_container(self.optimizer_args, resolve=True))

        sch = OmegaConf.to_container(self.scheduler_args, resolve=True)
        interval = sch.pop("interval", "step")
        # Epoch-based recipes leave trainer.max_steps at -1: size the cycle to the run.
        if sch.get("total_steps") in (None, -1):
            sch["total_steps"] = int(self.trainer.estimated_stepping_batches)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **sch)
        return [optimizer], [{"scheduler": scheduler, "interval": interval}]

    def on_save_checkpoint(self, checkpoint):
        # Store the config as plain containers so checkpoints load with torch.load(weights_only=True).
        hparams = checkpoint.get(self.CHECKPOINT_HYPER_PARAMS_KEY)
        if OmegaConf.is_config(hparams):
            checkpoint[self.CHECKPOINT_HYPER_PARAMS_KEY] = OmegaConf.to_container(hparams, resolve=True)
            checkpoint.pop(self.CHECKPOINT_HYPER_PARAMS_TYPE, None)
