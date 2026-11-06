"""Cross View Transformer wrapper.

CVT reads image, intrinsics and extrinsics (ego to camera) from the batch and returns one
tensor per ``outputs`` key; the config names them vehicle and vehicle_center, so the
output feeds the shared losses and metrics directly. The upstream model sits in ``net``,
the key prefix of the released checkpoints.
"""
import torch.nn as nn


class CVTWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, batch):
        return self.net(batch)
