"""Validation IoU with a fixed threshold, a visibility setting and a distinct label.

The GaussianLSS IoUMetric reports the maximum over thresholds 0.4/0.45/0.5 under the key
'IoU_<key>'. This variant takes the thresholds as an argument and labels each instance, so
IoU@0.5 at vis>=2 and over all cells can be logged side by side.
"""
import numpy as np
import torch

from GaussianLSS.metrics import IoUMetric, BaseIoUMetric


class IoUMetricCfg(IoUMetric):
    def __init__(self, min_visibility=0, key="vehicle", thresholds=(0.5,), label=None):
        BaseIoUMetric.__init__(self, thresholds=np.array(list(thresholds), dtype=np.float32))
        self.min_visibility = min_visibility
        self.key = key
        vis = "all" if not min_visibility else str(min_visibility)
        thr = "_".join(f"{float(t):g}" for t in thresholds)
        self.label = label or f"{key}@{thr}_vis{vis}"

    def compute(self):
        ious = BaseIoUMetric.compute(self)
        return {f"IoU_{self.label}": torch.round(torch.max(ious), decimals=4)}
