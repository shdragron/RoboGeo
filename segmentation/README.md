# GaussianLSS and CVT

[GaussianLSS](https://github.com/HCIS-Lab/GaussianLSS) (commit cd6d305) and
[CVT](https://github.com/bradyz/cross_view_transformers) (commit 4de6e641) for RoboGeo BEV vehicle
segmentation. The upstream model code is vendored in [`third_party/`](third_party).

> [!IMPORTANT]
> GaussianLSS needs `diff-gaussian-rasterization`, which is licensed for non-commercial research
> and evaluation only ([license](third_party/GaussianLSS/GaussianLSS/model/diff-gaussian-rasterization/LICENSE.md)).
> CVT does not use it.

## Setup

Use the [`env/segmentation`](../env/segmentation) image and run all commands from this folder.
Without Docker, run `bash env/segmentation/install.sh` from the repository root; `--cvt-only` skips
the rasterizer. The EfficientNet-B4 ImageNet weights are downloaded on first use.

| model | experiment | checkpoint (`$ROBOGEO_CKPT/segmentation/`) |
|---|---|---|
| GaussianLSS | `gaussianlss` | `gaussianlss/gaussianlss_{sedan,suv,bus}.pth` |
| CVT | `cvt` | `cvt/cvt_{sedan,suv,bus}.pth` |

## Evaluate

```bash
CK=$ROBOGEO_CKPT/segmentation/gaussianlss      # CVT: replace gaussianlss by cvt

# IoU on the evaluation split of one platform
python -m bevunify.eval +experiment=gaussianlss data.platform=sedan ckpt=$CK/gaussianlss_sedan.pth

# viewpoint perturbation and cross-platform transfer (same command again with bus)
python -m bevunify.eval +experiment=gaussianlss ckpt=$CK/gaussianlss_sedan.pth eval.protocol=vp
python -m bevunify.eval +experiment=gaussianlss ckpt=$CK/gaussianlss_sedan.pth eval.protocol=cts \
    eval.target=suv eval.oracle_ckpt=$CK/gaussianlss_suv.pth
python -m bevunify.aggregate --results eval_results/gaussianlss_full/results
```

IoU covers the vehicle classes 4, 5, 6, 7, 8, 10 and 11, with the prediction thresholded at 0.5
and cells of visibility < 2 ignored. Each cell is written to `eval_results/` in the format of the
detection results, and a restarted run skips the cells that are already finished. One pass over
the 3,792 evaluation frames takes about 2 minutes on an RTX 4090.

## Train

```bash
python -m bevunify.train +experiment=gaussianlss data.platform=sedan experiment.uuid=gaussianlss_sedan
```

The defaults follow the recipe of the released checkpoints:
- batch size 16 per GPU on 2 GPUs (total 32) with SyncBN, fp32 with TF32 matmul, seed 2025, no augmentation;
- the upstream optimizer, schedule and losses of each model, with the learning rates unchanged;
- GaussianLSS 50 epochs, CVT 30 epochs.

Checkpoints are saved to `logs/<project>/<uuid>/checkpoints/`: the best validation
`val/metrics/mIoU` and the last epoch. A relaunch with the same `experiment.uuid` resumes. The
released checkpoints were trained with an extra train-mode forward pass every 150 steps for
visualisation, which also updated the BatchNorm statistics. This code omits that pass.

## Citation

```bibtex
@inproceedings{Lu_2025_CVPR,
  author    = {Lu, Shu-Wei and Tsai, Yi-Hsuan and Chen, Yi-Ting},
  title     = {Toward Real-world BEV Perception: Depth Uncertainty Estimation via Gaussian Splatting},
  booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
  year      = {2025}
}

@inproceedings{zhou2022cross,
  title     = {Cross-view Transformers for real-time Map-view Semantic Segmentation},
  author    = {Zhou, Brady and Kr{\"a}henb{\"u}hl, Philipp},
  booktitle = {CVPR},
  year      = {2022}
}

@article{kerbl3Dgaussians,
  author    = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title     = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal   = {ACM Transactions on Graphics},
  number    = {4},
  volume    = {42},
  year      = {2023}
}
```
