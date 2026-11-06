"""Evaluation entry point: normal, viewpoint perturbation (vp) and cross-platform transfer (cts).

    # vehicle IoU of a model on its own platform
    python -m bevunify.eval +experiment=gaussianlss data.platform=suv ckpt=<gaussianlss_suv.pth>

    # viewpoint perturbation of the Sedan model (631 cells)
    python -m bevunify.eval +experiment=gaussianlss ckpt=<gaussianlss_sedan.pth> eval.protocol=vp

    # cross-platform transfer to the SUV: IMG and CAL with the Sedan model, ORACLE with the SUV model
    python -m bevunify.eval +experiment=gaussianlss ckpt=<gaussianlss_sedan.pth> eval.protocol=cts \
        eval.target=suv eval.oracle_ckpt=<gaussianlss_suv.pth>

IoU is computed over cells with visibility >= 2 after thresholding sigmoid(logit) at 0.5.
vp and cts write one JSON per cell to <eval.out_dir>/results/, skip cells already computed
with the same checkpoint and subset, and print the scores once a grid is complete.
"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from bevunify import ROOT
from bevunify.carla_data import build_datasets, sedan_rig_delta, subset_tokens
from bevunify.common import build_model, file_sha256, load_weights, setup_config
from bevunify.robustness import (VP_CONDITIONS, TARGETS, aggregate, cell_path, cts_cells,
                                 format_summary, load_results, parse_vp_cell, vp_cells)

log = logging.getLogger(__name__)


class ThreadedLoader:
    """Batches samples with a thread pool in the main process.

    The robustness protocols change the shared transform between cells; worker processes
    of a DataLoader would keep a stale copy, threads see every change.
    """

    def __init__(self, dataset, batch_size, num_threads):
        self.dataset = dataset
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=num_threads)

    def __iter__(self):
        n = len(self.dataset)
        for start in range(0, n, self.batch_size):
            indices = range(start, min(start + self.batch_size, n))
            yield default_collate(list(self.executor.map(self.dataset.__getitem__, indices)))


@torch.no_grad()
def run_inference(model, loader, threshold):
    """Returns (IoU over vis>=2 cells, IoU over all cells, number of frames)."""
    eps = 1e-9
    tp_v = fp_v = fn_v = 0.0
    tp_a = fp_a = fn_a = 0.0
    n_frames = 0
    for batch in loader:
        batch_cuda = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        prob = torch.sigmoid(model(batch_cuda)["vehicle"][:, 0]).cpu()
        gt = batch["vehicle"][:, 0]
        vis = batch["vehicle_visibility"]

        pred_bin = (prob > threshold).float()
        gt_bin = (gt > 0.5).float()
        mask_v = (vis >= 2).float()
        tp_v += (pred_bin * gt_bin * mask_v).sum().item()
        fp_v += (pred_bin * (1 - gt_bin) * mask_v).sum().item()
        fn_v += ((1 - pred_bin) * gt_bin * mask_v).sum().item()
        tp_a += (pred_bin * gt_bin).sum().item()
        fp_a += (pred_bin * (1 - gt_bin)).sum().item()
        fn_a += ((1 - pred_bin) * gt_bin).sum().item()
        n_frames += gt.shape[0]
    return tp_v / (tp_v + fp_v + fn_v + eps), tp_a / (tp_a + fp_a + fn_a + eps), n_frames


def warm_caches(datasets, ev):
    """Pre-load the labels, and optionally the unperturbed images, of every sample."""
    transform = datasets[0].transform
    samples = [s for d in datasets for s in d.samples]
    with ThreadPoolExecutor(max_workers=ev.num_threads) as executor:
        transform.gt_cache = {}
        list(executor.map(transform.get_bev, samples))
        if ev.cache_images:
            transform.image_cache = {}
            hooks = transform.vp_variant, transform.cts_ext_override
            transform.vp_variant, transform.cts_ext_override = None, None
            list(executor.map(transform.get_cameras, samples))
            transform.vp_variant, transform.cts_ext_override = hooks
    log.info(f"cached {len(samples)} samples")


def selected(cells, requested):
    if requested == "all":
        return cells
    requested = [requested] if isinstance(requested, str) else list(requested)
    unknown = [c for c in requested if c not in cells]
    if unknown:
        raise ValueError(f"cells not in this protocol: {unknown}")
    return [c for c in cells if c in requested]


def is_done(results_dir, cell, sha, subset):
    path = cell_path(results_dir, cell)
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    return record["checkpoint_sha256"] == sha and record["subset"] == subset


def write_cell(results_dir, cell, iou, n_frames, sha, subset):
    record = dict(cell=cell, metric="IoU", value=iou, n_frames=n_frames, checkpoint_sha256=sha, subset=subset)
    path = cell_path(results_dir, cell)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(path)
    log.info(f"{cell}: IoU {iou:.4f} ({n_frames} frames)")


def run_cells(cfg, model, datasets, cells, sha, setup_cell):
    ev = cfg.eval
    results_dir = Path(ev.out_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    pending = [c for c in cells if not is_done(results_dir, c, sha[c], ev.subset)]
    log.info(f"{len(cells) - len(pending)} of {len(cells)} cells already done; results in {results_dir}")
    if not pending:
        return
    warm_caches(datasets, ev)
    loader = ThreadedLoader(ConcatDataset(datasets), cfg.loader.val_batch_size, ev.num_threads)
    for cell in tqdm(pending, desc=ev.protocol):
        setup_cell(cell)
        iou, _, n_frames = run_inference(model, loader, ev.threshold)
        write_cell(results_dir, cell, iou, n_frames, sha[cell], ev.subset)


def run_normal(cfg, model):
    ev, platform = cfg.eval, cfg.data.platform
    tokens = subset_tokens(cfg.data.dataset_dir, platform, ev.subset)
    datasets = build_datasets(cfg.data, "val", tokens=tokens)
    loader = ThreadedLoader(ConcatDataset(datasets), cfg.loader.val_batch_size, ev.num_threads)
    iou, iou_all, n_frames = run_inference(model, loader, ev.threshold)
    record = dict(platform=platform, metric="IoU", value=iou, iou_all_cells=iou_all, n_frames=n_frames,
                  checkpoint_sha256=file_sha256(cfg.ckpt), subset=ev.subset)
    out = Path(ev.out_dir) / f"normal_{platform}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    log.info(f"{platform}: IoU (vis>=2) {iou:.4f}, IoU (all cells) {iou_all:.4f}, {n_frames} frames -> {out}")


def run_vp(cfg, model):
    ev = cfg.eval
    if cfg.data.platform != "sedan":
        raise ValueError("viewpoint perturbation is defined on the Sedan rig: use data.platform=sedan")
    cells = selected(vp_cells(), ev.cells)
    tokens = subset_tokens(cfg.data.dataset_dir, "sedan", ev.subset)
    datasets = build_datasets(cfg.data, "val", tokens=tokens)
    transform = datasets[0].transform
    sha = dict.fromkeys(cells, file_sha256(cfg.ckpt))

    def setup_cell(cell):
        if cell == "vp/NORMAL":
            transform.set_viewpoint(None, False, False)
            return
        condition, mode, variant = parse_vp_cell(cell)
        image_swap, extrinsic_swap = VP_CONDITIONS[condition]
        transform.set_viewpoint(variant, image_swap, extrinsic_swap, None if mode == "all" else [mode])

    run_cells(cfg, model, datasets, cells, sha, setup_cell)


def run_cts(cfg, model):
    ev = cfg.eval
    if ev.target not in TARGETS:
        raise ValueError(f"set eval.target to one of {TARGETS}")
    cells = selected(cts_cells(ev.target), ev.cells)
    labels_root = Path(cfg.data.labels_root)
    target_labels = labels_root / f"{ev.target}_eval"
    tokens = subset_tokens(cfg.data.dataset_dir, ev.target, ev.subset)
    datasets = build_datasets(cfg.data, "val", labels_dir=target_labels, tokens=tokens)
    transform = datasets[0].transform
    sedan_extrinsics = sedan_rig_delta(labels_root / "sedan_eval", target_labels)

    oracle = f"cts/{ev.target}/ORACLE"
    if oracle in cells and ev.oracle_ckpt is None:
        raise ValueError(f"{oracle} needs eval.oracle_ckpt, the checkpoint trained on {ev.target}")
    model_sha = file_sha256(cfg.ckpt)
    sha = {c: (file_sha256(ev.oracle_ckpt) if c == oracle else model_sha) for c in cells}

    def setup_cell(cell):
        condition = cell.split("/")[-1]
        # IMG: target images with the Sedan extrinsics; CAL and ORACLE: target extrinsics.
        transform.cts_ext_override = sedan_extrinsics if condition == "IMG" else None
        if condition == "ORACLE":
            load_weights(model, ev.oracle_ckpt)

    run_cells(cfg, model, datasets, cells, sha, setup_cell)


@hydra.main(version_base="1.3", config_path=str(ROOT / "config"), config_name="config")
def main(cfg):
    torch.set_float32_matmul_precision("high")
    setup_config(cfg)
    if cfg.ckpt is None:
        raise ValueError("set ckpt=<checkpoint file>")
    model = load_weights(build_model(cfg), cfg.ckpt).eval().cuda()

    protocol = cfg.eval.protocol
    if protocol == "normal":
        run_normal(cfg, model)
        return
    if protocol == "vp":
        run_vp(cfg, model)
    elif protocol == "cts":
        run_cts(cfg, model)
    else:
        raise ValueError(f"unknown eval.protocol {protocol!r} (normal | vp | cts)")

    summary, incomplete = aggregate(load_results(Path(cfg.eval.out_dir) / "results"))
    for message in incomplete:
        log.info(f"incomplete grid, not scored: {message}")
    log.info("\n" + format_summary(summary))


if __name__ == "__main__":
    main()
