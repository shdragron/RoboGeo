"""Robustness evaluation cells and scores (paper Eqs. 5-7).

Cell ids follow the shared RoboGeo per-cell format:

    vp/NORMAL
    vp/<EXT|IMG|CAL>/<CAM_FRONT_LEFT|...|CAM_BACK_RIGHT|all>/<roll|pitch|yaw><+|-><deg>
    cts/<suv|bus>/<IMG|CAL|ORACLE>

Each result file <results>/<cell id with '/' replaced by '__'>.json holds
{"cell", "metric", "value", "n_frames", "checkpoint_sha256", "subset"}.
"""
import json
import re
from pathlib import Path

import numpy as np

CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
MODES = CAMERAS + ["all"]
AXES = ("roll", "pitch", "yaw")
ANGLES = (-20, -16, -12, -8, -4, 4, 8, 12, 16, 20)
# condition -> (load the rotated images, apply the rotation to the extrinsics)
VP_CONDITIONS = {"EXT": (False, True), "IMG": (True, False), "CAL": (True, True)}
CTS_CONDITIONS = ("IMG", "CAL")
TARGETS = ("suv", "bus")


def vp_cells():
    cells = ["vp/NORMAL"]
    for condition in VP_CONDITIONS:
        for mode in MODES:
            for axis in AXES:
                for angle in ANGLES:
                    cells.append(f"vp/{condition}/{mode}/{axis}{angle:+d}")
    return cells


def cts_cells(target):
    return [f"cts/{target}/{c}" for c in (*CTS_CONDITIONS, "ORACLE")]


def parse_vp_cell(cell):
    """'vp/IMG/CAM_FRONT/pitch+12' -> ('IMG', 'CAM_FRONT', 'yaw0pitch12roll0')."""
    _, condition, mode, perturbation = cell.split("/")
    axis, angle = re.fullmatch(r"(roll|pitch|yaw)([+-]\d+)", perturbation).groups()
    angles = {"yaw": 0, "pitch": 0, "roll": 0}
    angles[axis] = int(angle)
    return condition, mode, f"yaw{angles['yaw']}pitch{angles['pitch']}roll{angles['roll']}"


def cell_path(results_dir, cell):
    return Path(results_dir) / (cell.replace("/", "__") + ".json")


def load_results(results_dir):
    records = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        record = json.loads(path.read_text())
        records[record["cell"]] = record
    return records


def aggregate(records, allow_partial=False):
    """Scores from per-cell records. Returns (summary, incomplete).

    VRS_c(m) = mean over the 30 (axis, angle) of P / P_NORMAL; mVRS_c = mean of VRS_c over
    the 7 modes (6 cameras and all); CTS_c(r) = P_c(r) / P_ORACLE(r) for c in IMG, CAL.
    A block (vp, or cts/<target>) with missing cells is listed in ``incomplete`` and left
    out of the summary unless ``allow_partial`` is set.
    """
    values = {cell: float(r["value"]) for cell, r in records.items()}
    subsets = sorted({r.get("subset", "full") for r in records.values()})
    if len(subsets) > 1:
        raise ValueError(f"results mix evaluation subsets {subsets}")
    summary = {"subset": subsets[0] if subsets else None, "partial": False}
    incomplete = []

    if any(cell.startswith("vp/") for cell in values):
        missing = [c for c in vp_cells() if c not in values]
        if missing:
            incomplete.append(f"vp: {len(missing)} of {len(vp_cells())} cells missing")
        if "vp/NORMAL" in values and (not missing or allow_partial):
            p_normal = values["vp/NORMAL"]
            vrs, mvrs = {}, {}
            for condition in VP_CONDITIONS:
                vrs[condition] = {}
                for mode in MODES:
                    cells = [f"vp/{condition}/{mode}/{a}{t:+d}" for a in AXES for t in ANGLES]
                    found = [values[c] / p_normal for c in cells if c in values]
                    if found:
                        vrs[condition][mode] = float(np.mean(found))
                if vrs[condition]:
                    mvrs[condition] = float(np.mean(list(vrs[condition].values())))
            summary.update(P_NORMAL=p_normal, VRS=vrs, mVRS=mvrs)
            summary["partial"] |= bool(missing)

    for target in TARGETS:
        cells = cts_cells(target)
        if not any(c in values for c in cells):
            continue
        missing = [c for c in cells if c not in values]
        if missing:
            incomplete.append(f"cts/{target}: missing {', '.join(missing)}")
        oracle = values.get(f"cts/{target}/ORACLE")
        if oracle is not None and (not missing or allow_partial):
            summary.setdefault("P_oracle", {})[target] = oracle
            summary.setdefault("CTS", {})[target] = {
                c: values[f"cts/{target}/{c}"] / oracle for c in CTS_CONDITIONS if f"cts/{target}/{c}" in values}
            summary["partial"] |= bool(missing)

    return summary, incomplete


def format_summary(summary):
    lines = [f"subset: {summary['subset']}" + ("  (fps16 is for quick checks only)" if summary["subset"] == "fps16" else "")]
    if "mVRS" in summary:
        lines.append(f"P_NORMAL: {summary['P_NORMAL']:.4f}")
        lines.append("mVRS (%): " + "  ".join(f"{c} {100 * v:.1f}" for c, v in summary["mVRS"].items()))
    for target, scores in summary.get("CTS", {}).items():
        lines.append(f"P_oracle {target}: {summary['P_oracle'][target]:.4f}   CTS (%): "
                     + "  ".join(f"{c} {100 * v:.1f}" for c, v in scores.items()))
    if summary["partial"]:
        lines.append("partial: some cells are missing, scores are not final")
    return "\n".join(lines)
