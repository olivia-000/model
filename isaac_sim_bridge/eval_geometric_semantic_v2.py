"""Offline check of geometric_semantic_v2 against the ground-truth (aabbs) semantic images saved by
data collection. Usage: python eval_geometric_semantic_v2.py <out_dir> <scene_dir> [<scene_dir> ...]"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geometric_semantic_v2 as gs  # noqa: E402

N_SAMPLES = 300
CAM_HEIGHT_ABOVE_FEET = 0.47  # camera z when standing on flat floor (live: leg FK + camera mount)
# depth at/below this is invalid. 1.01 for data collected with Isaac Sim's default 1.0m near clip
# (closer surfaces are cut away -- rays see through them, e.g. to the floor slab's underside);
# 0.35 for CAMERA_NEAR_CLIP=0.2 data, where only the dog's own head/legs come in that close.
MIN_RANGE = float(os.environ.get("MIN_RANGE", "1.01"))
HFOV = 87.0
# Railing (76,0,153) is NOT scored (explicit user decision, 2026-10-02): a railing is just thin posts,
# and depth-only semantics correctly call the posts obstacles and the floor seen between/behind them
# floor. The ground-truth railing boxes are coarse solid volumes (posts + gaps + the strip of tread /
# landing they overlap), so they can't tell right from wrong there.
GT_COLORS = {(0, 204, 0): gs.CLS_FLOOR, (0, 153, 0): gs.CLS_STAIRS, (102, 0, 204): gs.CLS_OBSTACLE}
NAMES = ["obstacle", "floor", "stairs"]

out_dir, scene_dirs = sys.argv[1], sys.argv[2:]
os.makedirs(out_dir, exist_ok=True)
rng = np.random.default_rng(0)
samples = []
for d in scene_dirs:
    ext = np.loadtxt(os.path.join(d, "camera_extrinsic.txt"), delimiter=",")
    samples += [(d, i, ext[i]) for i in range(len(ext))]
pick = rng.choice(len(samples), size=min(N_SAMPLES, len(samples)), replace=False)

conf = np.zeros((3, 3), dtype=np.int64)  # [gt, pred]
# Pixels above the camera are excluded: in this stacked switchback stairwell the AABB ground truth
# labels the UNDERSIDE of the flight overhead as stairs (it falls inside that flight's tread boxes),
# so it can't be scored -- and nothing above the camera is directly walkable anyway.
overhead_px = 0
# Second breakdown by surface orientation (from the depth itself): on horizontal surfaces the ground
# truth is reliable (floor top / tread top / wall top); on vertical ones it is debatable by
# construction -- a landing's or floor slab's SIDE face sits inside its "floor" box, a riser inside
# its "stairs" box.
conf_h = np.zeros((3, 3), dtype=np.int64)
conf_v = np.zeros((3, 3), dtype=np.int64)


def surface_horizontal(rel):
    H, W, _ = rel.shape
    k = gs.NORMAL_STEP
    out = np.zeros((H, W), dtype=bool)
    n = np.cross(rel[k:-k, 2 * k:] - rel[k:-k, :-2 * k], rel[2 * k:, k:-k] - rel[:-2 * k, k:-k])
    with np.errstate(invalid="ignore", divide="ignore"):
        out[k:-k, k:-k] = np.abs(n[..., 2]) / np.linalg.norm(n, axis=-1) > gs.HORIZ_MIN
    return out
for n, k in enumerate(pick):
    d, i, e = samples[k]
    depth = np.load(os.path.join(d, "depth", f"{i}.npy")).squeeze()
    gt_rgb = np.array(Image.open(os.path.join(d, "semantics", f"{i}.png")).convert("RGB"))
    gt = np.full(depth.shape, -1, dtype=np.int8)
    for c, v in GT_COLORS.items():
        gt[(gt_rgb == c).all(-1)] = v
    pred, rel = gs.classify(depth, e[3:7], CAM_HEIGHT_ABOVE_FEET, HFOV, min_range=MIN_RANGE)
    over = rel[..., 2] > CAM_HEIGHT_ABOVE_FEET + 0.1
    overhead_px += int(((gt >= 0) & over).sum())
    m = (gt >= 0) & ~over & (depth > MIN_RANGE)  # GT is just as wrong on pinned-near pixels
    np.add.at(conf, (gt[m], pred[m]), 1)
    hz = surface_horizontal(rel)
    np.add.at(conf_h, (gt[m & hz], pred[m & hz]), 1)
    np.add.at(conf_v, (gt[m & ~hz], pred[m & ~hz]), 1)
    if n < 12:
        dvis = np.clip(np.nan_to_num(depth, posinf=8.0) / 8.0 * 255, 0, 255).astype(np.uint8)
        row = np.concatenate([np.stack([dvis] * 3, -1), gt_rgb, gs.to_color(pred)], axis=1)
        Image.fromarray(row).resize((row.shape[1] * 2, row.shape[0] * 2), Image.NEAREST).save(
            os.path.join(out_dir, f"cmp_{n:02d}_{os.path.basename(d)}_{i}_z{e[2]:.1f}.png"))

print(f"{len(pick)} frames, {conf.sum()} labelled pixels scored ({overhead_px} overhead pixels excluded)")


def report(title, c):
    print(f"=== {title}: {c.sum()} px")
    print(f"{'gt/pred':>10}" + "".join(f"{n:>11}" for n in NAMES))
    for g in range(3):
        print(f"{NAMES[g]:>10}" + "".join(f"{c[g, p]:>11}" for p in range(3)))
    print(f"pixel accuracy: {np.trace(c) / max(c.sum(), 1):.3f}")
    for k in range(3):
        print(f"  {NAMES[k]:>8}: recall {c[k, k] / max(c[k].sum(), 1):.3f}  precision {c[k, k] / max(c[:, k].sum(), 1):.3f}")
    print(f"UNSAFE: ground-truth obstacle predicted walkable: {(c[0, 1] + c[0, 2]) / max(c[0].sum(), 1):.3%}")


report("ALL pixels", conf)
report("HORIZONTAL surfaces", conf_h)
report("VERTICAL surfaces", conf_v)
