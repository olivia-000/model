"""Offline Mask2Former check on images saved by the sim (SAVE_RGB=1 COLLECT=1).

Runs a COCO-panoptic Mask2Former on <data>/rgb/N.png, maps its classes to ViPlanner's semantic
classes with viplanner_core's own table (config/coco_sem_meta.py), and compares against the
ground-truth semantics/N.png the sim saved alongside.

Run in the `mask2former_env` conda env (kept apart from viplanner_train / the Isaac container):
    python m2f_test.py <data_dir> [--model facebook/mask2former-swin-tiny-coco-panoptic]
                       [--every 5] [--out <dir>] [--images a.jpg b.jpg]   # --images: no ground truth
Outputs: <out>/N.png side-by-side (rgb | Mask2Former | ground truth) and a printed summary.
"""
import argparse
import glob
import importlib.util
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "viplanner_core", "viplanner", "config")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_CFG, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


coco_meta = _load("coco_sem_meta")
vip_meta = _load("viplanner_sem_meta")
VIP_COLOR = {d["name"]: tuple(d["color"]) for d in vip_meta.VIPLANNER_SEM_META}

# ground-truth colours written by camera_semantics_lab.SEG_CLASS_COLOR
GT = {"floor": (0, 204, 0), "stairs": (0, 153, 0), "wall": (102, 0, 204), "railing": (76, 0, 153)}
# ViPlanner classes that count as "walkable ground" / "stairs"; everything else is an obstacle
WALKABLE = {"floor", "sidewalk", "road", "gravel", "sand", "snow", "terrain", "crosswalk"}


def build_class_table(id2label):
    """model class id -> ViPlanner class name (same keyword rule as coco_sem_meta.get_class_for_id)."""
    table = {}
    for idx, coco_name in id2label.items():
        hit = None
        for vip_name, keywords in coco_meta._COCO_MAPPING.items():
            if any(k in coco_name for k in keywords):
                hit = vip_name
                break
        table[int(idx)] = hit or "static"
    return table


def segment(model, proc, img, device):
    inputs = proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_panoptic_segmentation(out, target_sizes=[img.size[::-1]])[0]
    seg = res["segmentation"].cpu().numpy()
    cls = np.full(seg.shape, -1, dtype=np.int32)
    for info in res["segments_info"]:
        cls[seg == info["id"]] = info["label_id"]
    return cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?")
    ap.add_argument("--model", default="facebook/mask2former-swin-tiny-coco-panoptic")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--images", nargs="*", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(args.model)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model).to(device).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    table = build_class_table(id2label)

    if args.images:
        files, has_gt = args.images, False
        out_dir = args.out or "m2f_out"
    else:
        files = sorted(glob.glob(os.path.join(args.data_dir, "rgb", "*.png")),
                       key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))[::args.every]
        has_gt = True
        out_dir = args.out or os.path.join(args.data_dir, "m2f_" + args.model.split("/")[-1])
    os.makedirs(out_dir, exist_ok=True)
    print(f"model {args.model} on {device}, {len(files)} images -> {out_dir}")

    # confusion: ground-truth class -> {stairs, walkable, obstacle, none}
    conf = {g: {"stairs": 0, "walkable": 0, "obstacle": 0, "none": 0} for g in GT}
    coco_on_stairs = {}
    frames_with_stairs = frames_stairs_found = 0
    times = []
    for f in files:
        img = Image.open(f).convert("RGB")
        t0 = time.time()
        cls = segment(model, proc, img, device)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)

        vip = np.empty(cls.shape, dtype=object)
        color = np.zeros(cls.shape + (3,), dtype=np.uint8)
        kind = np.full(cls.shape, 3, dtype=np.uint8)  # 0 stairs 1 walkable 2 obstacle 3 none
        for cid in np.unique(cls):
            m = cls == cid
            if cid < 0:
                continue
            name = table[int(cid)]
            vip[m] = name
            color[m] = VIP_COLOR.get(name, (0, 0, 0))
            kind[m] = 0 if name == "stairs" else (1 if name in WALKABLE else 2)

        panels = [np.asarray(img), color]
        if has_gt:
            gt_path = os.path.join(args.data_dir, "semantics", os.path.basename(f))
            gt = np.asarray(Image.open(gt_path).convert("RGB"))
            panels.append(gt)
            for g, c in GT.items():
                m = np.all(gt == np.array(c, dtype=np.uint8), axis=-1)
                if not m.any():
                    continue
                k = kind[m]
                for i, key in enumerate(("stairs", "walkable", "obstacle", "none")):
                    conf[g][key] += int((k == i).sum())
                if g == "stairs":
                    if m.mean() > 0.03:  # stairs fill >3% of the frame
                        frames_with_stairs += 1
                        if (k == 0).mean() > 0.3:
                            frames_stairs_found += 1
                    ids, cnt = np.unique(cls[m], return_counts=True)
                    for i, n in zip(ids, cnt):
                        nm = id2label.get(int(i), "(nothing)")
                        coco_on_stairs[nm] = coco_on_stairs.get(nm, 0) + int(n)
        Image.fromarray(np.concatenate(panels, axis=1)).save(os.path.join(out_dir, os.path.basename(f)))

    t = np.array(times[1:] or times)
    print(f"inference: {t.mean()*1000:.0f} ms/image (~{1/t.mean():.1f} Hz) at {img.size[0]}x{img.size[1]}")
    if has_gt:
        print("ground truth -> Mask2Former says   stairs  walkable  obstacle  nothing")
        for g in GT:
            tot = sum(conf[g].values())
            if tot:
                r = conf[g]
                print(f"  {g:8s} ({tot/1e6:6.1f} Mpx)        "
                      + "  ".join(f"{100*r[k]/tot:6.1f}%" for k in ("stairs", "walkable", "obstacle", "none")))
        print(f"frames where stairs fill >3%: {frames_with_stairs}; "
              f"Mask2Former labels >30% of those stair pixels as stairs in {frames_stairs_found}")
        tot = sum(coco_on_stairs.values()) or 1
        print("what Mask2Former calls the real stair pixels:")
        for nm, n in sorted(coco_on_stairs.items(), key=lambda x: -x[1])[:8]:
            print(f"  {100*n/tot:5.1f}%  {nm}")


if __name__ == "__main__":
    main()
