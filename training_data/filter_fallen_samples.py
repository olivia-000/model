"""Drop collected samples whose camera fell out of the scene (camera z < Z_MIN), then renumber the
rest contiguously -- viplanner's dataset pairs line i of camera_extrinsic.txt with depth/i.npy and
semantics/i.png, so there can't be gaps. Original extrinsics kept as camera_extrinsic_unfiltered.txt.
Usage: python filter_fallen_samples.py <scene_dir> [z_min]"""
import os
import shutil
import sys

import numpy as np

root = sys.argv[1]
z_min = float(sys.argv[2]) if len(sys.argv) > 2 else -0.5
ext_path = os.path.join(root, "camera_extrinsic.txt")
ext = np.loadtxt(ext_path, delimiter=",")
keep = np.where(ext[:, 2] >= z_min)[0]
print(f"[filter] {len(ext)} samples, dropping {len(ext) - len(keep)} with camera z < {z_min}")
if len(keep) == len(ext):
    sys.exit(0)

shutil.copy(ext_path, os.path.join(root, "camera_extrinsic_unfiltered.txt"))
for sub, suffix in (("depth", ".npy"), ("semantics", ".png")):
    src, tmp = os.path.join(root, sub), os.path.join(root, sub + "_filtered")
    os.makedirs(tmp)
    for new_i, old_i in enumerate(keep):
        os.rename(os.path.join(src, f"{old_i}{suffix}"), os.path.join(tmp, f"{new_i}{suffix}"))
    shutil.rmtree(src)  # only the dropped samples are left in it
    os.rename(tmp, src)
np.savetxt(ext_path, ext[keep], delimiter=",")
print(f"[filter] kept {len(keep)} samples")
