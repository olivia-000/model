"""Export every collision mesh of a USD scene as world-space triangles (run with Isaac Sim's python.sh,
needs pxr). Phase 2 of the real-robot plan: the training cost map comes from the scene geometry
itself instead of hand-written aabbs.
Usage: python.sh export_collision_mesh.py <scene.usd> <out.npz>
Out: tris (N,3,3) float32 world coordinates, names (N,) index into mesh_names."""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import sys
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

src, out = sys.argv[1], sys.argv[2]
stage = Usd.Stage.Open(src)
tris, owner, names = [], [], []
for prim in stage.Traverse():
    if prim.GetTypeName() != "Mesh" or not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    mesh = UsdGeom.Mesh(prim)
    pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    M = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    wp = (np.c_[pts, np.ones(len(pts))] @ M.T)[:, :3]
    k = 0
    for c in counts:  # fan-triangulate each polygon
        f = idx[k:k + c]
        for t in range(1, c - 1):
            tris.append(wp[[f[0], f[t], f[t + 1]]])
            owner.append(len(names))
        k += c
    names.append(prim.GetName())
np.savez_compressed(out, tris=np.asarray(tris, dtype=np.float32), owner=np.asarray(owner, dtype=np.int32),
                    mesh_names=np.asarray(names))
print(f"[export] {len(names)} collision meshes, {len(tris)} triangles -> {out}")
app.close()
