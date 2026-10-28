"""URDF -> USD with Isaac Sim's importer (run with python.sh). Fixed joints are NOT merged: the foot
links must survive as their own prims for the per-foot contact sensors viplanner_control.py creates.
Usage: python.sh urdf_to_usd.py <in.urdf> <out.usd>"""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import sys
log = open(sys.argv[2] + ".import.txt", "w", buffering=1)
import omni.kit.commands
from pxr import Usd, UsdGeom, UsdPhysics

urdf, out = sys.argv[1], sys.argv[2]
_, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
cfg.merge_fixed_joints = False
cfg.fix_base = False
cfg.make_default_prim = True
cfg.create_physics_scene = False
cfg.import_inertia_tensor = True
cfg.self_collision = False
cfg.distance_scale = 1.0
ok, prim_path = omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=urdf, import_config=cfg, dest_path=out)
print("import ok:", ok, "prim:", prim_path, file=log)
s = Usd.Stage.Open(out)
# Post-import fixes, authored as overrides in the output's root layer:
#  - the importer creates ACCELERATION drives (torque = inertia * kp * error). A legged_gym policy's
#    kp/kd are plain PD torque gains, so with acceleration drives the 70 kg leo got ~40 Nm where
#    kp * error called for the 140 Nm limit, and collapsed onto its belly. -> force drives.
#  - links with no inertial (base, imu_link) come in with mass 0, which PhysX replaces with 1 kg each.
n_drv = n_mass = 0
for p in s.Traverse():
    if p.IsA(UsdPhysics.RevoluteJoint):
        UsdPhysics.DriveAPI.Get(p, "angular").CreateTypeAttr("force"); n_drv += 1
    if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.HasAPI(UsdPhysics.MassAPI) and not UsdPhysics.MassAPI(p).GetMassAttr().Get():
        UsdPhysics.MassAPI(p).CreateMassAttr(0.01); n_mass += 1
s.GetRootLayer().Save()
print(f"set {n_drv} revolute drives to force, {n_mass} massless links to 0.01 kg", file=log)
s = Usd.Stage.Open(out)
print("defaultPrim:", s.GetDefaultPrim().GetPath(), "| upAxis", UsdGeom.GetStageUpAxis(s), "| mpu", UsdGeom.GetStageMetersPerUnit(s), file=log)
print("articulation root:", [str(p.GetPath()) for p in s.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)], file=log)
print("revolute joints:", [p.GetName() for p in s.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)], file=log)
print("fixed joints:", [p.GetName() for p in s.Traverse() if p.IsA(UsdPhysics.FixedJoint)], file=log)
print("rigid bodies:", [str(p.GetPath()) for p in s.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)], file=log)
bc = UsdGeom.BBoxCache(0, ["default", "render"])
r = bc.ComputeWorldBound(s.GetPseudoRoot()).ComputeAlignedRange()
print("bbox (zero pose) min", [round(v, 3) for v in r.GetMin()], "max", [round(v, 3) for v in r.GetMax()], file=log)
for p in s.Traverse():
    if p.HasAPI(UsdPhysics.RigidBodyAPI):
        b = bc.ComputeWorldBound(p).ComputeAlignedRange()
        m = UsdPhysics.MassAPI(p).GetMassAttr().Get() if p.HasAPI(UsdPhysics.MassAPI) else None
        print(f"  {p.GetName():10s} bbox x[{b.GetMin()[0]:.3f},{b.GetMax()[0]:.3f}] y[{b.GetMin()[1]:.3f},{b.GetMax()[1]:.3f}] z[{b.GetMin()[2]:.3f},{b.GetMax()[2]:.3f}] mass {m}", file=log)
log.flush(); app.close()
