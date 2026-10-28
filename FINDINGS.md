# HIMLoco Gait Verification — Findings from IsaacLab Testing (2026-09-08)

Summary of everything confirmed while testing `checkpoints/low_level_policies/user_gait_0908/policy.pt`
inside IsaacLab/PhysX, for porting the verification to Isaac Gym. Written so none of this needs to be
re-discovered from scratch on the other machine.

## Checkpoint architecture (confirmed via TorchScript graph inspection)

`policy.pt` is `PolicyExporterHIM` (from `/home/itri/HIMLoco/legged_gym/legged_gym/utils/helpers.py`
per the embedded graph source references) — a full HIM (Hybrid Internal Model) policy, NOT a plain
actor:

- `estimator(obs_history_270dim)` -> output vector, of which:
  - columns `[0:3]` = estimated base velocity (3-dim)
  - columns `[3:19]` = latent embedding `z` (16-dim), L2-normalized before use
- `actor` input = `concat([latest_single_step_obs (45-dim), estimated_vel (3-dim), normalized_z (16-dim)])`
  = 64-dim, NOT the full 270-dim history — the actor only ever sees the LATEST single step directly,
  plus the estimator's compressed summary of the history.
- Both `actor` and `estimator` are `Sequential` MLPs with `ELU` activations (actor: 4 Linear layers /
  3 ELU; estimator: 3 Linear layers / 2 ELU — see `himloco_navigation_actions.py`'s own docstring for
  how the observation is assembled going INTO this whole module; the estimator/actor split above
  happens INSIDE the exported TorchScript module itself, so any correct Isaac Gym port just needs to
  feed the same 270-dim flattened history in and treat the whole thing as one black box — do not
  reimplement the estimator/actor split by hand unless retraining).

## Observation contract (45-dim per step, 6-step history = 270-dim, "time" priority flattening)

Order (confirmed against `/home/ashain/Desktop/config.yaml`'s `observations` field, which matches this
checkpoint's own `env_config.json` on every value that could be cross-checked):

```
[commands(3), ang_vel(3), gravity_vec(3), dof_pos(12), dof_vel(12), prev_actions(12)] = 45
```

- `commands` = `[lin_vel_x, lin_vel_y, ang_vel_yaw] * commands_scale`, `commands_scale = [2.0, 2.0, 0.25]`
- `ang_vel` = base angular velocity (body frame) `* ang_vel_scale (0.25)`
- `gravity_vec` = projected gravity vector (body frame), no scale
- `dof_pos` = `(joint_pos - default_joint_pos) * dof_pos_scale (1.0)`, in **policy joint order** (see
  below), default_joint_pos matches `env_config.json`'s `init_state.default_joint_angles`
- `dof_vel` = `joint_vel * dof_vel_scale (0.05)`, policy joint order
- `prev_actions` = the RAW policy output from the previous step, in **policy joint order** (i.e. NOT
  remapped to sim/hardware order — store whatever the network itself output last step)

History flattening: **"time" priority** — the 270-dim vector is 6 consecutive 45-dim blocks, each
block being one full timestep (`[obs_t0(45), obs_t-1(45), ..., obs_t-5(45)]`), NOT grouped by term
across time. On episode start (no real history yet), fill all 6 slots with the first real observation
(not zeros) — filling with zeros gives the estimator a nonsensical "nothing changed for 6 steps" signal.

`clip_obs = 100.0`, `clip_actions = 100.0` (both very permissive, essentially never binds in practice).

## Joint order (policy order vs. sim/hardware order)

Policy's own joint order (do NOT reuse `config.yaml`'s `joint_mapping` array — that's for a specific
real-hardware/SDK target's joint order, unrelated to any given simulator's own convention):

```
FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
```

Build the sim<->policy permutation by matching joint NAMES between the simulator's own joint order and
this list — never assume the simulator's native order already matches it.

`default_dof_pos` (policy order): `[0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5]`
(FL/FR/RL/RR hip/thigh/calf) — confirmed identical between `config.yaml` and this checkpoint's own
`env_config.json`.

`action_scale = 0.25` applied when converting policy output -> joint position target:
`joint_target = default_dof_pos + policy_action * 0.25` (standard offset-PD convention).

## Actuator / physical parameters — THE PART THAT ACTUALLY MATTERED

This checkpoint's own `env_config.json` specifies:
- `control.stiffness.joint = 40.0`, `control.damping.joint = 1.0` (flat, all 12 joints)
- `control.action_scale = 0.25`
- Implied by `config.yaml`'s flat `torque_limits: [33.5]*12` (same value across all 12 joints, no
  per-speed table anywhere) that the **original training used a plain flat-clip PD actuator model**,
  NOT a velocity-dependent DC-motor torque-speed curve.

**Confirmed empirically (IsaacLab, this session) that ALL THREE of the following matter, and testing
must match them exactly, or the gait degrades/destabilizes at higher commanded speed even though the
checkpoint and its wiring are otherwise completely correct:**

1. `stiffness=40.0, damping=1.0` (not some other simulator's stock default -- IsaacLab's own stock Go2
   config ships `stiffness=25.0, damping=0.5`, which is WRONG for this checkpoint and produces
   immediate, ~1.2s, bit-for-bit-repeating collapse on every episode).
2. `effort_limit / torque_limit = 33.5` per joint (IsaacLab's own stock Go2 config ships `23.5`, ~30%
   lower than trained).
3. **The actuator MODEL itself must be a flat-clip PD (`tau = clip(kp*(q-q_des) + kd*(qdot-qdot_des),
   -33.5, 33.5)`), not a DC-motor-style velocity-dependent saturation curve**
   (`tau_max(qdot) = clip(33.5*(1 - qdot/30.0), 0, 33.5)`, which is what IsaacLab's stock `DCMotorCfg`
   actuator model implements). Confirmed empirically: raising the DC-motor's effort/saturation values
   to 33.5 alone made **zero measurable difference** (bit-for-bit identical simulation trace) --
   only actually swapping the actuator class to a flat-clip PD model (IsaacLab's `IdealPDActuatorCfg`)
   changes anything. If Isaac Gym's own actuator model for this asset has an equivalent
   velocity-dependent curve, disable/bypass it the same way.

If Isaac Gym's Go2 asset/config already used a flat-clip model with these exact values (likely, since
this checkpoint was almost certainly originally trained IN Isaac Gym/legged_gym), none of the above
should need changing there — this list is specifically what had to be FIXED to deploy correctly in
IsaacLab, which defaults to different values than legged_gym's own convention.

## Speed-dependent instability — NOT fixed by any of the above, still open

Even with all three actuator parameters above matched, gait stability in IsaacLab is clearly
speed-dependent:

| commanded vx (raw, pre-scale) | observed stability |
|---|---|
| 0.0 (standing) | stable, settles ~0.32-0.35m base height, holds indefinitely |
| 0.1 | stable, ~0.30-0.34m, holds indefinitely |
| 0.3 | unstable/crouched, ~0.22-0.28m, noisy but not a hard reset within ~5s tested |
| 0.6 | **full collapse**, repeating ~1.2-4.4s collapse-and-reset cycle, foot clearance during this
  regime measured only ~0.02-0.06m (essentially no lift at all) |

This is a smooth degradation with speed, not a single threshold bug -- consistent with a genuine
sim-to-sim contact/friction/solver transfer gap between wherever this checkpoint was originally
validated and IsaacLab/PhysX specifically (all wiring, joint order, observation content, and the three
actuator parameters above were independently verified correct/matched; changing none of them further
affected the outcome). **This is the open question this verification pass on a second simulator
(Isaac Gym) is meant to help resolve**: if the SAME checkpoint is stable at vx=0.6 in Isaac Gym with
matched parameters, the gap is specifically an IsaacLab/PhysX contact-model issue, not a problem with
the checkpoint itself.

Separately, even at the LOW speeds where IsaacLab is stable (vx<=0.1), foot lift/clearance stays under
~0.06m throughout — nowhere near the ~0.15m needed to clear a real stair riser. Low speed fixes
stability but not foot clearance; this checkpoint has not been observed to produce a gait in IsaacLab
capable of clearing a 0.15m step at any tested speed.

## Stair geometry (`stair_scene/`)

`ramp_cfg.yml` + `multi_floor_scene.py`: the real target stair geometry this checkpoint needs to
eventually climb — U-shaped (switchback) staircase, two 10-step flights, each 0.15m riser x 0.30m
tread, 1.5m wide, connected by a landing. Constants: `STEP_HEIGHT=0.15, STEP_DEPTH=0.30,
STAIR_WIDTH=1.50, STEPS_PER_FLIGHT=10`. Useful if replicating the same stairs in Isaac Gym for a
matched comparison test.

## What's in this bundle

```
FINDINGS.md                                     -- this file
config.yaml                                     -- HIMLoco deployment template (obs order/scales/
                                                    joint order/PD gains/torque limits -- see above)
himloco_navigation_actions.py                   -- IsaacLab deployment wrapper reference code (the
                                                    exact, debugged-this-session obs assembly + action
                                                    application logic -- not runnable standalone
                                                    outside IsaacLab, kept as the authoritative spec)
checkpoints/low_level_policies/
  user_gait_0908/policy.pt, env_config.json     -- the checkpoint under test
  custom_gait/policy.pt                         -- older checkpoint, same architecture family
stair_scene/
  multi_floor_scene.py                          -- procedural generator for the U-shaped staircase
                                                    (constants: STEP_HEIGHT=0.15, STEP_DEPTH=0.30,
                                                    STAIR_WIDTH=1.50, STEPS_PER_FLIGHT=10)
  ramp_cfg.yml                                  -- exact per-step translation/size for every stair
                                                    block (both flights + landing), as actually built
  multi_floor.usd                               -- the built USD scene (1F/2F building shell,
                                                    corridors) the stairs sit inside -- IsaacLab/USD
                                                    format; a from-scratch Isaac Gym rebuild would use
                                                    ramp_cfg.yml's numbers directly instead of this file
  keyword_mapping.yml                           -- semantic-class-to-cost mapping used by ViPlanner's
                                                    cost-map builder (only relevant if also porting
                                                    navigation, not needed for gait-only testing)
viplanner_core/                                 -- core ViPlanner Python package (framework-agnostic
                                                    navigation logic: cost maps, path/trajectory
                                                    optimization, VIPlannerAlgo), WITHOUT the
                                                    IsaacLab/Omniverse extension -- won't run as-is
                                                    without an environment to plug into, but every
                                                    piece here has no IsaacLab dependency
  pyproject.toml                                -- package metadata/dependencies for viplanner_core
  viplanner/config/                             -- cost-map and reconstruction config dataclasses
  viplanner/cost_maps/                          -- TSDF/semantic cost-map builder (pure geometry/PCD,
                                                    no simulator dependency)
  viplanner/plannernet/                         -- ViPlanner's own trained path-planning network
  viplanner/traj_cost_opt/                      -- trajectory optimization / cost-map querying
  viplanner/utils/                              -- TopologicalPlanner, MultiFloorGoalManager,
                                                    StairConnector (this session's floor-routing work)
  viplanner/{cost_builder,depth_reconstruct,query_cost_map,train}.py -- standalone entry-point scripts
```

## What was deliberately left OUT (available in the main repo at /home/ashain/Desktop/viplanner if needed)

- `viplanner/third_party/` (836MB -- diffusion_policy, visualnav-transformer, unrelated external
  repos, not used by anything above).
- The full `omniverse/extension/omni.viplanner/` Omniverse/IsaacLab extension -- everything that
  actually RUNS `viplanner_core/` inside a simulator (scene configs, MDP action/observation/reward
  terms, the `viplanner_demo.py` entry point). This is IsaacLab/Isaac-Sim-specific and won't run in
  Isaac Gym; `himloco_navigation_actions.py` above is kept standalone as the spec of what it does, not
  as runnable code.
- The IsaacLab-native height-scan-based RL training pipeline (`go2_stairs_train_cfg.py` and friends)
  -- a completely different, incompatible observation contract (235-dim single-frame + height_scan),
  unrelated to this HIMLoco checkpoint family.
- Root-level debug/log files (`multi_floor_*.log`, `debug_stair_spawn*.py`, etc.) -- session scratch
  work, not part of the reusable codebase.
