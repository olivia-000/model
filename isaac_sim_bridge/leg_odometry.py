"""Body height from the legs alone (user request 2026-10-14: "先做腿部里程計估高度").

Everything here is what the real robot has: joint encoders, the robot's own URDF (link geometry),
the IMU (gravity direction and vertical acceleration) and joint torques (motor current). No world
pose from the simulator.

Idea: a planted foot does not move in the world, so its height below the base (along gravity)
changes only because the base moved. That velocity is fused with the IMU's vertical acceleration in
a small Kalman filter (LegHeightOdometry), which also carries the estimate through moments with no
foot reliably down.

Contact is taken from the joint torques (foot_up_forces): the sim's foot ContactSensors read
"no foot down" ~30% of the time on growin and gave metres of error.

Simulation note (2026-10-15): run with PHYSICS_HZ=500. At the default 60 Hz physics the torques the
sim reports are coarse and the same estimator drifts by metres.

The old estimate in viplanner_control (go2_foot_positions_body_frame) used Go2's link lengths and
axis signs and ignored body tilt; on growin it ran the wrong way (-10.6 m for a real +8.6 m climb).
Here the leg chain is read from the URDF itself, and heights are taken along gravity, not body z.
"""
import xml.etree.ElementTree as ET

import numpy as np

LEGS = ("FL", "FR", "RL", "RR")
CHAIN = ("hip_joint", "thigh_joint", "calf_joint", "foot_joint")


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, a):
    k = np.asarray(axis, dtype=np.float64)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


class LegKinematics:
    """Foot positions in the base frame from joint angles, chain taken from a URDF."""

    def __init__(self, urdf_path):
        joints = {j.get("name"): j for j in ET.parse(urdf_path).getroot().findall("joint")}
        self.chains = []
        for leg in LEGS:
            chain = []
            for part in CHAIN:
                j = joints[f"{leg}_{part}"]
                o = j.find("origin")
                xyz = np.array([float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()])
                rpy = [float(v) for v in ((o.get("rpy") if o is not None else None) or "0 0 0").split()]
                a = j.find("axis")
                axis = [float(v) for v in a.get("xyz").split()] if (a is not None and j.get("type") != "fixed") else None
                chain.append((xyz, _rpy(*rpy), axis))
            self.chains.append(chain)

    def feet(self, dof_pos):
        """dof_pos: (12,) hip/thigh/calf per leg in FL, FR, RL, RR order -> (4, 3) in the base frame."""
        out = np.zeros((4, 3))
        for i, chain in enumerate(self.chains):
            p, R, k = np.zeros(3), np.eye(3), 0
            for xyz, R_o, axis in chain:
                p = p + R @ xyz
                R = R @ R_o
                if axis is not None:
                    R = R @ _axis_angle(axis, float(dof_pos[i * 3 + k]))
                    k += 1
            out[i] = p
        return out

    def jacobians(self, dof_pos, eps=1e-4):
        """(4, 3, 3): d foot position (base frame) / d (hip, thigh, calf) per leg, numerically."""
        q = np.asarray(dof_pos, dtype=np.float64)
        base = self.feet(q)
        J = np.zeros((4, 3, 3))
        for k in range(12):
            dq = q.copy()
            dq[k] += eps
            J[k // 3, :, k % 3] = (self.feet(dq)[k // 3] - base[k // 3]) / eps
        return J


def foot_up_forces(kin, dof_pos, tau, gravity_body):
    """Ground push on each foot along "up" (N) from joint torques (motor current on the real robot):
    standing still, the motors' torque balances the ground force, tau = -J^T F."""
    g = np.asarray(gravity_body, dtype=np.float64)
    g = g / np.linalg.norm(g)
    J = kin.jacobians(dof_pos)
    tau = np.asarray(tau, dtype=np.float64).reshape(4, 3)
    out = np.zeros(4)
    for i in range(4):
        try:
            F = -np.linalg.solve(J[i].T, tau[i])
        except np.linalg.LinAlgError:
            continue
        out[i] = -(F @ g)
    return out


class LegHeightOdometry:
    """Base height: a vertical Kalman filter. The IMU's vertical acceleration predicts; the planted
    feet's kinematic velocity corrects.

    A foot is used only if it pushes the ground with more than contact_force (joint torque) now and
    last tick, AND its own vertical velocity under the prediction (base velocity + its velocity
    relative to the base) is within still_vel of zero. Climbing, a foot pushing off rises while still
    loaded; going down, a foot that has just landed is still sinking onto the step -- one shared
    3-sigma gate on the median let those through and the estimate ended 1 m high after one flight
    down (2026-10-15). If loaded feet keep failing that test for recover_ticks, the prediction itself
    is off: their median is taken as is.
    Offline (500 Hz recordings, 0.5 m/s^2 noise and +-0.1 m/s^2 bias on the acceleration): 2.3 m down
    -0.2 m, 3 m up -0.16 m, 9.2 m up -0.6 m."""

    def __init__(self, urdf_path, contact_force=200.0, still_vel=0.2, acc_std=5.0, vel_std=0.1,
                 recover_ticks=15):
        self.kin = LegKinematics(urdf_path)
        self.contact_force, self.still_vel = contact_force, still_vel
        self.acc_std, self.vel_std, self.recover_ticks = acc_std, vel_std, recover_ticks
        self.reset(0.0)

    def reset(self, base_z):
        self.x = np.array([float(base_z), 0.0])   # base height, vertical velocity (odometry frame)
        self.P = np.diag([1e-4, 1e-2])
        self.prev_h = None                        # (4,) foot heights relative to the base, last tick
        self.prev_f = None                        # (4,) foot push forces, last tick
        self.planted = np.zeros(4, dtype=bool)
        self.rejected = 0                         # ticks with loaded feet but none still
        self.recovered = 0
        self._miss = 0

    @property
    def base_z(self):
        return float(self.x[0])

    def update(self, dof_pos, gravity_body, joint_torque, acc_up, dt):
        """gravity_body: gravity direction in the base frame (IMU); joint_torque: (12,) N m (motor
        current); acc_up: base acceleration along "up" with gravity removed (IMU); dt: s since the
        last call. Returns the base height."""
        g = np.asarray(gravity_body, dtype=np.float64)
        g = g / np.linalg.norm(g)
        h = -(self.kin.feet(dof_pos) @ g)
        f = foot_up_forces(self.kin, dof_pos, joint_torque, g)
        A = np.array([[1.0, dt], [0.0, 1.0]])
        B = np.array([0.5 * dt * dt, dt])
        self.x = A @ self.x + B * float(acc_up)
        self.P = A @ self.P @ A.T + self.acc_std ** 2 * np.outer(B, B)
        self.planted[:] = False
        if self.prev_h is not None and dt > 0:
            loaded = (f > self.contact_force) & (self.prev_f > self.contact_force)
            v = -(h - self.prev_h) / dt                      # base vertical velocity if foot k is still
            still = loaded & (np.abs(v - self.x[1]) < self.still_vel)
            if loaded.any() and not still.any():
                self.rejected += 1
                self._miss += 1
                if self._miss >= self.recover_ticks:
                    still = loaded
                    self.recovered += 1
            if still.any():
                self._miss = 0
                self.planted = still
                S = self.P[1, 1] + self.vel_std ** 2 / still.sum()
                K = self.P[:, 1] / S
                self.x = self.x + K * (float(np.median(v[still])) - self.x[1])
                self.P = self.P - np.outer(K, self.P[1, :])
        self.prev_h, self.prev_f = h, f
        return self.base_z
