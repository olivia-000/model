"""Find the stairs to climb from what the robot itself sees (GUIDANCE=perception).

Input per replan tick: the semantic image (Mask2Former's "stairs" pixels), the depth points of the
same frame, the robot's pose and its feet level. No scene ground truth (aabbs) is used.

    stairs pixels + depth -> 3D points -> keep the ones reachable from the feet (rising, connected)
    -> fit a plane: its uphill direction is the flight's direction, its slope says "really stairs"
    -> entrance = lowest edge, centred across the flight's width
    -> sub-goal: line up in front of the entrance, then walk up along the flight's centreline

Points are handled in the world frame only for convenience: everything is relative to the robot, so
on a real robot "world" is just its odometry frame (IMU gives gravity, odometry gives the short-term
motion between detections).
"""
from collections import deque

import numpy as np

STAIRS_COLOR = np.array([0, 153, 0], dtype=np.uint8)  # VIPLANNER_SEM_META "stairs"

MAX_RANGE = 6.0            # m, ignore stair pixels farther than this (depth gets unreliable)
BELOW_FEET = 0.10          # m, stair points this far below the feet still count (the step being stood on)
ABOVE_FEET = 1.8           # m, ...and up to this far above (taller = another flight / a false positive)
CELL = 0.15                # m, grid cell for the connectivity check
MIN_POINTS = 150           # stair points (after subsampling) needed to trust a detection
MIN_SLOPE, MAX_SLOPE = 0.45, 1.5   # rise/run a real flight can have (0.18/0.27 = 0.67 here; 0.25 let a
#                                   distant half-seen flight through at 0.3-0.4 and led it away, 2026-10-14)
REACH_FROM_FEET = 1.0      # m, the flight's lowest VISIBLE point must be within this of the feet level (the camera cannot see the steps right at the feet)
APPROACH_DIST = 0.8        # m, line-up point in front of the entrance
LINEUP_TOL = 0.35          # m, lateral offset from the centreline that still counts as lined up
AHEAD = 1.0                # m, how far up the flight the sub-goal is placed
MEMORY_TICKS = 45          # replan ticks a detection is kept once the stairs leave the view
FLAT_PITCH = np.radians(6.0)      # body pitch below this = standing on level ground
RUNOUT = 0.9               # m walked on level ground after a flight before it counts as finished
FRONT_STOP = 1.1           # m, free space straight ahead below which a finished flight's run-out ends
DONE_MATCH = 1.0           # m, a detection this close across to a walked flight (same direction, same heights) is that flight
SMOOTH = 0.35              # blend factor for re-detections of the same flight
SEARCH_ANGLE = np.radians(100.0)  # sub-goal bearing used to turn on the spot while searching


def _largest_reachable_component(pts, feet_z, descend=False):
    """Grid-connected group of stair points that starts near the feet level (its lowest part when
    going up, its highest part when going down). Returns a bool mask."""
    ij = np.floor(pts[:, :2] / CELL).astype(np.int64)
    ij -= ij.min(axis=0)
    w = int(ij[:, 1].max()) + 1
    key = ij[:, 0] * w + ij[:, 1]
    cells, inv = np.unique(key, return_inverse=True)
    cell_set = {int(c): i for i, c in enumerate(cells)}
    zmin = np.full(len(cells), np.inf)
    np.minimum.at(zmin, inv, pts[:, 2])
    zmax = np.full(len(cells), -np.inf)
    np.maximum.at(zmax, inv, pts[:, 2])
    zref = zmax if descend else zmin  # the surface the feet would step onto
    count = np.bincount(inv, minlength=len(cells))
    label = np.full(len(cells), -1, dtype=np.int64)
    best, best_n = -1, 0
    n_lab = 0
    for start in range(len(cells)):
        if label[start] >= 0:
            continue
        q = deque([start])
        label[start] = n_lab
        members = []
        while q:
            c = q.popleft()
            members.append(c)
            ci, cj = divmod(int(cells[c]), w)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = cell_set.get((ci + di) * w + (cj + dj))
                    if nb is not None and label[nb] < 0 and abs(zref[nb] - zref[c]) < 0.25:
                        label[nb] = n_lab
                        q.append(nb)
        members = np.array(members)
        n = int(count[members].sum())
        reach = (zmax[members].max() >= feet_z - REACH_FROM_FEET) if descend \
            else (zmin[members].min() <= feet_z + REACH_FROM_FEET)
        if reach and n > best_n:
            best, best_n = n_lab, n
        n_lab += 1
    if best < 0:
        return np.zeros(len(pts), dtype=bool)
    return label[inv] == best


LAST_REJECT = [""]  # why the last detect_flight() returned None (for the log)


def _reject(why):
    LAST_REJECT[0] = why
    return None


def detect_flight(sem, world_xyz, robot_xy, feet_z, descend=False):
    """One frame -> dict(dir, perp, s_entr, t_centre, half_width, slope, n) or None.

    `dir` is the horizontal direction of travel (uphill, or downhill with descend=True); s/t are
    coordinates along `dir` / `perp` (world origin). The entrance is the flight's bottom edge going up,
    its top edge going down.
    """
    LAST_REJECT[0] = ""
    mask = np.all(sem == STAIRS_COLOR, axis=-1)
    pts = world_xyz[mask]
    if pts.shape[0] < MIN_POINTS:
        return _reject(f"few stair px {pts.shape[0]}")
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    rel = pts[:, :2] - np.asarray(robot_xy, dtype=np.float64)[None, :]
    rel_z = pts[:, 2] - feet_z
    if descend:  # mirror image: from just above the feet down to ABOVE_FEET below them
        zband = (pts[:, 2] < feet_z + BELOW_FEET) & (pts[:, 2] > feet_z - ABOVE_FEET)
    else:
        zband = (pts[:, 2] > feet_z - BELOW_FEET) & (pts[:, 2] < feet_z + ABOVE_FEET)
    keep = (np.hypot(rel[:, 0], rel[:, 1]) < MAX_RANGE) & zband
    pts = pts[keep]
    if pts.shape[0] > 6000:
        pts = pts[:: pts.shape[0] // 6000 + 1]
    if pts.shape[0] < MIN_POINTS:
        return _reject(f"few in range/height band {pts.shape[0]} (z rel feet {np.percentile(rel_z, [5, 50, 95]).round(2).tolist()})")
    pts = pts[_largest_reachable_component(pts, feet_z, descend)].astype(np.float64)
    if pts.shape[0] < MIN_POINTS:
        return _reject(f"no reachable group ({pts.shape[0]} pts)")

    # plane z = a x + b y + c through the flight (treads and risers average out to the incline)
    sel = np.ones(pts.shape[0], dtype=bool)
    for _ in range(3):
        A = np.c_[pts[sel, 0], pts[sel, 1], np.ones(int(sel.sum()))]
        coef, *_ = np.linalg.lstsq(A, pts[sel, 2], rcond=None)
        res = np.abs(pts[:, 0] * coef[0] + pts[:, 1] * coef[1] + coef[2] - pts[:, 2])
        sel = res < max(0.20, 2.0 * np.median(res))
        if sel.sum() < MIN_POINTS:
            return _reject("plane fit lost its points")
    slope = float(np.hypot(coef[0], coef[1]))
    if not (MIN_SLOPE <= slope <= MAX_SLOPE):
        return _reject(f"slope {slope:.2f} out of range")
    d = np.array([coef[0], coef[1]]) / slope  # uphill
    if descend:
        d = -d
    p = np.array([-d[1], d[0]])
    s = pts[sel, :2] @ d
    t = pts[sel, :2] @ p
    z = pts[sel, 2]
    low = (z > z.max() - 0.30) if descend else (z < z.min() + 0.30)  # the entrance edge
    t_lo, t_hi = np.percentile(t, [5, 95])
    return {
        "dir": d, "perp": p,
        "s_entr": float(np.percentile(s[low], 10)),
        "s_top": float(np.percentile(s, 97)),
        "t_centre": float(0.5 * (t_lo + t_hi)),
        "half_width": float(0.5 * (t_hi - t_lo)),
        "z_low": float(z.min()), "z_high": float(z.max()), "slope": slope, "n": int(sel.sum()),
    }


class StairSeeker:
    """Turns per-frame detections into a sub-goal, remembering the flight while it is out of view."""

    def __init__(self):
        self.flight = None
        self.age = 0          # ticks since the remembered flight was last seen
        self.lined_up = False
        self.search_sign = 1.0           # +1 = turn left while searching
        self.found_by_search_sign = 1.0  # the turn direction that last found a flight
        self.level_ticks = 0
        self.level_start_xy = None
        self.done = []                   # (dir, t_centre, z_low, z_high) of flights walked this episode
        self.on_flight = False           # has been on the remembered flight's incline
        self.state = "none"

    def reset(self):
        self.__init__()

    def track_level(self, pitch_up, robot_xy):
        """Call every tick: counts how long the body has been level (IMU pitch) and remembers where
        that started. Returns the count."""
        self.level_ticks = self.level_ticks + 1 if abs(pitch_up) < FLAT_PITCH else 0
        if self.level_ticks == 1:
            self.level_start_xy = np.asarray(robot_xy, dtype=np.float64).copy()
        return self.level_ticks

    def update(self, sem, world_xyz, robot_xy, robot_yaw, feet_z, pitch_up=0.0, front_clear=np.inf,
               descend=False):
        """Returns (sub_goal_xy, state). state: lineup / climb / memory / search.

        pitch_up: body pitch in rad (nose up positive, from the IMU) -- tells "still on the incline"
        from "back on level ground" once the flight has left the view.
        front_clear: free distance straight ahead from the depth image (m).
        """
        robot_xy = np.asarray(robot_xy, dtype=np.float64)
        det = detect_flight(sem, world_xyz, robot_xy, feet_z, descend)
        on_steps = (pitch_up < -FLAT_PITCH) if descend else (pitch_up > FLAT_PITCH)

        # Has the remembered flight been walked to its end? Checked BEFORE taking this frame's detection:
        # checked only while the flight was out of view, a robot that still saw the flight's last steps
        # behind/below it never finished it and walked straight on into the landing wall
        # (user: "它現在直直撞著牆壁", 2026-10-14).
        # - only once the robot has actually been on this flight: while walking to it across a landing
        #   the body is level the whole time, and dropping it there threw away a correct detection
        #   every time it turned to line up;
        # - level for a moment is not enough: the body levels out while the hind feet are still on the
        #   last steps, and turning there jams them;
        # - distance rather than time: at the bottom of a flight going down it walks fast, and a time
        #   limit let it run 2.5 m out across the floor before turning.
        if self.flight is not None and self.on_flight:
            runout = (np.linalg.norm(robot_xy - self.level_start_xy)
                      if self.level_ticks > 0 and self.level_start_xy is not None else 0.0)
            if runout > RUNOUT or (self.level_ticks >= 6 and front_clear < FRONT_STOP):
                self.done.append((self.flight["dir"].copy(), self.flight["t_centre"],
                                  self.flight["z_low"], self.flight["z_high"]))
                self.flight, self.on_flight, self.lined_up = None, False, False

        if det is not None and any(float(det["dir"] @ d) > 0.85 and abs(det["t_centre"] - t) < DONE_MATCH
                                   and det["z_low"] < zh + 0.3 and det["z_high"] > zl - 0.3
                                   for d, t, zl, zh in self.done):
            # (heights matter: in a switchback stairwell every second flight runs the same way right
            # above/below the last one)
            # a flight already walked: on a landing where the flight just come down and the next one
            # down run side by side the same way, stepping onto the old one's bottom steps made it look
            # like a way down again, and it walked back up (user: "它會往回走", 2026-10-14)
            det = None
        if (det is not None and self.flight is not None and self.on_flight and on_steps
                and float(det["dir"] @ self.flight["dir"]) < 0.85):
            # still on the steps of the remembered flight: the side/rear cameras already see the NEXT
            # flight, and switching to it mid-flight turned the robot around on the steps (2026-10-14).
            # Finish this flight first; the next one is found again from the landing.
            det = None
        if det is not None:
            old = self.flight
            if old is not None and float(det["dir"] @ old["dir"]) > 0.85:
                # same flight seen again: smooth the estimate (width / entrance jitter frame to frame)
                d = old["dir"] + SMOOTH * (det["dir"] - old["dir"])
                d /= np.linalg.norm(d)
                det["dir"], det["perp"] = d, np.array([-d[1], d[0]])
                for k in ("s_entr", "t_centre", "half_width"):
                    det[k] = old[k] + SMOOTH * (det[k] - old[k])
                det["z_low"], det["z_high"] = min(det["z_low"], old["z_low"]), max(det["z_high"], old["z_high"])
            else:
                self.lined_up = False  # a new flight
                self.on_flight = False
                if self.state == "search":
                    self.found_by_search_sign = self.search_sign
            self.flight, self.age = det, 0
        elif self.flight is not None:
            self.age += 1  # out of view: keep going along the remembered flight for a while
            if self.age > MEMORY_TICKS:
                self.flight = None

        if self.flight is None:
            self.lined_up = False
            self.on_flight = False
            self.state = "search"
            self.search_sign = self.found_by_search_sign
            # tilted the way the stairs go (nose down going down, up going up) = still on steps: never
            # turn on the spot there, keep going straight. Tilted the other way is something else (it
            # once reared up against a wall and "kept going straight" into it).
            if on_steps and front_clear > FRONT_STOP:
                self.state = "blind"
                return robot_xy + 1.0 * np.array([np.cos(robot_yaw), np.sin(robot_yaw)]), self.state
            a = robot_yaw + self.search_sign * SEARCH_ANGLE
            return robot_xy + 1.0 * np.array([np.cos(a), np.sin(a)]), self.state

        f = self.flight
        s_r, t_r = float(robot_xy @ f["dir"]), float(robot_xy @ f["perp"])
        lateral = abs(t_r - f["t_centre"])
        # on the incline the right way round (nose down going down, up going up). Counting either tilt
        # let it latch "lined up" after stepping onto the neighbouring UP flight on a landing where both
        # run the same way, and it then climbed that one (2026-10-14).
        on_incline = (pitch_up < -2.0 * FLAT_PITCH) if descend else (pitch_up > 2.0 * FLAT_PITCH)
        self.on_flight = self.on_flight or on_incline
        if lateral <= LINEUP_TOL or on_incline:
            self.lined_up = True
        if not self.lined_up:
            self.state = "lineup"
            s_goal = f["s_entr"] - APPROACH_DIST
        else:
            self.state = "climb" if self.age == 0 else "memory"
            s_goal = max(s_r, f["s_entr"]) + AHEAD
        return f["dir"] * s_goal + f["perp"] * f["t_centre"], self.state
