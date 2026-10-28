"""Stair map for GUIDANCE=perception: remember every flight seen, pick the next one from the map.

Replaces stair_perception.StairSeeker's frame-by-frame choice (kept in that file for reference). On
landings several flights are in view at once -- with the side/rear cameras even more so -- and
re-deciding from each frame made the robot hop between them, walk back up the flight it had just come
down, or walk on into the wall (2026-10-14). Here each frame only UPDATES the map:

    stairs pixels + depth -> 3D points -> grouped into separate flights (3D voxel connectivity, so
    flights stacked above each other stay apart) -> a plane per group: uphill direction, slope, extent
    -> matched to a known flight (same direction, same line across, overlapping heights) or added

and the robot walks the map:

    pick   the nearest confirmed, not yet walked flight whose entry end (bottom going up, top going
           down) is at the robot's own level
    approach   a point 0.8 m before that entry, on the flight's centreline
    walk   along the flight's centreline until past its far end and level again -> mark it walked
    search     no candidate: turn on the spot (never on the steps); after a full turn, move on 1.5 m

Everything is relative to the robot's own motion, so on a real robot "world" is its odometry frame.
"""
from collections import deque

import numpy as np

STAIRS_COLOR = np.array([0, 153, 0], dtype=np.uint8)  # VIPLANNER_SEM_META "stairs"

MAX_RANGE = 6.0            # m, stair pixels farther than this are ignored (depth gets unreliable)
Z_BAND = 3.0               # m, only points within this height of the feet (one storey up or down)
VOXEL = 0.15               # m, horizontal cell for grouping points into flights
VOXEL_Z = 0.20             # m, vertical cell (one riser is 0.18 m)
MIN_POINTS = 120           # points a group needs to count as a flight observation
MAX_SAMPLES = 12000        # stair points kept per frame (all cameras together)
MIN_SLOPE, MAX_SLOPE = 0.45, 1.5   # rise/run of a real flight (0.18/0.27 = 0.67 here)

MATCH_DIR = 0.9            # cos of the angle between two observations of the same flight
MATCH_ACROSS = 0.7         # m, centreline offset still counted as the same flight
MATCH_Z = 0.3              # m, height ranges must overlap within this (and the planes agree within it)
MERGE_GAP = 0.3            # m, side gap between two pieces still joined into one flight
MIN_INLIER_FRAC = 0.6     # share of a group's points on its plane; less = two flights joined
WRONG_WAY_DROP = 0.3       # m, feet going this far the wrong way while walking a flight = wrong flight
CONFIRM_HITS = 3           # observations before a flight is trusted
SMOOTH = 0.3               # blend factor for a re-observed flight's direction / centreline / width

ENTRY_LEVEL_TOL = 0.6      # m, entry end this close to the feet height = reachable from here
MIN_RISE = 1.0             # m, a flight worth taking climbs at least this far above the feet
# (was 1.5, sized for the Stair.usdc building's 2.3 m flights; the U-shaped scene's flights rise only
# 1.5 m, so none qualified, 2026-10-16. The flight just come up still fails: its top is at the feet.)
# going down, a flight below is seen from above at a grazing angle -- only its first few steps: the
# 2F->landing flight showed 0.9 m and MIN_RISE kept it out, the robot explored 2F and walked off the
# scene (2026-10-15). The pitch check in update() catches a wrong pick.
MIN_DROP = 0.6
APPROACH_DIST = 0.8        # m, line-up point before the entry
APPROACH_TOL = 0.35        # m, close enough to the line-up point / centreline to start walking the flight
LINE_TOL = 0.4             # m, most a robot already at the entry may be off the centreline and still walk
WRONG_PITCH = np.radians(15.0)   # body pitched this far the wrong way = on a flight going the wrong way
WRONG_PITCH_TICKS = 8      # update() calls (~1 s) of that before turning back
WALK_STALL_TICKS = 50     # update() calls (~8 sim steps each) without headway near the entry -> line up again
AHEAD = 1.0                # m, sub-goal distance along the flight
EXIT_MARGIN = 0.3          # m past the far end (along the flight) before it counts as walked
FLAT_PITCH = np.radians(6.0)
SEARCH_ANGLE = np.radians(100.0)
EXPLORE_DIST = 1.5         # m, move-on distance after a full turn without a candidate


def _groups(pts):
    """Split points into 26-connected voxel groups. Returns a list of index arrays."""
    key3 = np.c_[np.floor(pts[:, 0] / VOXEL), np.floor(pts[:, 1] / VOXEL), np.floor(pts[:, 2] / VOXEL_Z)].astype(np.int64)
    cells, inv = np.unique(key3, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    index = {tuple(c): i for i, c in enumerate(cells.tolist())}
    label = np.full(len(cells), -1, dtype=np.int64)
    n_lab = 0
    offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]
    for start in range(len(cells)):
        if label[start] >= 0:
            continue
        label[start] = n_lab
        q = deque([start])
        while q:
            c = q.popleft()
            ci, cj, ck = cells[c]
            for a, b, d in offs:
                nb = index.get((ci + a, cj + b, ck + d))
                if nb is not None and label[nb] < 0:
                    label[nb] = n_lab
                    q.append(nb)
        n_lab += 1
    point_label = label[inv]
    return [np.nonzero(point_label == k)[0] for k in range(n_lab)]


def _fit_flight(pts):
    """Plane through one group -> flight observation dict, or None if it is not stair-like."""
    sel = np.ones(pts.shape[0], dtype=bool)
    for _ in range(3):
        A = np.c_[pts[sel, 0], pts[sel, 1], np.ones(int(sel.sum()))]
        coef, *_ = np.linalg.lstsq(A, pts[sel, 2], rcond=None)
        res = np.abs(pts[:, 0] * coef[0] + pts[:, 1] * coef[1] + coef[2] - pts[:, 2])
        sel = res < max(0.20, 2.0 * np.median(res))
        if sel.sum() < MIN_POINTS:
            return None
    slope = float(np.hypot(coef[0], coef[1]))
    if not (MIN_SLOPE <= slope <= MAX_SLOPE):
        return None
    # two side-by-side flights of a switchback can join into one group; the plane through both is
    # tilted and its extent runs down the other flight (it led the climb down flight 1, 2026-10-14)
    if sel.sum() < MIN_INLIER_FRAC * pts.shape[0]:
        return None
    d = np.array([coef[0], coef[1]]) / slope          # uphill
    p = np.array([-d[1], d[0]])
    q = pts[sel]
    s, t = q[:, :2] @ d, q[:, :2] @ p
    t_lo, t_hi = np.percentile(t, [5, 95])
    return {"dir": d, "t_centre": float(0.5 * (t_lo + t_hi)), "half_width": float(0.5 * (t_hi - t_lo)),
            "s_lo": float(np.percentile(s, 3)), "s_hi": float(np.percentile(s, 97)),
            "z_low": float(np.percentile(q[:, 2], 2)), "z_high": float(np.percentile(q[:, 2], 98)),
            "slope": slope, "z_c": float(coef[2]), "n": int(sel.sum())}   # plane: z = slope*s + z_c


def _plane_z(f, xy):
    return f["slope"] * float(np.asarray(xy) @ f["dir"]) + f["z_c"]


def _mid(f):
    """xy of the middle of a flight."""
    return f["dir"] * 0.5 * (f["s_lo"] + f["s_hi"]) + np.array([-f["dir"][1], f["dir"][0]]) * f["t_centre"]


def detect_flights(sem, world_xyz, robot_xy, feet_z):
    """All stair-like groups in one frame (any number of cameras, arrays of shape (..., 3))."""
    mask = np.all(sem == STAIRS_COLOR, axis=-1)
    pts = world_xyz[mask]
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] < MIN_POINTS:
        return []
    rel = pts[:, :2] - np.asarray(robot_xy, dtype=np.float64)[None, :]
    pts = pts[(np.hypot(rel[:, 0], rel[:, 1]) < MAX_RANGE) & (np.abs(pts[:, 2] - feet_z) < Z_BAND)]
    if pts.shape[0] > MAX_SAMPLES:
        pts = pts[:: pts.shape[0] // MAX_SAMPLES + 1]
    pts = pts.astype(np.float64)
    out = []
    for idx in _groups(pts):
        if idx.size >= MIN_POINTS:
            f = _fit_flight(pts[idx])
            if f is not None:
                out.append(f)
    return out


class StairMap:
    """Flights seen so far (world / odometry frame) and the walk through them."""

    def __init__(self):
        self.flights = []      # dicts: detect fields + id, hits, walked
        self.next_id = 0
        self.target = None     # id of the flight being approached / walked
        self.phase = "none"    # approach / walk
        self.state = "none"
        self.level_ticks = 0
        self.walk_z = 0.0      # best feet height reached on the flight being walked
        self.entry_z = None    # feet height when the target flight was picked (its entry height)
        self.walk_best_s = 0.0 # furthest point reached along the flight being walked
        self.walk_stall = 0    # ticks without headway along it
        self.wrong_ticks = 0   # consecutive ticks pitched the wrong way
        self.search_yaw = 0.0  # accumulated turning while searching
        self.last_yaw = None
        self.explore_goal = None
        self.last_obs = 0

    def reset(self):
        self.__init__()

    # -- map ------------------------------------------------------------------------------------
    def _match(self, ob):
        for f in self.flights:
            if (float(ob["dir"] @ f["dir"]) > MATCH_DIR
                    and abs(ob["t_centre"] - f["t_centre"]) < MATCH_ACROSS
                    and ob["z_low"] < f["z_high"] + MATCH_Z and ob["z_high"] > f["z_low"] - MATCH_Z):
                if abs(_plane_z(ob, _mid(ob)) - _plane_z(f, _mid(ob))) < MATCH_Z:
                    return f
        return None

    def _integrate(self, obs):
        self.last_obs = len(obs)
        for ob in obs:
            f = self._match(ob)
            if f is None:
                ob.update(id=self.next_id, hits=1, walked=False)
                self.next_id += 1
                self.flights.append(ob)
                continue
            # a view narrower than the flight as known so far sees only part of its width and its
            # centre is off to one side: it moved the 4th flight's centreline 0.7 m into the side
            # wall and the line-up point into the landing's corner ("一直想撞牆", 2026-10-15)
            w = SMOOTH * min(1.0, ob["half_width"] / max(f["half_width"], 1e-3)) ** 2
            d = f["dir"] + w * (ob["dir"] - f["dir"])
            f["dir"] = d / np.linalg.norm(d)
            for k in ("t_centre", "half_width", "slope", "z_c"):
                f[k] += w * (ob[k] - f[k])
            # extents only grow: a flight is usually seen piece by piece
            f["s_lo"], f["s_hi"] = min(f["s_lo"], ob["s_lo"]), max(f["s_hi"], ob["s_hi"])
            f["z_low"], f["z_high"] = min(f["z_low"], ob["z_low"]), max(f["z_high"], ob["z_high"])
            f["hits"] += 1
        self._merge()

    def _merge(self):
        """Join flights that are one flight seen in pieces: same direction and plane, side spans
        touching. The 2F->3F flight was split in two 1.5 m apart, one centreline in the wall (2026-10-14)."""
        i = 0
        while i < len(self.flights):
            a = self.flights[i]
            for b in self.flights[i + 1:]:
                # never reshape the flight being walked: a stray piece joined to it moved its far end
                # out of reach and it never counted as walked
                if self.phase == "walk" and self.target in (a["id"], b["id"]):
                    continue
                if (a["walked"] == b["walked"] and float(a["dir"] @ b["dir"]) > MATCH_DIR
                        and abs(a["t_centre"] - b["t_centre"]) < a["half_width"] + b["half_width"] + MERGE_GAP
                        and a["z_low"] < b["z_high"] and a["z_high"] > b["z_low"]
                        and abs(_plane_z(a, _mid(b)) - _plane_z(b, _mid(b))) < MATCH_Z):
                    t_lo = min(a["t_centre"] - a["half_width"], b["t_centre"] - b["half_width"])
                    t_hi = max(a["t_centre"] + a["half_width"], b["t_centre"] + b["half_width"])
                    keep, drop = (a, b) if a["hits"] >= b["hits"] else (b, a)
                    keep["t_centre"], keep["half_width"] = 0.5 * (t_lo + t_hi), 0.5 * (t_hi - t_lo)
                    keep["s_lo"], keep["s_hi"] = min(a["s_lo"], b["s_lo"]), max(a["s_hi"], b["s_hi"])
                    keep["z_low"], keep["z_high"] = min(a["z_low"], b["z_low"]), max(a["z_high"], b["z_high"])
                    keep["hits"] = a["hits"] + b["hits"]
                    if self.target == drop["id"]:
                        self.target = keep["id"]
                    self.flights = [f for f in self.flights if f is not drop]
                    i = -1   # restart: the merged flight may now touch another
                    break
            i += 1

    def _get(self, fid):
        return next((f for f in self.flights if f["id"] == fid), None)

    @staticmethod
    def frame(f, descend, entry_z=None):
        """(travel dir d, across dir p, s of entry, s of exit, t of the centreline), s/t along d/p.

        The map stores everything along the UPHILL direction; going down, d and p both flip sign, so
        s and t do too, and the entry is the high end. With entry_z (feet height where the flight is
        joined) the entry is where the stair plane reaches that height, not the end of the seen points:
        a few stray low points put flight 2's bottom 0.7 m under the landing (2026-10-14)."""
        sign = -1.0 if descend else 1.0
        d = sign * f["dir"]
        p = np.array([-d[1], d[0]])
        s_lo, s_hi = f["s_lo"], f["s_hi"]
        if entry_z is not None:
            s_level = min(max((entry_z - f["z_c"]) / f["slope"], s_lo), s_hi)
            s_hi, s_lo = (s_level, s_lo) if descend else (s_hi, s_level)
        s_entry = -s_hi if descend else s_lo
        s_exit = -f["s_lo"] if descend else f["s_hi"]
        return d, p, s_entry, s_exit, sign * f["t_centre"]

    def _candidates(self, robot_xy, feet_z, descend):
        out = []
        for f in self.flights:
            if f["walked"] or f["hits"] < CONFIRM_HITS:
                continue
            # the flight must pass the feet height and lead well away from it the right way
            if descend:
                ok = f["z_high"] > feet_z - ENTRY_LEVEL_TOL and f["z_low"] < feet_z - MIN_DROP
            else:
                ok = f["z_low"] < feet_z + ENTRY_LEVEL_TOL and f["z_high"] > feet_z + MIN_RISE
            if not ok:
                continue
            d, p, s_entry, _, t = self.frame(f, descend, feet_z)
            entry_xy = d * s_entry + p * t
            out.append((float(np.linalg.norm(entry_xy - robot_xy)), f))
        out.sort(key=lambda x: x[0])
        return [f for _, f in out]

    # -- walking ----------------------------------------------------------------------------------
    def track_level(self, pitch_up, robot_xy=None):
        """Call every tick: how many ticks the body has been level (IMU pitch)."""
        self.level_ticks = self.level_ticks + 1 if abs(pitch_up) < FLAT_PITCH else 0
        return self.level_ticks

    def update(self, sem, world_xyz, robot_xy, robot_yaw, feet_z, pitch_up=0.0, front_clear=np.inf,
               descend=False):
        """Returns (sub_goal_xy, state): approach / walk / search / explore / blind."""
        robot_xy = np.asarray(robot_xy, dtype=np.float64)
        self._integrate(detect_flights(sem, world_xyz, robot_xy, feet_z))
        on_steps = (pitch_up < -FLAT_PITCH) if descend else (pitch_up > FLAT_PITCH)

        # On stairs the wrong way (nose up while going down, or down while going up) -- read from the
        # IMU pitch alone, so a drifting height estimate cannot hide it. Leg odometry ran 1 m high
        # after the first flight down, a stray fragment then looked like the next flight down and its
        # line-up point sent the robot back up the flight it had just come down (2026-10-15). Drop
        # the target for good and head back down (up) until off that slope.
        wrong_slope = (pitch_up > WRONG_PITCH) if descend else (pitch_up < -WRONG_PITCH)
        self.wrong_ticks = self.wrong_ticks + 1 if wrong_slope else 0
        if self.wrong_ticks >= WRONG_PITCH_TICKS:
            if self.target is not None:
                bad = self._get(self.target)
                if bad is not None:
                    bad["walked"] = True
                print(f"[stair_map] on a slope the wrong way: dropped flight #{self.target}, turning back")
                self.target, self.phase = None, "none"
            self.state = "back"
            return robot_xy - np.array([np.cos(robot_yaw), np.sin(robot_yaw)]), self.state

        f = self._get(self.target) if self.target is not None else None
        if f is not None:
            d, p, s_entry, s_exit, t = self.frame(f, descend, self.entry_z)
            s_r, t_r = float(robot_xy @ d), float(robot_xy @ p)
            # walked to the end: past the far end and level again
            if self.phase == "walk" and s_r > s_exit + EXIT_MARGIN and self.level_ticks >= 4:
                f["walked"] = True
                f = None
            # walked into a wall at the end of it (the far end is often only half seen)
            elif self.phase == "walk" and self.level_ticks >= 6 and front_clear < 0.8 and s_r > s_entry + 0.5:
                f["walked"] = True
                f = None
            # height going the wrong way (down while climbing / up while descending): this is not the
            # flight we are on; drop it so it is never picked again
            elif self.phase == "walk":
                self.walk_z = min(self.walk_z, feet_z) if descend else max(self.walk_z, feet_z)
                if (feet_z - self.walk_z) * (-1.0 if descend else 1.0) < -WRONG_WAY_DROP:
                    print(f"[stair_map] flight #{f['id']}: feet went {abs(feet_z - self.walk_z):.2f} m the wrong way, dropped")
                    f["walked"] = True
                    f = None
                else:
                    # no headway near the bottom (a foot caught on the side wall 0.9 m off the
                    # centreline, 2026-10-14): back off and line up again. Never mid-flight.
                    if s_r > self.walk_best_s + 0.1:
                        self.walk_best_s, self.walk_stall = s_r, 0
                    else:
                        self.walk_stall += 1
                    if self.walk_stall > WALK_STALL_TICKS and s_r < s_entry + 1.0:
                        print(f"[stair_map] flight #{f['id']}: no headway at the entry, lining up again")
                        self.phase = "approach"
        if f is None:
            self.target, self.phase = None, "none"
            cands = self._candidates(robot_xy, feet_z, descend)
            if cands:
                f = cands[0]
                self.target, self.phase, self.entry_z = f["id"], "approach", feet_z

        if f is not None:
            self.search_yaw, self.last_yaw, self.explore_goal = 0.0, None, None
            d, p, s_entry, s_exit, t = self.frame(f, descend, self.entry_z)
            s_r, t_r = float(robot_xy @ d), float(robot_xy @ p)
            if self.phase == "approach":
                app = d * (s_entry - APPROACH_DIST) + p * t
                close = np.linalg.norm(robot_xy - app) < APPROACH_TOL
                lined = abs(t_r - t) < APPROACH_TOL and s_entry - APPROACH_DIST - 0.3 < s_r < s_entry + 0.3
                if close or lined or (s_r > s_entry - 0.2 and abs(t_r - t) < min(f["half_width"], LINE_TOL)):
                    self.phase, self.walk_z = "walk", feet_z
                    self.walk_best_s, self.walk_stall = s_r, 0
                else:
                    self.state = "approach"
                    return app, self.state
            self.state = "walk"
            return d * (max(s_r, s_entry) + AHEAD) + p * t, self.state

        # no flight to take from here
        if on_steps and front_clear > 1.1:
            self.state = "blind"   # never turn on the spot on steps
            return robot_xy + np.array([np.cos(robot_yaw), np.sin(robot_yaw)]), self.state
        if self.explore_goal is not None:
            if np.linalg.norm(self.explore_goal - robot_xy) > 0.3 and front_clear > 0.6:
                self.state = "explore"
                return self.explore_goal, self.state
            self.explore_goal = None
        if self.last_yaw is not None:
            self.search_yaw += abs((robot_yaw - self.last_yaw + np.pi) % (2 * np.pi) - np.pi)
        self.last_yaw = robot_yaw
        if self.search_yaw > 2 * np.pi and front_clear > EXPLORE_DIST:
            self.search_yaw, self.last_yaw = 0.0, None
            self.explore_goal = robot_xy + EXPLORE_DIST * np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
            self.state = "explore"
            return self.explore_goal, self.state
        self.state = "search"
        a = robot_yaw + SEARCH_ANGLE
        return robot_xy + np.array([np.cos(a), np.sin(a)]), self.state

    # -- for the log / rviz -----------------------------------------------------------------------
    def summary(self):
        parts = []
        for f in self.flights:
            if f["hits"] < CONFIRM_HITS:
                continue
            mark = "*" if f["id"] == self.target else ("w" if f["walked"] else "")
            parts.append(f"#{f['id']}{mark} up=({f['dir'][0]:+.1f},{f['dir'][1]:+.1f}) "
                         f"across={f['t_centre']:.1f} z={f['z_low']:.1f}-{f['z_high']:.1f} n={f['hits']}")
        return "; ".join(parts) if parts else "empty"

    def outlines(self):
        """[(corners (5,3) closed loop, kind)] kind: target / walked / known / tentative."""
        out = []
        for f in self.flights:
            # only flights worth looking at: one-off detections are mostly Mask2Former noise and drew
            # stray slanted boxes in rviz (user: "路線圖怎麼出現奇怪的線", 2026-10-14)
            if not (f["id"] == self.target or f["walked"] or f["hits"] >= 2 * CONFIRM_HITS):
                continue
            d = f["dir"]
            p = np.array([-d[1], d[0]])
            hw = max(f["half_width"], 0.2)
            c = []
            for s, z in ((f["s_lo"], f["z_low"]), (f["s_hi"], f["z_high"])):
                for sign in (-1, 1):
                    xy = d * s + p * (f["t_centre"] + sign * hw)
                    c.append((xy[0], xy[1], z))
            loop = np.array([c[0], c[1], c[3], c[2], c[0]])
            kind = ("target" if f["id"] == self.target else "walked" if f["walked"]
                    else "known" if f["hits"] >= CONFIRM_HITS else "tentative")
            out.append((loop, kind))
        return out
