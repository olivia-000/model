"""Top-view plot of VIPLANNER_GHOST episodes: the path the (gait-free) base actually took, every Nth
ViPlanner planned path, the planning_goal the guidance handed ViPlanner, and the waypoints -- to judge
how much the planner itself winds around, with the gait taken out of the picture.

Usage: python plot_ghost_paths.py <ghost_log_dir> [every_nth_plan=5]
Writes <ghost_log_dir>/episode_NNN.png plus a path-length summary.
"""
import glob
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Stair.usdc footprint, for orientation only
WALLS = [((-0.2, 0.0), (-2.0, 5.44)), ((4.4, 4.6), (-2.0, 5.44)), ((-0.2, 4.6), (5.24, 5.44))]
FLIGHTS = {"Flight1/3": ((2.4, 4.4), (0.0, 3.24)), "Flight2/4": ((0.0, 2.0), (0.0, 3.24))}


def main(d, nth):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "episode_*.npz"))):
        z = np.load(f)
        path, wps = z["path"], z["waypoints"]
        if len(path) < 2:
            continue
        xy = path[:, 2:4]
        length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        # shortest reasonable route: straight segments start -> waypoint 1 -> ... (XY only)
        ref = np.vstack([xy[:1], wps[:, :2]])
        straight = float(np.linalg.norm(np.diff(ref, axis=0), axis=1).sum())
        yaw_turn = float(np.abs(np.diff(np.unwrap(path[:, 5]))).sum())
        rows.append((os.path.basename(f), bool(z["reached"]), len(path), length, straight, np.degrees(yaw_turn)))

        fig, ax = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [1.1, 1]})
        a = ax[0]
        for (x0, x1), (y0, y1) in WALLS:
            a.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color="0.6"))
        for name, ((x0, x1), (y0, y1)) in FLIGHTS.items():
            a.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls="--", color="0.5"))
            a.text(x0 + 0.05, y1 - 0.25, name, color="0.4", fontsize=8)
        pp = z["plan_pts"]
        for k in range(0, len(pp), nth):
            a.plot(pp[k][:, 0], pp[k][:, 1], "-", color=plt.cm.autumn(min(float(z["plan_fear"][k]), 1.0)), lw=0.7, alpha=0.6)
        sc = a.scatter(xy[:, 0], xy[:, 1], c=path[:, 4], cmap="viridis", s=2, zorder=3)
        plt.colorbar(sc, ax=a, label="height z [m]", fraction=0.04)
        a.plot(*z["plan_goal"][::nth, :2].T, "x", color="tab:blue", ms=3, alpha=0.5, label="planning_goal (guidance)")
        a.plot(wps[:, 0], wps[:, 1], "*", color="red", ms=14, label="waypoints")
        a.plot(*xy[0], "o", color="black", ms=8, label="start")
        a.set_aspect("equal"); a.set_xlim(-0.6, 5.0); a.set_ylim(-2.8, 5.8); a.legend(loc="lower right", fontsize=8)
        a.set_title(f"{os.path.basename(f)}  reached={bool(z['reached'])}\n"
                    f"path {length:.1f} m vs straight-line {straight:.1f} m, total turning {np.degrees(yaw_turn):.0f} deg\n"
                    f"thin lines: every {nth}th ViPlanner plan (yellow=low fear, red=high fear)", fontsize=9)
        b = ax[1]
        t = path[:, 1] - path[0, 1]
        b.plot(t, path[:, 4], label="height z")
        b.plot(t, path[:, 8] * 2.34, label="waypoint index x 2.34", alpha=0.6)
        b2 = b.twinx(); b2.plot(t, path[:, 7], color="tab:red", lw=0.6, alpha=0.6, label="wz (turn rate)")
        b.set_xlabel("sim time [s]"); b.set_ylabel("m"); b2.set_ylabel("wz [rad/s]")
        b.legend(loc="upper left", fontsize=8); b2.legend(loc="upper right", fontsize=8); b.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(f.replace(".npz", ".png"), dpi=80); plt.close(fig)

    print(f"{'episode':>18} {'reached':>8} {'steps':>7} {'path m':>8} {'straight m':>11} {'ratio':>6} {'turning deg':>12}")
    for n, r, k, L, S, tw in rows:
        print(f"{n:>18} {str(r):>8} {k:>7} {L:8.1f} {S:11.1f} {L / max(S, 1e-6):6.2f} {tw:12.0f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
