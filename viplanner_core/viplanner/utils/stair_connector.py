# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# 2026-09-03: `StairConnector` -- an explicit, graph-bound metadata model for a single stairwell
# connecting two floors, generalizing the previously hardcoded per-mission `Step_WP1..4`/`Landing`
# waypoints in `viplanner_demo.py`'s `TOPOLOGICAL_GRAPHS["multi_floor"]`.
#
# Deliberately NOT a free-floating dict keyed by name (too disconnected from the topology graph it's
# meant to drive) and NOT derived from waypoint-name substring matching like
# `viplanner_demo._is_stair_waypoint` (too error-prone as the sole source of truth -- e.g.
# `second_flight_pose` has no substring-matchable equivalent in the current waypoint naming
# convention at all). Instead, `MultiFloorGoalManager` (multi_floor_goal_manager.py) builds its
# `TopologicalPlanner` graph DIRECTLY from a `dict[str, StairConnector]` via `.to_waypoints()`/
# `.to_edges()` -- the connector metadata IS the source of truth the graph's topology comes from,
# not a separate table consulted after the fact. `TopologicalPlanner` itself needs no changes for
# this: it already only requires a plain waypoints dict + edge list.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from .topological_planner import Point3


@dataclass
class StairConnector:
    """One stairwell connecting `floor_from` to `floor_to`.

    `entrance_pose`/`landing_pose`/`second_flight_pose`/`exit_pose` are the coarse, always-present
    waypoints. `intermediate_waypoints` is an OPTIONAL, explicitly-ordered list of finer subgoals
    spliced between `entrance_pose` and `landing_pose` -- this is what preserves the existing
    `Step_WP1..4` tread-level subdivision (needed because a single entrance->landing hop was
    diagnosed, in the 2026-08-28 investigation, as asking ViPlanner to jump too far in one shot;
    see the historical comment above `TOPOLOGICAL_GRAPHS["multi_floor"]` in `viplanner_demo.py`).
    Only StairA/StairB currently needs this; a future connector with a gentler/shorter first flight
    may not.
    """

    name: str
    floor_from: int
    floor_to: int
    entrance_pose: Point3
    landing_pose: Point3
    second_flight_pose: Point3
    exit_pose: Point3
    direction: Literal["up", "down"] = "up"
    traversable: bool = True
    intermediate_waypoints: List[Tuple[str, Point3]] = field(default_factory=list)
    """Ordered (name, pos) pairs chained between `entrance_pose` and `landing_pose`. Empty by
    default -- a connector with no diagnosed need for tread-level subdivision can omit this."""

    def to_waypoints(self, prefix: Optional[str] = None) -> Dict[str, Point3]:
        """Named-waypoint dict for this connector alone, ready to merge into a
        `TopologicalPlanner(waypoints=..., edges=...)` call alongside every other connector's."""
        p = prefix or self.name
        waypoints: Dict[str, Point3] = {f"{p}_Entrance": self.entrance_pose}
        for wp_name, pos in self.intermediate_waypoints:
            waypoints[f"{p}_{wp_name}"] = pos
        waypoints[f"{p}_Landing"] = self.landing_pose
        waypoints[f"{p}_SecondFlight"] = self.second_flight_pose
        waypoints[f"{p}_Exit"] = self.exit_pose
        return waypoints

    def to_edges(self, prefix: Optional[str] = None) -> List[Tuple[str, str]]:
        """Linear chain Entrance -> [intermediate...] -> Landing -> SecondFlight -> Exit -- matches
        the simple path topology every stair graph in this repo has used so far (no branching
        within a single flight)."""
        p = prefix or self.name
        chain = [f"{p}_Entrance"] + [f"{p}_{wp_name}" for wp_name, _ in self.intermediate_waypoints]
        chain += [f"{p}_Landing", f"{p}_SecondFlight", f"{p}_Exit"]
        return list(zip(chain[:-1], chain[1:]))

    def is_stair_node(self, name: str) -> bool:
        """Whether a waypoint name (as produced by `to_waypoints`) belongs to this connector --
        the graph-bound replacement for `viplanner_demo._is_stair_waypoint`'s substring guess."""
        return name.startswith(f"{self.name}_")


def multi_floor_stair_connectors() -> Dict[str, "StairConnector"]:
    """The one real stair connector this repo currently has (multi_floor's StairA/Landing/StairB),
    hand-populated from `data/multi_floor/ramp_cfg.yml`'s real geometry and the poses previously
    hardcoded in `viplanner_demo.py`'s `TOPOLOGICAL_GRAPHS["multi_floor"]`. Only one entry today --
    the lookup/binding mechanism (`MultiFloorGoalManager`) is what's built to scale to more; adding a
    second stairwell later is "add one more dict entry", not "extend an inference heuristic".
    """
    return {
        "StairA": StairConnector(
            name="StairA",
            floor_from=0,
            floor_to=1,
            # 2026-09-09: recomputed for go2_stairs_env.py's new _build_multifloor_boxes() geometry
            # (ported from the user's reference Isaac Gym script) -- FIRST_START_X moved from -4.0
            # to -4.15, and Stair B's exit now leads into a real walled room whose door sits
            # DOOR_BUFFER=0.8m further back than the bare stairwell mouth, not directly abutting it.
            entrance_pose=(-4.0, -1.2, 1.0),  # base of Stair_A, 1F (FIRST_START_X + STEP_DEPTH/2)
            intermediate_waypoints=[
                ("Step_WP1", (-3.85, -1.2, 0.3)),  # ~StairA_01 tread
                ("Step_WP2", (-2.95, -1.2, 0.75)),  # ~StairA_04 tread
                ("Step_WP3", (-2.05, -1.2, 1.2)),  # ~StairA_06/07 tread
                ("Step_WP4", (-1.45, -1.2, 1.5)),  # ~StairA_09 tread, just before the landing
            ],
            landing_pose=(-0.40, 0.0, 2.0),  # mid-climb waypoint at the U-turn
            # 2026-09-03: no substring-matchable equivalent existed for this in the previous
            # waypoint naming -- new node, splitting what was previously a single direct
            # Landing -> StairB_Exit hop into two, at StairB_00's base (the first/lowest tread of
            # flight B, right after the landing -- StairB_00 is nearest the landing and StairB_09
            # is nearest the 2F exit, i.e. the reverse of flight A's numbering).
            second_flight_pose=(-1.30, 1.2, 1.65),
            # top of Stair_B, just past the room's actual door (not just the bare stairwell mouth
            # -- the new scene has a real DOOR_BUFFER=0.8m foyer + door before the walled 2F room
            # starts), aligned with the door's own center_y (=SECOND_Y=1.2).
            exit_pose=(-5.25, 1.2, 4.0),
            direction="up",
            traversable=True,
        )
    }
