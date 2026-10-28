# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# 2026-09-03: `MultiFloorGoalManager` -- lets the robot decide FOR ITSELF whether a goal requires
# finding/climbing stairs, instead of a mission script hardcoding a full waypoint sequence up front
# (`viplanner_demo.py`'s previous `TOPOLOGICAL_GRAPHS["multi_floor"]` construction). Checks whether
# the robot and the goal are on the same floor; same floor routes straight through (today's existing
# single-floor ViPlanner behavior, completely unchanged); different floor selects a `StairConnector`
# (stair_connector.py) and builds a `TopologicalPlanner` subgoal plan through it.

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .stair_connector import StairConnector, multi_floor_stair_connectors
from .topological_planner import Point3, TopologicalPlanner

# Ordered floor-boundary Z heights per scene: floor index = how many boundaries the position's Z is
# at-or-above. Z-bucketing chosen over authoring an explicit floor tag on every waypoint/goal prim --
# it reuses geometry that already exists (scene layout constants), one small constant list per scene,
# instead of a per-node/per-prim authoring burden that has to be kept in sync by hand.
#
# multi_floor's single boundary (1.50m) is literally the scene's own HALF_HEIGHT constant (see
# data/multi_floor/ramp_cfg.yml's header comment) -- the U-turn/landing height exactly halfway
# between 1F (z~0) and 2F (z=3.0), i.e. the natural floor separator for this building already.
FLOOR_BOUNDARIES: Dict[str, Sequence[float]] = {
    "multi_floor": (1.50,),
}


class MultiFloorGoalManager:
    """Wraps one `TopologicalPlanner` (built from `stair_connectors`) plus floor metadata.

    `route()` is the single decision point: same floor -> `None` (direct ViPlanner nav, todays's
    existing single-floor behavior, zero change); different floor -> a named subgoal plan through
    the right `StairConnector`, same shape `TopologicalPlanner.plan_named` already returns so it
    drops straight into the existing `SubgoalSequencer` construction in `viplanner_demo.py`.
    """

    def __init__(
        self,
        floor_boundaries: Sequence[float],
        stair_connectors: Optional[Dict[str, StairConnector]] = None,
    ):
        self.floor_boundaries: List[float] = sorted(floor_boundaries)
        self.stair_connectors: Dict[str, StairConnector] = dict(stair_connectors or {})

        waypoints: Dict[str, Point3] = {}
        edges: List[Tuple[str, str]] = []
        for connector_name, connector in self.stair_connectors.items():
            waypoints.update(connector.to_waypoints(prefix=connector_name))
            edges.extend(connector.to_edges(prefix=connector_name))
        self._connector_graph: Optional[TopologicalPlanner] = TopologicalPlanner(waypoints, edges) if waypoints else None

    @classmethod
    def for_scene(cls, scene_name: str) -> Optional["MultiFloorGoalManager"]:
        """Convenience constructor for a known scene name (matches `args_cli.scene` in
        `viplanner_demo.py`). Returns None for any scene with no registered floor boundaries -- all
        single-floor scenes never need a goal manager at all, exactly like today's
        `TOPOLOGICAL_GRAPHS.get(scene)` returning None for them."""
        boundaries = FLOOR_BOUNDARIES.get(scene_name)
        if boundaries is None:
            return None
        connectors = multi_floor_stair_connectors() if scene_name == "multi_floor" else {}
        return cls(floor_boundaries=boundaries, stair_connectors=connectors)

    def floor_of(self, pos: Sequence[float]) -> int:
        z = pos[2]
        floor = 0
        for boundary in self.floor_boundaries:
            if z >= boundary:
                floor += 1
        return floor

    def is_stair_node(self, name: Optional[str]) -> bool:
        """Graph-bound replacement for `viplanner_demo._is_stair_waypoint`'s substring guess --
        checks real connector membership instead of pattern-matching the name."""
        if name is None:
            return False
        return any(connector.is_stair_node(name) for connector in self.stair_connectors.values())

    def route(self, robot_pos: Sequence[float], goal_pos: Sequence[float]) -> Optional[List[Tuple[str, Point3]]]:
        """Same floor -> None (direct nav). Different floor -> named subgoal plan through the
        selected StairConnector, same shape as `TopologicalPlanner.plan_named`.

        Note: with today's single-connector case, "selection" is trivial (there's exactly one).
        Once a second connector exists, `_connector_graph`'s own Dijkstra/A* search (over ALL
        connectors' merged waypoints/edges) already picks whichever registered route is shortest --
        no separate connector-selection step needs to be added later.
        """
        robot_floor = self.floor_of(robot_pos)
        goal_floor = self.floor_of(goal_pos)
        if robot_floor == goal_floor:
            return None

        if self._connector_graph is None:
            raise ValueError(
                f"goal is on a different floor (robot floor={robot_floor}, goal floor={goal_floor}) "
                "but no stair connectors are registered for this scene"
            )
        return self._connector_graph.plan_named(start_pos=robot_pos, goal_pos=goal_pos)
