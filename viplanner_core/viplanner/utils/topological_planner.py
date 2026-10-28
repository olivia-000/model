# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Global, topology-aware planner sitting ABOVE ViPlanner in the navigation stack.

ViPlanner is a purely local, vision-based planner: given a single 3D goal, it reactively steers
toward whatever direction its depth/semantic view suggests is closest to that goal, with no map,
no memory, and no concept of rooms, corridors, floors, or which staircase leads where. Handed a
goal on the other side of the building (or a different floor), it has no way to discover "go
through the corridor, then up this staircase" -- it just tries to walk toward the goal's direction
directly, which usually isn't a navigable path, and the goal is often also far enough away to sit
outside ViPlanner's own trained max_goal_distance range, so its behavior there isn't well-defined.

TopologicalPlanner fixes this with a plain waypoint graph: named nodes (a room, a corridor
midpoint, a stair entrance/landing/exit, ...) each with a 3D position, and edges wherever two
nodes are directly, locally reachable by ViPlanner (i.e. the same kind of leg ViPlanner already
handles fine on a single floor). Dijkstra (or A*, using straight-line distance as the heuristic)
finds the shortest node-to-node path from wherever the robot currently is to wherever the final
goal is, and that path's waypoints become the SUBGOAL sequence ViPlanner executes one at a time.
This generalizes cleanly to any number of floors/rooms/parallel staircases -- e.g. two different
stairwells connecting the same two floors are just two different edges, and the shortest-path
search picks whichever route is actually shorter, without any special-casing for "floors" at all.

This is a plain topology decomposition (a graph search over discrete waypoints), not a metric path
planner: it does not know or care about the exact walking path between two adjacent waypoints --
that's still entirely ViPlanner's job, the same way it always was for a single-floor goal.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx

Point3 = Tuple[float, float, float]


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


class TopologicalPlanner:
    """Decomposes a goal into an ordered list of subgoals via shortest-path search over a named
    waypoint graph.

    Args:
        waypoints: node name -> 3D position (world frame). Can be anything meaningful for the
            building: a room center, a corridor midpoint, a stair entrance/landing/exit, etc.
        edges: pairs of waypoint names that are directly reachable by ViPlanner (a single-floor,
            no-topology-decisions-needed leg). The graph is undirected -- edges are traversable
            both ways. Edge weight is the straight-line distance between the two waypoints,
            computed automatically (not authored by hand), used both for Dijkstra's shortest-path
            search and as A*'s admissible heuristic.
    """

    def __init__(self, waypoints: Dict[str, Point3], edges: Sequence[Tuple[str, str]]):
        assert len(waypoints) > 0, "need at least one waypoint"
        self.waypoints = dict(waypoints)

        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.waypoints.keys())
        for a, b in edges:
            assert a in self.waypoints, f"edge references unknown waypoint '{a}'"
            assert b in self.waypoints, f"edge references unknown waypoint '{b}'"
            self.graph.add_edge(a, b, weight=_distance(self.waypoints[a], self.waypoints[b]))

    def nearest_node(self, position: Sequence[float]) -> str:
        """The waypoint closest (straight-line) to an arbitrary 3D position."""
        return min(self.waypoints, key=lambda name: _distance(self.waypoints[name], position))

    def _node_path(
        self,
        start_pos: Sequence[float],
        goal_pos: Sequence[float],
        start_node: Optional[str],
        goal_node: Optional[str],
        algorithm: str,
    ) -> List[str]:
        start_node = start_node or self.nearest_node(start_pos)
        goal_node = goal_node or self.nearest_node(goal_pos)

        if algorithm == "astar":
            return nx.astar_path(
                self.graph, start_node, goal_node, heuristic=lambda a, b: _distance(self.waypoints[a], self.waypoints[b]), weight="weight"
            )
        try:
            return nx.shortest_path(self.graph, start_node, goal_node, weight="weight")
        except nx.NetworkXNoPath as exc:
            raise ValueError(f"no path from waypoint '{start_node}' to '{goal_node}'") from exc

    def plan(
        self,
        start_pos: Sequence[float],
        goal_pos: Sequence[float],
        start_node: Optional[str] = None,
        goal_node: Optional[str] = None,
        algorithm: str = "dijkstra",
    ) -> List[Point3]:
        """Returns an ordered list of subgoals ending with goal_pos itself.

        start_node/goal_node let you pin the search to a specific waypoint instead of inferring it
        from start_pos/goal_pos by nearest distance (useful when the nearest-by-distance node isn't
        actually the intended one, e.g. two waypoints close together on different floors).

        algorithm: "dijkstra" (plain shortest path by total edge weight) or "astar" (same result,
        guided by straight-line-distance heuristic -- matters for very large graphs, immaterial for
        small building-sized ones).
        """
        node_path = self._node_path(start_pos, goal_pos, start_node, goal_node, algorithm)

        # skip the start node itself (the robot is already there); everything after it becomes a
        # subgoal, then the actual final goal_pos (which may differ from the goal node's own
        # marked position -- e.g. a specific spot inside "Room_B_2F", not that node's exact point)
        subgoals: List[Point3] = [self.waypoints[n] for n in node_path[1:]]
        goal_pos_t = tuple(goal_pos)
        if not subgoals or subgoals[-1] != goal_pos_t:
            subgoals.append(goal_pos_t)
        return subgoals

    def plan_named(
        self,
        start_pos: Sequence[float],
        goal_pos: Sequence[float],
        start_node: Optional[str] = None,
        goal_node: Optional[str] = None,
        algorithm: str = "dijkstra",
    ) -> List[Tuple[str, Point3]]:
        """Same search as `plan`, but pairs each subgoal with the waypoint name it came from --
        e.g. lets a driving script tell "currently routed through a stair waypoint" from the
        active subgoal's name, for the reactive safety layer's stair-aware mode (see
        PathFollowerCommandGeneratorCfg.stair_center_frac). The final appended goal_pos (when it
        differs from the last node's own position) is paired with that last node's name too, since
        it's still "in" that node's region."""
        node_path = self._node_path(start_pos, goal_pos, start_node, goal_node, algorithm)

        named_subgoals: List[Tuple[str, Point3]] = [(n, self.waypoints[n]) for n in node_path[1:]]
        goal_pos_t = tuple(goal_pos)
        if not named_subgoals or named_subgoals[-1][1] != goal_pos_t:
            named_subgoals.append((node_path[-1], goal_pos_t))
        return named_subgoals


class SubgoalSequencer:
    """Runtime helper: walks through a precomputed subgoal list one at a time, advancing once the
    robot gets within `reach_threshold` meters of the current subgoal.

    XY-plane distance only (not full 3D) -- Z isn't a reliable "have I arrived" signal while
    mid-climb on a staircase, where the robot's actual height changes continuously and doesn't
    match any single subgoal's Z until it's essentially already there.
    """

    def __init__(self, subgoals: Sequence[Point3], reach_threshold: float = 0.75, names: Optional[Sequence[str]] = None):
        assert len(subgoals) > 0, "need at least one subgoal (the final goal, at minimum)"
        self.subgoals: List[Point3] = list(subgoals)
        self.reach_threshold = reach_threshold
        self.idx = 0
        if names is not None:
            assert len(names) == len(subgoals), "names must be the same length as subgoals"
        self.names: Optional[List[str]] = list(names) if names is not None else None
        """Optional per-subgoal waypoint name (from TopologicalPlanner.plan_named), for callers
        that need to know WHICH named waypoint the current subgoal came from -- e.g. the reactive
        safety layer's stair-aware mode. None if this sequencer was built from plain positions."""

    @property
    def current(self) -> Point3:
        return self.subgoals[self.idx]

    @property
    def current_name(self) -> Optional[str]:
        """The current subgoal's waypoint name, or None if this sequencer has no names."""
        return self.names[self.idx] if self.names is not None else None

    @property
    def is_final(self) -> bool:
        return self.idx == len(self.subgoals) - 1

    def update(self, robot_xy: Sequence[float]) -> bool:
        """Call every step with the robot's current (x, y) position (any Z is ignored). Returns
        True the one step it advances to a new subgoal, False otherwise (including once
        is_final -- there's nothing left to advance to)."""
        if self.is_final:
            return False
        tx, ty, _ = self.current
        dist = math.hypot(robot_xy[0] - tx, robot_xy[1] - ty)
        if dist < self.reach_threshold:
            self.idx += 1
            return True
        return False


# EoF
