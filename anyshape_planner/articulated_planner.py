"""Compact SE(3) guides and RRT-Connect in the full configuration space."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .articulated_collision import ArticulatedCollision, Configuration
from .collision import CollisionChecker, Pose
from .planner import Plan, PlanningError, RRTConnect, connect_trees, shortcut


@dataclass
class ArticulatedPlan:
    path: list[Configuration]
    iterations: int
    method: str


class ArticulatedPlanner:
    def __init__(self, checker: ArticulatedCollision, bounds, portals=(), seed=0):
        self.checker = checker
        self.robot = checker.robot
        self.bounds = bounds
        self.portals = portals
        self.rng = np.random.default_rng(seed)
        self.nominal = self.robot.postures["nominal"]
        self.compact = self.robot.postures["compact"]

    def _project(self, pose):
        height = self.checker.base_height
        return pose if height is None else pose.on_plane(height)

    def _targets(self, start, goal, free_orientation, position_tolerance):
        if self.checker.base_height is not None and not goal.base.is_planar(self.checker.base_height):
            raise PlanningError("目标底盘位姿必须在地面平面内。")
        goals = [goal] if self.checker.is_free(goal) else []
        if goals and not free_orientation and position_tolerance == 0:
            return goals
        for index in range(90):
            if not free_orientation and not position_tolerance:
                break
            position = goal.position
            if position_tolerance and (not free_orientation or index % 2):
                offset = self.rng.normal(size=3)
                position = position + position_tolerance * self.rng.random()**(1 / 3) * offset / np.linalg.norm(offset)
            rotation = goal.rotation
            if free_orientation:
                rotation = start.rotation if index == 0 else Rotation.random(random_state=self.rng)
            candidate = Configuration(self._project(Pose.at(position, rotation)), goal.joints)
            if self.checker.is_free(candidate):
                goals.append(candidate)
                if len(goals) == 5:
                    break
        if not goals:
            raise PlanningError("目标位姿或目标关节姿态发生碰撞。")
        return goals

    def _compact_anchor(self, fixed, base, preferred):
        if preferred is not None and fixed.is_free(preferred):
            return preferred
        if fixed.is_free(base):
            return base
        for _ in range(80):
            position = np.clip(base.position + self.rng.normal(0, .28, 3), *self.bounds)
            rotation = Rotation.from_rotvec(self.rng.normal(0, .32, 3)) * base.rotation
            pose = self._project(Pose.at(position, rotation))
            if fixed.is_free(pose):
                return pose
        raise PlanningError("未找到可用的紧凑姿态入口。")

    def _base_guide(self, start, goal, cancelled):
        if self.checker.self_clearance(self.compact) <= self.checker.self_margin:
            raise PlanningError("紧凑姿态发生自碰撞。")
        fixed = CollisionChecker(self.robot.frozen(self.compact), self.checker.field, self.checker.margin)
        left, right = sorted((start.position[0], goal.position[0]))
        portals = sorted((p for p in self.portals if left < p.position[0] < right),
                         key=lambda p: p.position[0], reverse=start.position[0] > goal.position[0])
        entry = portals[0] if portals and start.position[0] * portals[0].position[0] > 0 else None
        exit = portals[-1] if portals and goal.position[0] * portals[-1].position[0] > 0 else None
        a = self._compact_anchor(fixed, start.base, entry)
        b = self._compact_anchor(fixed, goal.base, exit)
        anchors = [a, *(p for p in portals if p is not a and p is not b and fixed.is_free(p)), b]

        def sample_portal(rng):
            if not portals:
                return Pose.at(rng.uniform(*self.bounds), Rotation.random(random_state=rng))
            center = portals[int(rng.integers(len(portals)))]
            position = np.clip(center.position + rng.normal(0, (.68, .12, .12)), *self.bounds)
            rotation = Rotation.from_rotvec(rng.normal(0, .18, 3)) * center.rotation
            return Pose.at(position, rotation)

        guide = RRTConnect(fixed, self.bounds, int(self.rng.integers(2**32)), sample_portal,
                           base_height=self.checker.base_height)
        path, iterations = [a], 0
        for source, target in zip(anchors, anchors[1:]):
            segment = guide.plan(source, target, max_iterations=520, cancelled=cancelled)
            path.extend(segment.path[1:])
            iterations += segment.iterations
        return [Configuration(base, self.compact) for base in path], iterations

    def _search(self, start, goals, guide, max_iterations, cancelled):
        edges = [self.checker.edge(a, b) for a, b in zip(guide, guide[1:])]

        def sample(choice):
            if guide and choice < .22:
                return guide[int(self.rng.integers(len(guide)))]
            if edges and choice < .82:
                edge = edges[int(self.rng.integers(len(edges)))]
                base = edge.base.at(self.rng.random())
                position = np.clip(base.position + self.rng.normal(0, (.16, .075, .075)), *self.bounds)
                rotation = Rotation.from_rotvec(self.rng.normal(0, .11, 3)) * base.rotation
            else:
                position = self.rng.uniform(*self.bounds)
                rotation = Rotation.random(random_state=self.rng)
            mode = self.rng.random()
            if mode < .25:
                joints = self.rng.uniform(*self.robot.limits.T)
            else:
                center = self.nominal if mode < .50 else self.compact
                joints = self.robot.clamp(center + self.rng.normal(0, .13, len(center)))
            return Configuration(self._project(Pose.at(position, rotation)), joints)

        plan = connect_trees(start, goals, self.robot.radius, self.rng, sample,
                             self.checker.edge, self.checker.edge_is_free, max_iterations, cancelled,
                             step=.55, joint_weights=self.robot.joint_radii, continuous=self.robot.continuous)
        return Plan(shortcut(plan.path, self.checker.edge_is_free, self.rng, cancelled, attempts=32),
                    plan.iterations)

    def plan(self, start, goal, *, free_orientation=False, position_tolerance=0.0,
             max_iterations=1600, force_full=False, cancelled=lambda: False):
        if position_tolerance < 0:
            raise ValueError("Goal position tolerance must be nonnegative.")
        if not self.checker.is_free(start):
            raise PlanningError("起始构型发生碰撞。")
        goals = self._targets(start, goal, free_orientation, position_tolerance)
        if not force_full and not any(min(start.position[0], goals[0].position[0]) < p.position[0]
                                     < max(start.position[0], goals[0].position[0]) for p in self.portals):
            for target in goals:
                if self.checker.edge_is_free(start, target):
                    return ArticulatedPlan([start, target], 0, "直接连接")

        guide, base_iterations = [], 0
        for target in goals:
            if cancelled():
                raise PlanningError("Planning cancelled.")
            try:
                guide, base_iterations = self._base_guide(start, target, cancelled)
                reference = [start, *guide, target]
                if force_full:
                    guide = reference
                    break
                left = self._search(start, [guide[0]], reference[:2], 420, cancelled)
                right = self._search(guide[-1], [target], reference[-2:], 420, cancelled)
                middle = shortcut(guide, self.checker.edge_is_free, self.rng, cancelled, attempts=24)
                path = left.path + middle[1:] + right.path[1:]
                if all(self.checker.edge_is_free(a, b) for a, b in zip(path, path[1:])):
                    return ArticulatedPlan(path, base_iterations + left.iterations + right.iterations,
                                           "紧凑姿态引导")
            except PlanningError:
                if cancelled():
                    raise
        plan = self._search(start, goals, guide, max_iterations, cancelled)
        return ArticulatedPlan(plan.path, base_iterations + plan.iterations, "全维 RRT-Connect")
