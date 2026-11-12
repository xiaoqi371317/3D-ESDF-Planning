"""Bidirectional RRT-Connect over translations and unit quaternions."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .collision import CollisionChecker, Edge, Pose


class PlanningError(RuntimeError):
    pass


@dataclass
class Plan:
    path: list[Pose]
    iterations: int


class Tree:
    def __init__(self, roots: list[Pose]):
        self.poses = list(roots)
        self.parents = [-1] * len(roots)
        self.keys = np.empty((max(128, len(roots)), 7))
        for index, pose in enumerate(roots):
            self.keys[index] = self.key(pose)

    @staticmethod
    def key(pose: Pose):
        return np.r_[pose.position, pose.rotation.as_quat()]

    def add(self, pose: Pose, parent: int) -> int:
        index = len(self.poses)
        if index == len(self.keys):
            self.keys = np.vstack((self.keys, np.empty_like(self.keys)))
        self.keys[index] = self.key(pose)
        self.poses.append(pose)
        self.parents.append(parent)
        return index

    def nearest(self, target: Pose, radius: float) -> int:
        keys = self.keys[:len(self.poses)]
        translation = np.linalg.norm(keys[:, :3] - target.position, axis=1)
        dots = np.sum(keys[:, 3:] * target.rotation.as_quat(), axis=1)
        angle = 2 * np.arccos(np.clip(np.abs(dots), 0, 1))
        return int(np.argmin(translation + radius * angle))

    def branch(self, index: int) -> list[Pose]:
        poses = []
        while index >= 0:
            poses.append(self.poses[index])
            index = self.parents[index]
        return poses[::-1]


class RRTConnect:
    def __init__(self, checker: CollisionChecker, bounds, seed: int = 0,
                 guide: Callable[[np.random.Generator], Pose] | None = None):
        self.checker = checker
        self.bounds = bounds
        self.rng = np.random.default_rng(seed)
        self.guide = guide
        self.radius = checker.robot.radius

    def _goal_roots(self, start: Pose, goal: Pose, free_orientation: bool,
                    tolerance: float) -> list[Pose]:
        roots = []
        if self.checker.is_free(goal):
            roots.append(goal)
        if not free_orientation and roots and tolerance == 0:
            return roots

        for attempt in range(96):
            position = goal.position
            if tolerance > 0 and (not free_orientation or attempt % 2):
                offset = self.rng.normal(size=3)
                position = position + tolerance * self.rng.random()**(1 / 3) * (
                    offset / np.linalg.norm(offset))
            rotation = goal.rotation
            if free_orientation:
                rotation = start.rotation if attempt == 0 else Rotation.random(random_state=self.rng)
            pose = Pose.at(position, rotation)
            if self.checker.is_free(pose):
                roots.append(pose)
                if len(roots) == 6:
                    break
        if not roots:
            raise PlanningError("No collision-free pose was found in the goal region.")
        return roots

    def _extend(self, tree: Tree, target: Pose, step: float, parent: int):
        start = tree.poses[parent]
        edge = Edge(start, target, self.radius)
        if edge.bound == 0.0:
            return parent, True
        reached = edge.bound <= step
        next_pose = target if reached else edge.at(step / edge.bound)
        if not self.checker.edge_is_free(start, next_pose):
            return None, False
        return tree.add(next_pose, parent), reached

    def _shortcut(self, path: list[Pose], cancelled: Callable[[], bool]) -> list[Pose]:
        for _ in range(90):
            if cancelled():
                raise PlanningError("Planning cancelled.")
            if len(path) < 3:
                break
            i, j = sorted(self.rng.choice(len(path), 2, replace=False))
            if j - i > 1 and self.checker.edge_is_free(path[i], path[j]):
                path = path[:i + 1] + path[j:]
        return path

    def plan(self, start: Pose, goal: Pose, *, free_orientation: bool = False,
             position_tolerance: float = 0.0, max_iterations: int = 2600,
             cancelled: Callable[[], bool] = lambda: False) -> Plan:
        if position_tolerance < 0:
            raise ValueError("Goal position tolerance must be nonnegative.")
        if not self.checker.is_free(start):
            raise PlanningError("The start pose is in collision.")
        roots = self._goal_roots(start, goal, free_orientation, position_tolerance)
        for root in roots:
            if self.checker.edge_is_free(start, root):
                return Plan([start, root], 0)

        front, back = Tree([start]), Tree(roots)
        front_is_start = True
        step = .40
        for iteration in range(1, max_iterations + 1):
            if cancelled():
                raise PlanningError("Planning cancelled.")
            choice = self.rng.random()
            if choice < .08:
                root_count = len(roots) if front_is_start else 1
                target = back.poses[int(self.rng.integers(root_count))]
            elif choice < .40 and self.guide is not None:
                target = self.guide(self.rng)
            else:
                target = Pose.at(self.rng.uniform(*self.bounds), Rotation.random(random_state=self.rng))
            index, _ = self._extend(front, target, step, front.nearest(target, self.radius))
            if index is not None:
                meeting = front.poses[index]
                other = back.nearest(meeting, self.radius)
                while not cancelled():
                    other, reached = self._extend(back, meeting, step, other)
                    if other is None:
                        break
                    if reached:
                        start_side, goal_side = front.branch(index), back.branch(other)
                        if not front_is_start:
                            start_side, goal_side = goal_side, start_side
                        path = start_side + goal_side[-2::-1]
                        return Plan(self._shortcut(path, cancelled), iteration)
            front, back = back, front
            front_is_start = not front_is_start
        raise PlanningError(f"No path after {max_iterations} iterations.")
