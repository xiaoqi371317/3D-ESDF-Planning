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
    def __init__(self, roots, joint_weights=None, continuous=None):
        self.poses = list(roots)
        self.parents = [-1] * len(roots)
        self.joint_weights = joint_weights
        self.continuous = continuous
        count = 0 if joint_weights is None else len(joint_weights)
        self.keys = np.empty((max(128, len(roots)), 7 + count))
        for index, pose in enumerate(roots):
            self.keys[index] = self.key(pose)

    def key(self, pose):
        rigid = np.r_[pose.position, pose.rotation.as_quat()]
        return rigid if self.joint_weights is None else np.r_[rigid, pose.joints]

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
        dots = np.sum(keys[:, 3:7] * target.rotation.as_quat(), axis=1)
        angle = 2 * np.arccos(np.clip(np.abs(dots), 0, 1))
        distance = translation + radius * angle
        if self.joint_weights is not None:
            delta = keys[:, 7:] - target.joints
            delta[:, self.continuous] = (delta[:, self.continuous] + np.pi) % (2 * np.pi) - np.pi
            distance += np.abs(delta) @ self.joint_weights
        return int(np.argmin(distance))

    def branch(self, index: int) -> list[Pose]:
        poses = []
        while index >= 0:
            poses.append(self.poses[index])
            index = self.parents[index]
        return poses[::-1]


def shortcut(path, edge_is_free, rng, cancelled, attempts=90):
    checked = set()
    for _ in range(attempts):
        if cancelled():
            raise PlanningError("Planning cancelled.")
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i <= 1 or (id(path[i]), id(path[j])) in checked:
            continue
        checked.add((id(path[i]), id(path[j])))
        if edge_is_free(path[i], path[j]):
            path = path[:i + 1] + path[j:]
    return path


def connect_trees(start, roots, radius, rng, sample, make_edge, edge_is_free,
                  max_iterations, cancelled, step=.40, joint_weights=None, continuous=None):
    def extend(tree, target, parent):
        source = tree.poses[parent]
        edge = make_edge(source, target)
        if edge.bound == 0.0:
            return parent, True
        reached = edge.bound <= step
        next_pose = target if reached else edge.at(step / edge.bound)
        if not edge_is_free(source, next_pose):
            return None, False
        return tree.add(next_pose, parent), reached

    for root in roots:
        if cancelled():
            raise PlanningError("Planning cancelled.")
        if edge_is_free(start, root):
            return Plan([start, root], 0)

    front = Tree([start], joint_weights, continuous)
    back = Tree(roots, joint_weights, continuous)
    front_is_start = True
    for iteration in range(1, max_iterations + 1):
        if cancelled():
            raise PlanningError("Planning cancelled.")
        choice = rng.random()
        if choice < .08:
            root_count = len(roots) if front_is_start else 1
            target = back.poses[int(rng.integers(root_count))]
        else:
            target = sample(choice)
        index, _ = extend(front, target, front.nearest(target, radius))
        if index is not None:
            meeting = front.poses[index]
            other = back.nearest(meeting, radius)
            while not cancelled():
                other, reached = extend(back, meeting, other)
                if other is None:
                    break
                if reached:
                    start_side, goal_side = front.branch(index), back.branch(other)
                    if not front_is_start:
                        start_side, goal_side = goal_side, start_side
                    path = start_side + goal_side[-2::-1]
                    return Plan(path, iteration)
        front, back = back, front
        front_is_start = not front_is_start
    raise PlanningError(f"No path after {max_iterations} iterations.")


class RRTConnect:
    def __init__(self, checker: CollisionChecker, bounds, seed: int = 0,
                 guide: Callable[[np.random.Generator], Pose] | None = None, *, base_height=None):
        self.checker = checker
        self.bounds = bounds
        self.rng = np.random.default_rng(seed)
        self.guide = guide
        self.radius = checker.robot.radius
        self.base_height = base_height

    def _project(self, pose):
        return pose if self.base_height is None else pose.on_plane(self.base_height)

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
            pose = self._project(Pose.at(position, rotation))
            if self.checker.is_free(pose):
                roots.append(pose)
                if len(roots) == 6:
                    break
        if not roots:
            raise PlanningError("No collision-free pose was found in the goal region.")
        return roots

    def plan(self, start: Pose, goal: Pose, *, free_orientation: bool = False,
             position_tolerance: float = 0.0, max_iterations: int = 2600,
             cancelled: Callable[[], bool] = lambda: False) -> Plan:
        if position_tolerance < 0:
            raise ValueError("Goal position tolerance must be nonnegative.")
        if self.base_height is not None and not all(pose.is_planar(self.base_height) for pose in (start, goal)):
            raise PlanningError("Start and goal must be on the base plane.")
        if not self.checker.is_free(start):
            raise PlanningError("The start pose is in collision.")
        roots = self._goal_roots(start, goal, free_orientation, position_tolerance)

        def sample(choice):
            if choice < .40 and self.guide is not None:
                return self._project(self.guide(self.rng))
            return self._project(Pose.at(self.rng.uniform(*self.bounds), Rotation.random(random_state=self.rng)))

        plan = connect_trees(start, roots, self.radius, self.rng, sample,
                             lambda a, b: Edge(a, b, self.radius), self.checker.edge_is_free,
                             max_iterations, cancelled)
        return Plan(shortcut(plan.path, self.checker.edge_is_free, self.rng, cancelled), plan.iterations)
