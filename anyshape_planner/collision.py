"""Rigid poses and continuous collision checks for the sampled ESDF field."""

from dataclasses import dataclass
from math import ceil

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import ESDF, Robot


def free_radius(slack: float, rate: float) -> float:
    """Convert a clearance margin into a certified interval radius."""
    if not slack > 0:
        return 0.0
    return slack / rate if rate else np.inf


def certify_interval(radius_at, start=0.0, end=1.0, *, max_samples=512) -> bool:
    """Cover an interval with collision-free neighborhoods of sampled states."""
    right_radius = radius_at(end)
    if not right_radius > 0:
        return False
    left_radius = radius_at(start)
    if not left_radius > 0:
        return False

    samples = 2
    pending = [(start, end, left_radius, right_radius)]
    while pending:
        left, right, left_radius, right_radius = pending.pop()
        # Strict overlap keeps the whole interval strictly above the margin.
        if left_radius + right_radius > right - left:
            continue
        middle = left + (right - left) / 2
        if samples >= max_samples or not left < middle < right:
            return False
        radius = radius_at(middle)
        samples += 1
        if not radius > 0:
            return False
        pending.append((middle, right, radius, right_radius))
        pending.append((left, middle, left_radius, radius))
    return True


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    rotation: Rotation

    @classmethod
    def at(cls, position, rotation: Rotation | None = None):
        return cls(np.asarray(position, dtype=float),
                   Rotation.identity() if rotation is None else rotation)

    def relative_rotation(self, target: "Pose") -> np.ndarray:
        return (target.rotation * self.rotation.inv()).as_rotvec()

    def on_plane(self, height: float) -> "Pose":
        matrix = self.rotation.as_matrix()
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
        return Pose.at((*self.position[:2], height), Rotation.from_euler("z", yaw))

    def is_planar(self, height: float) -> bool:
        return (abs(self.position[2] - height) <= 1e-9
                and np.allclose(self.rotation.as_matrix()[:, 2], (0, 0, 1), atol=1e-9, rtol=0))


class Edge:
    def __init__(self, start: Pose, end: Pose, radius: float):
        self.start = start
        self.delta = end.position - start.position
        self.rotvec = start.relative_rotation(end)
        self.bound = float(np.linalg.norm(self.delta) + radius * np.linalg.norm(self.rotvec))

    def at(self, fraction: float) -> Pose:
        return Pose(self.start.position + fraction * self.delta,
                    Rotation.from_rotvec(fraction * self.rotvec) * self.start.rotation)


@dataclass
class CollisionChecker:
    robot: Robot
    field: ESDF
    margin: float = 0.005

    def clearance(self, pose: Pose) -> float:
        points = pose.rotation.apply(self.robot.points) + pose.position
        return float(self.field.query(points).min())

    def is_free(self, pose: Pose) -> bool:
        return self.clearance(pose) > self.margin

    def edge_is_free(self, start: Pose, end: Pose) -> bool:
        edge = Edge(start, end, self.robot.radius)
        variation = edge.bound * self.field.lipschitz

        def radius_at(fraction):
            pose = start if fraction == 0 else end if fraction == 1 else edge.at(fraction)
            return free_radius(self.clearance(pose) - self.margin, variation)

        return certify_interval(radius_at)

    def dense_clearance(self, path: list[Pose], step: float = 0.005) -> float:
        if step <= 0:
            raise ValueError("Dense sampling step must be positive.")
        minimum = self.clearance(path[0])
        for start, end in zip(path, path[1:]):
            edge = Edge(start, end, self.robot.radius)
            count = max(1, ceil(edge.bound / step))
            minimum = min(minimum, *(self.clearance(edge.at(k / count))
                                     for k in range(1, count + 1)))
        return minimum
