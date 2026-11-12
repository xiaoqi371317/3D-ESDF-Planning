"""Rigid poses and conservative advancement for the sampled ESDF field."""

from dataclasses import dataclass
from math import ceil

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import ESDF, Robot


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
        if not self.is_free(end):
            return False

        # Work in field units. One unit of rigid-body point motion can change
        # the sampled, interpolated field by at most its Lipschitz constant.
        variation = edge.bound * self.field.lipschitz
        fraction = 0.0
        for _ in range(512):
            slack = self.clearance(edge.at(fraction)) - self.margin
            if slack <= 0.0:
                return False
            if slack > (1.0 - fraction) * variation:
                return True
            advance = 0.95 * slack / variation
            if advance < 1e-8:
                return False
            fraction += advance
        return False

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
