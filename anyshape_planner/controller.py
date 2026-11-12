"""Time-parameterized SE(3) references and bounded kinematic pose tracking."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .collision import CollisionChecker, Edge, Pose


class TrackingError(RuntimeError):
    pass


@dataclass
class TrackingSample:
    time: float
    pose: Pose
    clearance: float
    position_error: float
    angle_error: float


def _limit(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector * (maximum / norm) if norm > maximum else vector


class Follower:
    def __init__(self, path: list[Pose], checker: CollisionChecker,
                 linear_limit: float = 1.0, angular_limit: float = 2.4):
        if len(path) < 2 or min(linear_limit, angular_limit) <= 0:
            raise ValueError("Tracking requires two poses and positive speed limits.")
        self.checker = checker
        self.goal = path[-1]
        self.edges = [Edge(a, b, checker.robot.radius) for a, b in zip(path, path[1:])]
        # The smoothstep reference peaks at 1.5 times its average speed.
        self.durations = [max(.02, 1.5 * np.linalg.norm(edge.delta) / linear_limit,
                              1.5 * np.linalg.norm(edge.rotvec) / angular_limit)
                          for edge in self.edges]
        self.linear_limit = linear_limit
        self.angular_limit = angular_limit
        self.pose = path[0]
        self.segment = 0
        self.segment_time = 0.0
        self.time = 0.0
        self.samples = [TrackingSample(0.0, self.pose, checker.clearance(self.pose), 0.0, 0.0)]

    @property
    def done(self) -> bool:
        return self.segment == len(self.edges)

    @property
    def terminal_error(self) -> tuple[float, float]:
        return (float(np.linalg.norm(self.pose.position - self.goal.position)),
                float(np.linalg.norm(self.pose.relative_rotation(self.goal))))

    def step(self, dt: float = .02) -> TrackingSample:
        if self.done:
            return self.samples[-1]
        if not 0 < dt <= .05:
            raise ValueError("Use a tracking timestep in (0, 0.05] seconds.")
        duration = self.durations[self.segment]
        segment_time = self.segment_time + dt
        fraction = min(segment_time / duration, 1.0)
        reference = self.edges[self.segment].at(fraction**2 * (3 - 2 * fraction))
        v = _limit(18 * (reference.position - self.pose.position), self.linear_limit)
        omega = _limit(18 * self.pose.relative_rotation(reference), self.angular_limit)
        next_pose = Pose.at(self.pose.position + dt * v,
                            Rotation.from_rotvec(dt * omega) * self.pose.rotation)
        if not self.checker.edge_is_free(self.pose, next_pose):
            raise TrackingError("Tracking stopped before an unsafe motion.")
        self.pose = next_pose
        self.time += dt
        self.segment_time = segment_time
        sample = TrackingSample(self.time, next_pose, self.checker.clearance(next_pose),
                                float(np.linalg.norm(reference.position - next_pose.position)),
                                float(np.linalg.norm(next_pose.relative_rotation(reference))))
        self.samples.append(sample)
        # Settle at the corner before tracking the next edge.
        if fraction == 1 and sample.position_error < .0005 and sample.angle_error < .001:
            self.segment += 1
            self.segment_time = 0.0
        elif segment_time > duration + 2:
            raise TrackingError("Tracking did not settle at the waypoint.")
        return sample
