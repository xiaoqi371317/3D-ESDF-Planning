"""Time-parameterized SE(3) references and bounded kinematic pose tracking."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .articulated_collision import ArticulatedCollision, Configuration
from .collision import CollisionChecker, Edge, Pose


class TrackingError(RuntimeError):
    pass


@dataclass
class TrackingSample:
    time: float
    pose: Pose | Configuration
    clearance: float
    position_error: float
    angle_error: float
    self_clearance: float = np.inf
    joint_error: float = 0.0


def _limit(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector * (maximum / norm) if norm > maximum else vector


class Follower:
    def __init__(self, path: list[Pose] | list[Configuration], checker: CollisionChecker | ArticulatedCollision,
                 linear_limit: float = 1.0, angular_limit: float = 2.4):
        if len(path) < 2 or min(linear_limit, angular_limit) <= 0:
            raise ValueError("Tracking requires two poses and positive speed limits.")
        self.checker = checker
        self.articulated = isinstance(checker, ArticulatedCollision)
        self.goal = path[-1]
        self.linear_limit = linear_limit
        self.angular_limit = angular_limit
        self.edges = [checker.edge(a, b) if self.articulated else Edge(a, b, checker.robot.radius)
                      for a, b in zip(path, path[1:])]
        self.durations = [self._duration(edge) for edge in self.edges]
        self.pose = path[0]
        self.segment = 0
        self.segment_time = 0.0
        self.time = 0.0
        self.samples = [self._sample(self.pose)]

    def _duration(self, edge):
        rigid = edge.base if self.articulated else edge
        # The smoothstep reference peaks at 1.5 times its average speed.
        duration = max(.02, 1.5 * np.linalg.norm(rigid.delta) / self.linear_limit,
                       1.5 * np.linalg.norm(rigid.rotvec) / self.angular_limit)
        if self.articulated:
            duration = max(duration, 1.5 * (np.abs(edge.joint_delta) / self.checker.robot.speeds).max(initial=0))
        return duration

    def _base(self, pose):
        return pose.base if self.articulated else pose

    def _joint_error(self, reference):
        if not self.articulated:
            return 0.0
        delta = self.checker.robot.delta(self.pose.joints, reference.joints)
        return float(np.abs(delta).max(initial=0))

    def _errors(self, reference):
        return (float(np.linalg.norm(self.pose.position - reference.position)),
                float(np.linalg.norm(self._base(self.pose).relative_rotation(self._base(reference)))))

    def _sample(self, reference):
        position, angle = self._errors(reference)
        env, self_distance = (self.checker.clearances(self.pose) if self.articulated else
                              (self.checker.clearance(self.pose), np.inf))
        return TrackingSample(self.time, self.pose, env, position, angle, self_distance,
                              self._joint_error(reference))

    @property
    def done(self) -> bool:
        return self.segment == len(self.edges)

    @property
    def terminal_error(self) -> tuple[float, float]:
        return self._errors(self.goal)

    @property
    def terminal_joint_error(self) -> float:
        return self._joint_error(self.goal)

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
        omega = _limit(18 * self._base(self.pose).relative_rotation(self._base(reference)), self.angular_limit)
        next_base = Pose.at(self.pose.position + dt * v, Rotation.from_rotvec(dt * omega) * self.pose.rotation)
        next_pose = next_base
        if self.articulated:
            robot = self.checker.robot
            velocity = np.clip(18 * robot.delta(self.pose.joints, reference.joints), -robot.speeds, robot.speeds)
            next_pose = Configuration(next_base, robot.clamp(self.pose.joints + dt * velocity))
        if not self.checker.edge_is_free(self.pose, next_pose):
            raise TrackingError("Tracking stopped before an unsafe motion.")
        self.pose = next_pose
        self.time += dt
        self.segment_time = segment_time
        sample = self._sample(reference)
        self.samples.append(sample)
        # Settle at the corner before tracking the next edge.
        if fraction == 1 and sample.position_error < .0005 and max(sample.angle_error, sample.joint_error) < .001:
            self.segment += 1
            self.segment_time = 0.0
        elif segment_time > duration + 2:
            raise TrackingError("Tracking did not settle at the waypoint.")
        return sample
