"""Environment and self collision for a floating-base configuration."""

from dataclasses import dataclass
from math import ceil

import numpy as np

from .articulated import ArticulatedRobot
from .collision import Edge, Pose
from .geometry import ESDF


@dataclass(frozen=True)
class Configuration:
    base: Pose
    joints: np.ndarray

    @property
    def position(self):
        return self.base.position

    @property
    def rotation(self):
        return self.base.rotation


class ArticulatedEdge:
    def __init__(self, start, end, robot: ArticulatedRobot):
        self.start = start
        self.base = Edge(start.base, end.base, robot.radius)
        self.joint_delta = robot.delta(start.joints, end.joints)
        self.bound = self.base.bound + float(robot.joint_radii @ np.abs(self.joint_delta))

    def at(self, fraction):
        return Configuration(self.base.at(fraction), self.start.joints + fraction * self.joint_delta)


class ArticulatedCollision:
    def __init__(self, robot: ArticulatedRobot, field: ESDF, margin=.004, local_resolution=.008,
                 self_margin=.001, extra_allowed=()):
        self.robot = robot
        self.field = field
        self.margin = margin
        self.self_margin = self_margin
        self.offsets = np.cumsum([0] + [len(link.geometry.points) for link in robot.links[:-1]])
        names = {link.name: index for index, link in enumerate(robot.links)}
        self.allowed = robot.adjacent | {tuple(sorted((names[a], names[b]))) for a, b in extra_allowed}
        self.pairs = [(a, b) for a in range(len(names)) for b in range(a + 1, len(names))
                      if (a, b) not in self.allowed]
        self.self_horizon = .07
        self.local_fields = []
        for link in robot.links:
            # Align box faces halfway between cell centers where possible.
            bounds = link.geometry.mesh.bounds + np.array([[-.104], [.104]])
            self.local_fields.append(ESDF.build(bounds, local_resolution, [link.obstacle]))
        active = [joint for joint in robot.joints if joint.dof is not None]
        self.relative_rates = np.zeros((len(self.pairs), len(active)))
        for pair_index, (a, b) in enumerate(self.pairs):
            lipschitz = max(self.local_fields[a].lipschitz, self.local_fields[b].lipschitz)
            for joint in active:
                if robot.ancestors[a, joint.dof] != robot.ancestors[b, joint.dof]:
                    self.relative_rates[pair_index, joint.dof] = lipschitz * max(
                        robot.reach[joint.child, a], robot.reach[joint.child, b])

    def edge(self, start, end):
        return ArticulatedEdge(start, end, self.robot)

    def world_points(self, state, fk=None):
        rotations, translations = self.robot.fk(state.joints) if fk is None else fk
        base = state.rotation.as_matrix()
        return [link.geometry.points @ rotation.T + position for link, rotation, position in
                zip(self.robot.links, base @ rotations, state.rotation.apply(translations) + state.position)]

    def link_clearances(self, state, fk=None):
        return np.minimum.reduceat(self.field.query(np.vstack(self.world_points(state, fk))), self.offsets)

    def _pair_distance(self, a, b, rotations, translations):
        rotation = rotations[a].T @ rotations[b]
        offset = rotations[a].T @ (translations[b] - translations[a])
        centers, half = self.robot.centers, self.robot.half_extents
        # Signed interval gap along the local axes of link A. A positive gap
        # guarantees separation even if a sampled local field is pessimistic.
        gap = float((np.abs(rotation @ centers[b] + offset - centers[a])
                     - half[a] - np.abs(rotation) @ half[b]).max())
        if gap >= self.self_horizon:
            return self.self_horizon
        local_b = self.robot.links[b].geometry.points @ rotation.T + offset
        local_a = (self.robot.links[a].geometry.points - offset) @ rotation
        distance = min(self.local_fields[a].interpolate(local_b).min(),
                       self.local_fields[b].interpolate(local_a).min())
        return min(self.self_horizon, max(gap, distance))

    def self_clearance(self, q, fk=None):
        rotations, translations = self.robot.fk(q) if fk is None else fk
        return min((self._pair_distance(a, b, rotations, translations) for a, b in self.pairs),
                   default=np.inf)

    def clearances(self, state):
        fk = self.robot.fk(state.joints)
        return float(self.link_clearances(state, fk).min()), float(self.self_clearance(state.joints, fk))

    def is_free(self, state):
        if not self.robot.valid(state.joints):
            return False
        environment, self_distance = self.clearances(state)
        return environment > self.margin and self_distance > self.self_margin

    def edge_is_free(self, start, end):
        if not self.robot.valid(start.joints) or not self.is_free(end):
            return False
        edge = self.edge(start, end)
        env_rate = edge.bound * self.field.lipschitz
        pair_rates = self.relative_rates @ np.abs(edge.joint_delta)
        self_rate = float(pair_rates.max(initial=0))
        fraction = 0.0
        for _ in range(512):
            env, self_distance = self.clearances(edge.at(fraction))
            env_slack, self_slack = env - self.margin, self_distance - self.self_margin
            if min(env_slack, self_slack) <= 0:
                return False
            remaining = 1 - fraction
            if env_slack > remaining * env_rate and self_slack > remaining * self_rate:
                return True
            advance = min(env_slack / env_rate if env_rate else np.inf,
                          self_slack / self_rate if self_rate else np.inf)
            if advance < 1e-8:
                return False
            fraction += .95 * advance
        return False

    def dense_states(self, path, step=.005):
        if step <= 0:
            raise ValueError("Dense sampling step must be positive.")
        yield path[0]
        for start, end in zip(path, path[1:]):
            edge = self.edge(start, end)
            count = max(1, ceil(edge.bound / step))
            for index in range(1, count + 1):
                yield edge.at(index / count)

    def dense_clearances(self, path, step=.005):
        distances = np.array([self.clearances(state) for state in self.dense_states(path, step)])
        return tuple(distances.min(axis=0))
