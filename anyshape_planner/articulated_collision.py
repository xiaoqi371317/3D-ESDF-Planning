"""Environment and self collision for a floating-base configuration."""

from dataclasses import dataclass
from math import ceil

import numpy as np

from .articulated import ArticulatedRobot
from .collision import Edge, Pose, certify_interval, free_radius
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
                 self_margin=.001, extra_allowed=(), base_height=None, accelerate=False):
        self.robot = robot
        self.field = field
        self.margin = margin
        self.self_margin = self_margin
        self.base_height = base_height
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
        self._fast_field = None
        self._collision_fk = robot.fk
        if accelerate:
            self._prepare_minimum_queries()
        active = [joint for joint in robot.joints if joint.dof is not None]
        self.relative_rates = np.zeros((len(self.pairs), len(active)))
        for pair_index, (a, b) in enumerate(self.pairs):
            lipschitz = max(self.local_fields[a].lipschitz, self.local_fields[b].lipschitz)
            for joint in active:
                if robot.ancestors[a, joint.dof] != robot.ancestors[b, joint.dof]:
                    reach = 1.0 if joint.prismatic else max(robot.reach[joint.child, a], robot.reach[joint.child, b])
                    self.relative_rates[pair_index, joint.dof] = lipschitz * reach
        dynamic = np.any(self.relative_rates, axis=1)
        self.dynamic_pairs = [pair for pair, moving in zip(self.pairs, dynamic) if moving]
        self.dynamic_rates = self.relative_rates[dynamic]
        self._dynamic_indices = np.asarray(self.pairs, dtype=np.intp).reshape(-1, 2)[dynamic].T
        reference = robot.fk(robot.limits.mean(axis=1))
        self.fixed_self_clearance = min((self._pair_distance(a, b, *reference)
                                         for (a, b), moving in zip(self.pairs, dynamic) if not moving),
                                        default=np.inf)
        if self._fast_field is not None:
            self.self_clearance(robot.limits.mean(axis=1))

    def _prepare_minimum_queries(self):
        try:
            from ._collision_kinematics import CollisionKinematics, pair_geometry
            from ._esdf_minimum import ESDFMinimum, SampleGroups
        except ImportError:
            return
        clouds = [link.geometry.points for link in self.robot.links]
        self._collision_fk = CollisionKinematics(self.robot).forward
        self._pair_geometry_batch = pair_geometry
        self._groups = SampleGroups
        self._fast_field = ESDFMinimum(self.field)
        self._world_samples = SampleGroups(clouds)
        self._local_samples = [SampleGroups([points]) for points in clouds]
        self._fast_local = [ESDFMinimum(field) for field in self.local_fields]
        # Load or compile the kernel while preparing the map, before planning.
        self._fast_field.distance(self._local_samples[0], np.eye(3)[None], np.zeros((1, 3)))

    def prepare_environment_samples(self, points):
        return self._groups([points]) if self._fast_field is not None else points

    def environment_clearance(self, samples, rotation, position):
        if self._fast_field is not None:
            return self._fast_field.distance(samples, rotation[None], position[None])
        return self.field.query(samples @ rotation.T + position).min()

    def edge(self, start, end):
        return ArticulatedEdge(start, end, self.robot)

    def world_points(self, state, fk=None):
        rotations, translations = self.robot.fk(state.joints) if fk is None else fk
        base = state.rotation.as_matrix()
        return [link.geometry.points @ rotation.T + position for link, rotation, position in
                zip(self.robot.links, base @ rotations, state.rotation.apply(translations) + state.position)]

    def link_clearances(self, state, fk=None):
        return np.minimum.reduceat(self.field.query(np.vstack(self.world_points(state, fk))), self.offsets)

    def _pair_geometry(self, a, b, rotations, translations):
        inverse = rotations[a].swapaxes(-1, -2)
        rotation = inverse @ rotations[b]
        offset = (inverse @ (translations[b] - translations[a])[..., None])[..., 0]
        centers, half = self.robot.centers, self.robot.half_extents
        # Project link B's bounding box onto the local axes of link A.
        center = (rotation @ centers[b][..., None])[..., 0] + offset - centers[a]
        extent = (np.abs(rotation) @ half[b][..., None])[..., 0]
        gap = (np.abs(center) - half[a] - extent).max(axis=-1)
        return rotation, offset, gap

    def _local_pair_distance(self, a, b, rotation, offset):
        if self._fast_field is not None:
            inverse = rotation.T.copy()
            return min(self._fast_local[a].distance(self._local_samples[b], rotation[None], offset[None], world=False),
                       self._fast_local[b].distance(self._local_samples[a], inverse[None],
                                                    (-inverse @ offset)[None], world=False))
        local_b = self.robot.links[b].geometry.points @ rotation.T + offset
        local_a = (self.robot.links[a].geometry.points - offset) @ rotation
        return min(self.local_fields[a].interpolate(local_b).min(),
                   self.local_fields[b].interpolate(local_a).min())

    def _pair_distance(self, a, b, rotations, translations):
        rotation, offset, gap = self._pair_geometry(a, b, rotations, translations)
        if gap >= self.self_horizon:
            return self.self_horizon
        distance = self._local_pair_distance(a, b, rotation, offset)
        return min(self.self_horizon, max(gap, distance))

    def self_clearance(self, q, fk=None):
        if not self.dynamic_pairs:
            return self.fixed_self_clearance
        rotations, translations = self._collision_fk(q) if fk is None else fk
        a, b = self._dynamic_indices
        if self._fast_field is None:
            relative, offsets, gaps = self._pair_geometry(a, b, rotations, translations)
        else:
            relative, offsets, gaps = self._pair_geometry_batch(self._dynamic_indices, rotations, translations,
                                                                self.robot.centers, self.robot.half_extents)
        nearest = min(self.fixed_self_clearance, self.self_horizon)
        # The pair distance is at least its box gap below the horizon.
        # Once a gap reaches the best distance, the remaining pairs cannot lower it.
        for index in np.argsort(gaps):
            gap = gaps[index]
            if gap >= nearest:
                break
            distance = self._local_pair_distance(a[index], b[index], relative[index], offsets[index])
            nearest = min(nearest, max(gap, distance))
        return nearest

    def clearances(self, state):
        fk = self._collision_fk(state.joints)
        return float(self._environment_minimum(state, fk)), float(self.self_clearance(state.joints, fk))

    def _environment_minimum(self, state, fk):
        if self._fast_field is None:
            return self.link_clearances(state, fk).min()
        rotations, translations = fk
        base = state.rotation.as_matrix()
        return self._fast_field.distance(self._world_samples, base @ rotations,
                                         translations @ base.T + state.position)

    def free_interval_radius(self, state, env_rate, pair_rates):
        """Combine the environment bound with the motion bound for each pair."""
        if self._fast_field is None:
            environment, self_distance = self.clearances(state)
            return min(free_radius(environment - self.margin, env_rate),
                       free_radius(self_distance - self.self_margin, pair_rates.max(initial=0)))
        fk = self._collision_fk(state.joints)
        radius = free_radius(self._environment_minimum(state, fk) - self.margin, env_rate)
        return self.self_interval_radius(fk, pair_rates, radius)

    def self_interval_radius(self, fk, pair_rates, radius):
        """Limit an environment certificate by the allowed self clearance."""
        if radius == 0 or self.fixed_self_clearance <= self.self_margin:
            return 0.0
        if not self.dynamic_pairs:
            return radius
        a, b = self._dynamic_indices
        if self._fast_field is None:
            rotations, offsets, gaps = self._pair_geometry(a, b, *fk)
        else:
            rotations, offsets, gaps = self._pair_geometry_batch(self._dynamic_indices, *fk,
                                                                 self.robot.centers, self.robot.half_extents)
        slack = np.minimum(gaps, self.self_horizon) - self.self_margin
        # Prune any pair whose box gap already certifies the current radius.
        lower = np.divide(slack, pair_rates, out=np.full_like(slack, np.inf), where=pair_rates > 0)
        lower[slack <= 0] = 0.0
        for index in np.argsort(lower):
            if lower[index] >= radius:
                break
            if gaps[index] >= self.self_horizon:
                radius = lower[index]
            else:
                distance = self._local_pair_distance(a[index], b[index], rotations[index], offsets[index])
                distance = min(self.self_horizon, max(gaps[index], distance))
                radius = min(radius, free_radius(distance - self.self_margin, pair_rates[index]))
        return radius

    def valid(self, state):
        return (self.robot.valid(state.joints)
                and (self.base_height is None or state.base.is_planar(self.base_height)))

    def is_free(self, state):
        if not self.valid(state):
            return False
        environment, self_distance = self.clearances(state)
        return environment > self.margin and self_distance > self.self_margin

    def edge_is_free(self, start, end):
        if not self.valid(start) or not self.valid(end):
            return False
        edge = self.edge(start, end)
        env_rate = edge.bound * self.field.lipschitz
        pair_rates = self.dynamic_rates @ np.abs(edge.joint_delta)

        def radius_at(fraction):
            state = start if fraction == 0 else end if fraction == 1 else edge.at(fraction)
            return self.free_interval_radius(state, env_rate, pair_rates)

        return certify_interval(radius_at)

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
