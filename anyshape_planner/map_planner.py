"""Planar base and joint search guided by the obstacle ESDF alone."""

from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from .articulated_collision import Configuration
from .articulated_planner import ArticulatedPlan, ArticulatedPlanner
from .collision import Pose
from .planner import PlanningError


@dataclass
class MapPlan(ArticulatedPlan):
    guide: np.ndarray | None = None
    search: dict | None = None


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _yaw(pose):
    return pose.rotation.as_rotvec()[2]


def _grid_path(clearance, start, goal, radius, cancelled):
    """A* on the chassis-height slice, with a cost for passing near walls."""
    valid = clearance > radius
    if not valid[start] or not valid[goal]:
        raise PlanningError("地图搜索的起终点没有足够的底盘净空。")
    penalty = 1 + .35 / np.maximum(.06, clearance - radius + .06)
    neighbors = [(x, y, np.hypot(x, y)) for x in (-1, 0, 1) for y in (-1, 0, 1) if x or y]
    costs = np.full(clearance.shape, np.inf)
    costs[start] = 0.0
    queue = [(0.0, 0.0, start)]
    parent = {}
    while queue:
        if cancelled():
            raise PlanningError("Planning cancelled.")
        _, distance, node = heappop(queue)
        if distance != costs[node]:
            continue
        if node == goal:
            path = [node]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return np.asarray(path[::-1])
        for dx, dy, length in neighbors:
            x, y = node[0] + dx, node[1] + dy
            if not (0 <= x < valid.shape[0] and 0 <= y < valid.shape[1] and valid[x, y]):
                continue
            if dx and dy and not (valid[node[0], y] and valid[x, node[1]]):
                continue
            new = distance + length * (penalty[node] + penalty[x, y]) / 2
            if new < costs[x, y]:
                costs[x, y] = new
                parent[x, y] = node
                heappush(queue, (new + np.hypot(goal[0] - x, goal[1] - y), new, (x, y)))
    raise PlanningError("障碍物地图中没有连通的底盘引导路径。")


class MapPlanner(ArticulatedPlanner):
    """Search configuration layers along an automatically discovered guide."""

    def __init__(self, checker, seed=0):
        if checker.base_height is None:
            raise ValueError("MapPlanner requires a planar base.")
        super().__init__(checker, np.stack((checker.field.origin, checker.field.high)), seed=seed)
        self.height = checker.base_height
        self.chassis_z = self.height + checker.robot.centers[0, 2]
        self.radius_xy = float(checker.robot.half_extents[0, :2].min())
        compact = self.robot.frozen(self.compact).mesh
        self.guide_radius = max(self.radius_xy, float(compact.extents[:2].min() / 2))
        self.guide_reach = np.linalg.norm(compact.vertices[:, :2], axis=1).max()
        self.prior = []
        delta = self.robot.delta(self.nominal, self.compact)
        self.fold = np.where(self.robot.prismatic, 0., delta)
        self.arm_reach = np.linalg.norm(self.robot.frozen(self.nominal).mesh.vertices[:, :2], axis=1).max()
        for fraction in (0., .60, .83, 1.):
            q = self.robot.clamp(self.nominal + fraction * delta)
            if checker.self_clearance(q) > checker.self_margin:
                self.prior.append((q, self.robot.frozen(q).points))

    def _clearance_2d(self, points):
        height = np.full((len(points), 1), self.chassis_z)
        wall = self.checker.field.interpolate(np.hstack((points, height)))
        boundary = np.minimum(points - self.bounds[0, :2], self.bounds[1, :2] - points).min(axis=1)
        return np.minimum(wall, boundary)

    def _guide(self, start, goal, cancelled):
        pitch = max(.075, 3 * self.checker.field.resolution)
        origin = self.bounds[0, :2]
        shape = np.floor((self.bounds[1, :2] - origin) / pitch).astype(int) + 1
        grid = np.indices(shape).reshape(2, -1).T
        clearance = self._clearance_2d(origin + pitch * grid).reshape(tuple(shape))
        endpoints = np.rint((np.array([start.position[:2], goal.position[:2]]) - origin) / pitch).astype(int)
        endpoints = np.clip(endpoints, 0, shape - 1)
        try:
            cells = _grid_path(clearance, tuple(endpoints[0]), tuple(endpoints[1]),
                               self.guide_radius + self.checker.margin, cancelled)
        except PlanningError:
            if cancelled() or self.guide_radius <= self.radius_xy:
                raise
            cells = _grid_path(clearance, tuple(endpoints[0]), tuple(endpoints[1]),
                               self.radius_xy + self.checker.margin, cancelled)
        points = origin + pitch * cells
        points[0], points[-1] = start.position[:2], goal.position[:2]
        if len(points) < 2:
            return np.array([start.position[:2], goal.position[:2]])
        smooth = gaussian_filter1d(points, sigma=2.6, axis=0, mode="nearest")
        allowed = self._clearance_2d(smooth) > self.radius_xy + .015
        points[1:-1] = np.where(allowed[1:-1, None], smooth[1:-1], points[1:-1])
        points = self._align_approaches(points)
        length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        step = min(.36, length[-1] / 2)
        if step <= 1e-9:
            return points[[0, -1]]
        stations = np.linspace(0, length[-1], int(np.ceil(length[-1] / step)) + 1)
        return np.column_stack([np.interp(stations, length, coordinates) for coordinates in points.T])

    def _align_approaches(self, points):
        # Infer the local axes of narrow runs from the clearance along the
        # discovered route, then straighten the approach over one body reach.
        narrow = self._clearance_2d(points) < self.guide_radius + .145
        changes = np.flatnonzero(np.diff(np.r_[False, narrow, False]))
        lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        result, strength = points.copy(), np.zeros(len(points))
        for begin, end in changes.reshape(-1, 2):
            throat = points[begin:end]
            if len(throat) < 4 or lengths[end - 1] - lengths[begin] < .24:
                continue
            center = throat.mean(axis=0)
            _, extent, axes = np.linalg.svd(throat - center, full_matrices=False)
            if extent[1] > .24 * extent[0]:
                continue
            axis = axes[0]
            along = (points - center) @ axis
            projection = center + along[:, None] * axis
            distance = np.maximum(lengths[begin] - lengths, lengths - lengths[end - 1])
            weight = np.clip((self.guide_reach + .40 - distance) / .48, 0., 1.)
            proposal = points + weight[:, None] * (projection - points)
            allowed = (weight > strength) & (self._clearance_2d(proposal) > self.radius_xy + .022)
            allowed[[0, -1]] = False
            result[allowed], strength[allowed] = proposal[allowed], weight[allowed]
        return result

    def _refine(self, pose, q, coordinates, cancelled):
        result = q.copy()
        for index in coordinates:
            change = self.robot.delta(result, self.nominal)[index]
            if abs(change) < 1e-5:
                continue
            initial = result.copy()
            lo, hi = 0.0, 1.0
            for _ in range(6):
                if cancelled():
                    raise PlanningError("Planning cancelled.")
                fraction = (lo + hi) / 2
                trial = initial.copy()
                trial[index] += fraction * change
                trial = self.robot.clamp(trial)
                env, self_distance = self.checker.clearances(Configuration(pose, trial))
                if env > self.checker.margin + .019 and self_distance > self.checker.self_margin + .007:
                    lo, result = fraction, trial
                else:
                    hi = fraction
        return result

    def _layer(self, center, direction, broad, cancelled):
        theta = np.arctan2(direction[1], direction[0])
        normal = np.array((-np.sin(theta), np.cos(theta)))
        roomy = self._clearance_2d(center[None])[0] > self.radius_xy + .22
        offsets = ((0., -.10, .10, -.19, .19, -.28, .28) if broad else
                   (0., -.12, .12, -.20, .20) if roomy else (0.,))
        headings = [*np.linspace(-np.pi, np.pi, 16 if broad else 8, endpoint=False), theta, theta + np.pi]
        candidates, refinable = [], []
        for offset in offsets:
            p = np.r_[center + offset * normal, self.height]
            tried = []
            for yaw in headings:
                if any(abs(_wrap(yaw - old)) < .045 for old in tried):
                    continue
                tried.append(yaw)
                if cancelled():
                    raise PlanningError("Planning cancelled.")
                pose = Pose.at(p, Rotation.from_euler("z", yaw))
                options = []
                for q, points in self.prior:
                    distance = self.checker.field.query(pose.rotation.apply(points) + p).min()
                    if distance > self.checker.margin + .008:
                        options.append((q, distance))
                        candidates.append(Configuration(pose, q))
                if options:
                    # Rank refinements near the discovered route but retain the
                    # alternative yaw and joint combinations in the graph.
                    q, distance = options[-1]
                    refinable.append((.18 * abs(_wrap(yaw - theta)) + abs(offset) - .7 * distance, pose, q))
        lift = np.flatnonzero(self.robot.prismatic)
        for _, pose, q in sorted(refinable, key=lambda item: item[0])[:(6 if broad else 3)]:
            raised = self._refine(pose, q, lift, cancelled)
            unfolded = self._refine(pose, raised, np.flatnonzero(~self.robot.prismatic), cancelled)
            if np.linalg.norm(raised - q) > 1e-4:
                candidates.append(Configuration(pose, raised))
            if min(np.linalg.norm(unfolded - q), np.linalg.norm(unfolded - raised)) > 1e-4:
                candidates.append(Configuration(pose, unfolded))
        return candidates

    def _edge_costs(self, source, target):
        left = np.array([state.position[:2] for state in source])
        right = np.array([state.position[:2] for state in target])
        translation = right[None] - left[:, None]
        travel = np.linalg.norm(translation, axis=-1)
        ya = np.array([_yaw(state) for state in source])[:, None]
        yb = np.array([_yaw(state) for state in target])[None]
        yaw = np.abs(_wrap(yb - ya))
        heading = np.arctan2(translation[..., 1], translation[..., 0])
        alignment = np.sin(ya - heading)**2 + np.sin(yb - heading)**2
        qa, qb = (np.array([state.joints for state in layer]) for layer in (source, target))
        delta = np.abs(self.robot.delta(qa[:, None], qb[None]))
        move = delta @ self.robot.joint_radii
        weights = self.robot.joint_radii + 2.6 * self.robot.prismatic
        # A pose close to a wall corner may still have links inside an opening.
        # Bias its rotary joints toward a smaller footprint before and after it.
        prefer_a, prefer_b = (self.nominal + np.clip((self.arm_reach - self._clearance_2d(p)) / .35,
                                                   0., 1.)[:, None] * self.fold for p in (left, right))
        effort_a = np.abs(self.robot.delta(prefer_a, qa)) @ weights
        effort_b = np.abs(self.robot.delta(prefer_b, qb)) @ weights
        effort = (effort_a[:, None] + effort_b[None]) / 2
        cost = travel * (1 + .17 * alignment + .95 * effort) + .34 * yaw + (
            .25 * yaw**2 + .24 * move**2) / (travel + .15) + .24 * move
        return np.where(yaw <= np.pi / 2 + .05, cost, np.inf)

    @staticmethod
    def _shortest(costs):
        distance, parents = np.zeros(1), []
        for matrix in costs:
            incoming = distance[:, None] + matrix
            indices = incoming.argmin(axis=0)
            distance = incoming[indices, np.arange(len(indices))]
            if not np.isfinite(distance).any():
                return None
            parents.append(indices)
        path = [int(distance.argmin())]
        for previous in reversed(parents):
            path.append(int(previous[path[-1]]))
        return path[::-1]

    def _lattice(self, layers, cancelled, attempts=70):
        costs = [self._edge_costs(a, b) for a, b in zip(layers, layers[1:])]
        checked, rejected = set(), 0
        for _ in range(attempts):
            indices = self._shortest(costs)
            if indices is None:
                break
            for layer, (i, j) in enumerate(zip(indices, indices[1:])):
                if cancelled():
                    raise PlanningError("Planning cancelled.")
                key = layer, i, j
                if key in checked:
                    continue
                checked.add(key)
                a, b = layers[layer][i], layers[layer + 1][j]
                edge = self.checker.edge(a, b)
                if not (all(self.checker.is_free(edge.at(t)) for t in (.5, .2, .8)) and
                        self.checker.edge_is_free(a, b)):
                    costs[layer][i, j] = np.inf
                    rejected += 1
                    break
            else:
                return [layer[index] for layer, index in zip(layers, indices)], len(checked), rejected
        return None, len(checked), rejected

    def _adjust_posture(self, path, cancelled, throats_only=False):
        path = list(path)
        lift = np.flatnonzero(self.robot.prismatic)
        rotary = np.flatnonzero(~self.robot.prismatic)[::-1]
        narrow = self._clearance_2d(np.array([state.position[:2] for state in path])) < self.guide_radius + .145
        for index in range(1, len(path) - 1):
            if throats_only and not narrow[index]:
                continue
            before, state, after = path[index - 1:index + 2]
            q = state.joints if throats_only else self._refine(state.base, state.joints, lift, cancelled)
            unfolded = self._refine(state.base, q, rotary, cancelled)
            for posture in (unfolded, q):
                if np.linalg.norm(posture - state.joints) <= 1e-4:
                    continue
                candidate = Configuration(state.base, posture)
                if self.checker.edge_is_free(before, candidate) and self.checker.edge_is_free(candidate, after):
                    path[index] = candidate
                    break
        return path

    def _simplify(self, path, cancelled):
        result = [path[0]]
        index = 0
        while index < len(path) - 1:
            if cancelled():
                raise PlanningError("Planning cancelled.")
            for end in range(min(index + 10, len(path) - 1), index + 1, -1):
                edge = self.checker.edge(path[index], path[end])
                intermediate = path[index + 1:end]
                lengths = np.r_[0, np.cumsum([self.checker.edge(a, b).bound
                                              for a, b in zip(path[index:end], path[index + 1:end + 1])])]
                if lengths[-1] <= 1e-10:
                    continue
                approximates = True
                for state, fraction in zip(intermediate, lengths[1:-1] / lengths[-1]):
                    error = self.checker.edge(edge.at(fraction), state)
                    if np.linalg.norm(error.base.delta) > .075 or np.linalg.norm(error.base.rotvec) > .15 or (
                        np.abs(error.joint_delta) @ self.robot.joint_radii > .055
                    ):
                        approximates = False
                        break
                if approximates and self.checker.edge_is_free(path[index], path[end]):
                    break
            else:
                end = index + 1
            result.append(path[end])
            index = end
        return result

    def _remove_joint_reversals(self, path, cancelled):
        result = list(path)
        for index in range(1, len(result) - 1):
            if cancelled():
                raise PlanningError("Planning cancelled.")
            before, state, after = result[index - 1:index + 2]
            a, b = (self.robot.delta(state.joints, neighbor.joints) for neighbor in (before, after))
            offset = np.clip(np.zeros_like(a), np.minimum(a, b), np.maximum(a, b))
            if np.abs(offset) @ self.robot.joint_radii < .012:
                continue
            for scale in (1., .5):
                q = self.robot.clamp(state.joints + scale * offset)
                candidate = Configuration(state.base, q)
                if self.checker.edge_is_free(before, candidate) and self.checker.edge_is_free(candidate, after):
                    result[index] = candidate
                    break
        return result

    def plan(self, start, goal, *, free_orientation=False, position_tolerance=0., cancelled=lambda: False):
        if position_tolerance < 0:
            raise ValueError("Goal position tolerance must be nonnegative.")
        if isinstance(start, Pose):
            start = Configuration(start, self.nominal)
        if isinstance(goal, Pose):
            candidate = next((q for q, _ in self.prior if self.checker.is_free(Configuration(goal, q))),
                             self.nominal)
            goal = Configuration(goal, candidate)
        if not self.checker.is_free(start):
            raise PlanningError("起始构型发生碰撞。")
        targets = self._targets(start, goal, free_orientation, position_tolerance)
        guide_found = False
        guide_error = None
        for target in targets:
            if self.checker.edge_is_free(start, target):
                return MapPlan([start, target], 0, "地图直接连接")
            try:
                guide = self._guide(start, target, cancelled)
                guide_found = True
            except PlanningError as error:
                if cancelled():
                    raise
                guide_error = error
                continue
            for broad in (False, True):
                layers = [[start], *(self._layer(center, after - before, broad, cancelled)
                                     for before, center, after in zip(guide, guide[1:], guide[2:])), [target]]
                if not all(layers):
                    continue
                path, checked, rejected = self._lattice(layers, cancelled)
                repaired = False
                if path is None:
                    compact = [layers[0], *([state for state in layer if np.allclose(state.joints, self.compact)]
                                            for layer in layers[1:-1]), layers[-1]]
                    if all(compact):
                        path, checks, failures = self._lattice(compact, cancelled, attempts=65)
                        checked, rejected = checked + checks, rejected + failures
                        if path is not None:
                            path = self._adjust_posture(path, cancelled)
                            repaired = True
                if path is not None:
                    shaped = self._adjust_posture(path, cancelled, throats_only=True) if not repaired else path
                    simplified = self._simplify(shaped, cancelled)
                    smooth_joints = self._remove_joint_reversals(simplified, cancelled)
                    return MapPlan(smooth_joints, checked, "地图引导 · 底盘与关节联合搜索", guide=guide,
                                   search={"layers": len(layers), "states": sum(map(len, layers)),
                                           "checked_edges": checked, "rejected_edges": rejected,
                                           "compact_seed_repair": repaired})
        if not guide_found and guide_error is not None:
            raise guide_error
        raise PlanningError("地图引导已找到，联合搜索未找到可安全通过的构型序列。")
