"""Clearance of upright link bounding boxes in a box obstacle map."""

import numpy as np
from numba import njit

from .collision import certify_interval


@njit(cache=True, inline="always")
def _point_rectangle_squared(x, y, hx, hy):
    dx, dy = max(abs(x) - hx, 0.), max(abs(y) - hy, 0.)
    return dx * dx + dy * dy


@njit(cache=True, nogil=True)
def _link_clearances(centers, axes, half, obstacles, bounds):
    result = np.empty(len(centers))
    for link, (x, y, z) in enumerate(centers):
        ux, uy = axes[link, :2]
        vx, vy = axes[link, 2:]
        hx, hy, hz = half[link]
        ex = abs(ux) * hx + abs(vx) * hy
        ey = abs(uy) * hx + abs(vy) * hy
        best = min(x - ex - bounds[0, 0], bounds[1, 0] - x - ex,
                   y - ey - bounds[0, 1], bounds[1, 1] - y - ey,
                   z - hz - bounds[0, 2], bounds[1, 2] - z - hz)
        best = max(0., best)
        for ox, oy, oz, bx, by, bz in obstacles:
            dx, dy, dz = x - ox, y - oy, z - oz
            gx = abs(dx) - bx - ex
            gy = abs(dy) - by - ey
            gz = max(0., abs(dz) - bz - hz)
            lower = max(gx, 0.)**2 + max(gy, 0.)**2 + gz**2
            if lower >= best**2:
                continue
            gu = abs(ux * dx + uy * dy) - hx - abs(ux) * bx - abs(uy) * by
            gv = abs(vx * dx + vy * dy) - hy - abs(vx) * bx - abs(vy) * by
            horizontal = 0.
            if max(gx, gy, gu, gv) > 0.:
                # For two disjoint rectangles, a closest pair includes a vertex
                # of at least one rectangle. Test both vertex sets.
                horizontal = np.inf
                for sign_x in (-1., 1.):
                    for sign_y in (-1., 1.):
                        a, b = sign_x * hx, sign_y * hy
                        horizontal = min(horizontal, _point_rectangle_squared(
                            dx + a * ux + b * vx, dy + a * uy + b * vy, bx, by))
                        a, b = sign_x * bx - dx, sign_y * by - dy
                        horizontal = min(horizontal, _point_rectangle_squared(
                            a * ux + b * uy, a * vx + b * vy, hx, hy))
            best = min(best, np.sqrt(horizontal + gz**2))
        result[link] = best
    return result


class UprightBoxClearance:
    """Unsigned separation of upright boxes under the robot's planar kinematics."""

    @classmethod
    def for_scene(cls, scene, *, margin):
        if scene.base_height is None or any(obstacle.box_bounds is None for obstacle in scene.obstacles):
            return None
        checker = scene.checker
        robot = scene.robot
        reference, _ = robot.fk(robot.limits.mean(axis=1))
        for joint in robot.joints:
            if joint.dof is not None and not joint.prismatic:
                axis = reference[joint.parent] @ joint.basis @ joint.axis
                if np.linalg.norm(axis[:2]) > 1e-12:
                    return None
        vertical = np.abs(reference[:, 2]).argmax(axis=1)
        if not np.allclose(np.abs(reference[np.arange(len(vertical)), 2, vertical]), 1, rtol=0, atol=1e-14):
            return None
        model = cls(checker, scene.bounds, scene.obstacles, vertical, margin)
        # Compile or load the distance kernel before timing a planning query.
        model.clearances(scene.start)
        return model

    def __init__(self, checker, bounds, obstacles, vertical, margin):
        self.checker = checker
        self.bounds = bounds
        self.margin = max(margin, checker.margin)
        self.obstacles = np.array([np.r_[box.box_bounds.mean(axis=0),
                                         np.diff(box.box_bounds, axis=0)[0] / 2]
                                   for box in obstacles]).reshape(-1, 6)
        robot = checker.robot
        self.order = np.array([[*(axis for axis in range(3) if axis != up), up] for up in vertical])
        self.half = np.take_along_axis(robot.half_extents, self.order, axis=1)
        self.joints = [joint for joint in robot.joints if joint.dof is not None]
        motion = np.zeros((len(robot.links), len(robot.speeds) + 3))
        motion[:, 2] = robot.reach[0]
        for joint in self.joints:
            reaches = 1. if joint.prismatic else robot.reach[joint.child]
            motion[:, joint.dof + 3] = robot.ancestors[:, joint.dof] * reaches
        self.motion = motion

    def transforms(self, state, fk):
        local_r, local_t = fk
        base = state.rotation.as_matrix()
        rotations = base @ local_r
        centers = ((rotations @ self.checker.robot.centers[..., None])[..., 0]
                   + local_t @ base.T + state.position)
        ordered = np.take_along_axis(rotations, self.order[:, None, :], axis=2)
        return centers, rotations, ordered

    def link_clearances(self, state, fk):
        centers, _, ordered = self.transforms(state, fk)
        axes = ordered[:, :2, :2].swapaxes(1, 2).reshape(-1, 4)
        # The subtraction also covers floating-point noise in the upright tests.
        distances = _link_clearances(centers, axes, self.half, self.obstacles, self.bounds)
        return np.maximum(0., distances - 1e-10)

    def clearances(self, state):
        checker = self.checker
        fk = checker._collision_fk(state.joints)
        return self.link_clearances(state, fk).min(), checker.self_clearance(state.joints, fk)

    def free_interval_radius(self, state, link_rates, pair_rates, margin=None):
        checker = self.checker
        fk = checker._collision_fk(state.joints)
        slack = self.link_clearances(state, fk) - (self.margin if margin is None else margin)
        if np.any(slack <= 0.):
            return 0.
        moving = link_rates > 0.
        radius = np.divide(slack, link_rates, out=np.full_like(slack, np.inf), where=moving).min()
        return checker.self_interval_radius(fk, pair_rates, radius)

    def edge_is_free(self, start, end, *, margin):
        checker = self.checker
        if not checker.valid(start) or not checker.valid(end):
            return False
        edge = checker.edge(start, end)
        joint = np.abs(edge.joint_delta)
        link_rates = (np.linalg.norm(edge.base.delta) + np.linalg.norm(edge.base.rotvec) * self.motion[:, 2]
                      + self.motion[:, 3:] @ joint)
        pair_rates = checker.dynamic_rates @ joint

        def radius_at(fraction):
            state = start if fraction == 0. else end if fraction == 1. else edge.at(fraction)
            return self.free_interval_radius(state, link_rates, pair_rates, margin)

        return certify_interval(radius_at)

    def planes(self, state, activation):
        """Yield nearby obstacle planes with a support-point Jacobian."""
        if not len(self.obstacles):
            return
        robot = self.checker.robot
        fk = self.checker._collision_fk(state.joints)
        centers, rotations, ordered = self.transforms(state, fk)
        count = len(centers)
        horizontal = ordered[:, :, :2].swapaxes(1, 2).copy()
        horizontal[:, :, 2] = 0.
        axes = np.concatenate((np.broadcast_to(np.eye(3), (count, 3, 3)), horizontal), axis=1)
        centers_delta = centers[:, None] - self.obstacles[None, :, :3]
        projection = centers_delta @ axes.swapaxes(1, 2)
        obstacle_extent = self.obstacles[:, 3:] @ np.abs(axes).swapaxes(1, 2)
        local_axes = axes @ rotations
        link_extent = (np.abs(local_axes) @ robot.half_extents[..., None])[..., 0]
        gaps = np.abs(projection) - obstacle_extent - link_extent[:, None, :]
        choice = gaps.argmax(axis=2)
        best = np.take_along_axis(gaps, choice[..., None], axis=2)[..., 0]
        local_r, local_t = fk
        base = state.rotation.as_matrix()
        directions = [(base @ local_r[joint.parent] @ joint.basis @ joint.axis) for joint in self.joints]
        origins = [state.position + base @ (local_t[joint.parent] + local_r[joint.parent] @ joint.offset)
                   for joint in self.joints]
        for link, obstacle in zip(*np.where(best < activation)):
            axis = choice[link, obstacle]
            sign = 1. if projection[link, obstacle, axis] >= 0. else -1.
            normal = sign * axes[link, axis]
            support_direction = sign * local_axes[link, axis]
            # On a support face, the center is a valid support point.
            support_sign = np.where(np.abs(support_direction) < 1e-12, 0., np.sign(support_direction))
            witness = centers[link] - rotations[link] @ (robot.half_extents[link] * support_sign)
            delta = witness - state.position
            gradient = np.zeros(3 + len(self.joints))
            gradient[:2] = normal[:2]
            gradient[2] = -normal[0] * delta[1] + normal[1] * delta[0]
            for joint, direction, origin in zip(self.joints, directions, origins):
                if robot.ancestors[link, joint.dof]:
                    jacobian = direction if joint.prismatic else np.cross(direction, witness - origin)
                    gradient[3 + joint.dof] = normal @ jacobian
            yield best[link, obstacle], gradient
