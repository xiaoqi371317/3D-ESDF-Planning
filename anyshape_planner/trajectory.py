"""Collision-checked C2 trajectories for a planar mobile manipulator."""

from math import ceil

import numpy as np
from scipy.interpolate import BPoly, PchipInterpolator
from scipy.spatial.transform import Rotation

from .articulated_collision import Configuration
from .collision import Pose
from .controller import TrackingError, TrackingSample
from .planner import PlanningError


def _bisect(points):
    left, right = [points[0]], [points[-1]]
    while len(points) > 1:
        points = (points[:-1] + points[1:]) * .5
        left.append(points[0])
        right.append(points[-1])
    return np.array(left), np.array(right[::-1])


def _velocity_hull(controls, duration):
    parts = [5 * np.diff(controls, axis=0) / duration]
    # Subdivision tightens the convex-hull speed bound without sampling gaps.
    for _ in range(3):
        parts = [part for control in parts for part in _bisect(control)]
    return np.concatenate(parts)


class PlanarTrajectory:
    def __init__(self, path, checker, drive, tangent_scale):
        self.checker = checker
        self.drive = drive
        self.tangent_scale = tangent_scale
        self.start, self.goal = path[0], path[-1]
        robot = checker.robot
        yaw = self.start.rotation.as_rotvec()[2]
        joints = self.start.joints.copy()
        coordinates = [np.r_[self.start.position[:2], yaw, joints]]
        durations = []
        for a, b in zip(path, path[1:]):
            edge = checker.edge(a, b)
            if edge.bound <= 1e-10:
                continue
            yaw += edge.base.rotvec[2]
            joints = joints + edge.joint_delta
            coordinates.append(np.r_[b.position[:2], yaw, joints])
            durations.append(max(.08, np.linalg.norm(edge.base.delta), abs(edge.base.rotvec[2]) / 2.4,
                                 (np.abs(edge.joint_delta) / robot.speeds).max(initial=0),
                                 drive.duration_bound(edge.base) / 1.5))
        if not durations:
            durations.append(.08)
            coordinates.append(coordinates[0])

        coordinates = np.asarray(coordinates)
        duration = np.array(durations)
        knots = np.r_[0, duration.cumsum()]
        tangents = PchipInterpolator(knots, coordinates).derivative()(knots)
        tangents[[0, -1]] = 0
        # Monotone control points keep every joint within adjacent knot limits.
        slopes = 1.25 * np.abs(np.diff(coordinates, axis=0)) / duration[:, None]
        cap = np.minimum(slopes[:-1], slopes[1:])
        tangents[1:-1] = np.clip(tangents[1:-1], -cap, cap) * tangent_scale
        start, end = coordinates[:-1], coordinates[1:]
        outgoing = tangents[:-1] * duration[:, None] / 5
        incoming = tangents[1:] * duration[:, None] / 5
        # The quintic has the same tangent and zero acceleration on both sides
        # of each knot, with zero velocity at the two trajectory endpoints.
        controls = np.stack((start, start + outgoing, start + 2 * outgoing,
                             end - 2 * incoming, end - incoming, end), axis=1)

        rates = []
        scale = 1.0
        for control, seconds in zip(controls, duration):
            hull = _velocity_hull(control, seconds)
            speed = np.linalg.norm(hull[:, :2], axis=1)
            turn, joint = np.abs(hull[:, 2]), np.abs(hull[:, 3:])
            wheel = (np.sqrt(2) * speed + drive.lever * turn) / drive.wheel_radius
            scale = max(scale, 1.01 * max(speed.max(), turn.max() / 2.4,
                        wheel.max() / drive.wheel_speed_limit, (joint / robot.speeds).max(initial=0)))
            point = (speed + robot.radius * turn + joint @ robot.joint_radii).max()
            relative = (joint @ checker.relative_rates.T).max(initial=0)
            rates.append((point, relative))

        self.knots = knots * scale
        self.duration = float(self.knots[-1])
        self.spline = BPoly(np.swapaxes(controls, 0, 1), self.knots)
        self.rates = np.asarray(rates) / scale

    def at(self, time):
        if time <= 0:
            return self.start
        if time >= self.duration:
            return self.goal
        value = self.spline(time)
        robot = self.checker.robot
        joints = np.clip(value[3:], *robot.limits.T)
        joints[robot.continuous] = value[3:][robot.continuous]
        base = Pose.at((*value[:2], self.checker.base_height), Rotation.from_euler("z", value[2]))
        return Configuration(base, joints)

    def _certify(self, cancelled):
        checker = self.checker
        if not checker.is_free(self.goal):
            return False
        for start, end, (point, relative) in zip(self.knots[:-1], self.knots[1:], self.rates):
            time = start
            env_rate = point * checker.field.lipschitz
            for _ in range(2500):
                if cancelled():
                    raise PlanningError("Planning cancelled.")
                env, self_distance = checker.clearances(self.at(time))
                env_slack, self_slack = env - checker.margin, self_distance - checker.self_margin
                if min(env_slack, self_slack) <= 0:
                    return False
                step = min(env_slack / env_rate if env_rate else np.inf,
                           self_slack / relative if relative else np.inf)
                if step > end - time:
                    break
                if step < 1e-8:
                    return False
                time += .95 * step
            else:
                return False
        return True

    def dense_states(self, step=.005):
        if step <= 0:
            raise ValueError("Dense sampling step must be positive.")
        yield self.start
        for a, b, (speed, _) in zip(self.knots[:-1], self.knots[1:], self.rates):
            count = max(1, ceil((b - a) * speed / step))
            for time in np.linspace(a, b, count + 1)[1:]:
                yield self.at(time)

    def wheel_speeds(self, times):
        yaw = self.spline(times)[..., 2]
        vx, vy, turn = np.moveaxis(self.spline(times, nu=1)[..., :3], -1, 0)
        local_x = np.cos(yaw) * vx + np.sin(yaw) * vy
        local_y = -np.sin(yaw) * vx + np.cos(yaw) * vy
        return np.moveaxis(self.drive.wheel_speeds(local_x, local_y, turn), 0, -1)

    @classmethod
    def fit(cls, path, checker, drive, cancelled=lambda: False):
        if len(path) < 2 or checker.base_height is None or not all(checker.valid(state) for state in path):
            raise PlanningError("需要有效的地面平面路点才能生成轨迹。")
        for scale in (1., .35, 0.):
            trajectory = cls(path, checker, drive, scale)
            if not trajectory._certify(cancelled):
                continue
            minimum = np.array([np.inf, np.inf])
            for state in trajectory.dense_states():
                if cancelled():
                    raise PlanningError("Planning cancelled.")
                minimum = np.minimum(minimum, checker.clearances(state))
            if np.all(minimum > (checker.margin, checker.self_margin)):
                return trajectory, tuple(minimum)
        raise PlanningError("样条及保守回退均未通过连续碰撞复检。")


class TrajectoryPlayback:
    """Kinematic playback of the certified time law, including interior knots."""

    def __init__(self, trajectory):
        self.trajectory = trajectory
        self.goal = trajectory.goal
        self.pose = trajectory.start
        self.time = 0.0
        self.samples = [self._sample()]

    def _sample(self):
        checker = self.trajectory.checker
        env, self_distance = checker.clearances(self.pose)
        if env <= checker.margin or self_distance <= checker.self_margin:
            raise TrackingError("轨迹执行时净空不足。")
        return TrackingSample(self.time, self.pose, env, 0, 0, self_distance)

    @property
    def done(self):
        return self.time >= self.trajectory.duration

    @property
    def terminal_error(self):
        return (float(np.linalg.norm(self.pose.position - self.goal.position)),
                float(np.linalg.norm(self.pose.base.relative_rotation(self.goal.base))))

    def step(self, dt):
        if not 0 < dt <= .05:
            raise ValueError("Use a playback timestep in (0, 0.05] seconds.")
        if self.done:
            return self.samples[-1]
        self.time = min(self.time + dt, self.trajectory.duration)
        self.pose = self.trajectory.at(self.time)
        sample = self._sample()
        self.samples.append(sample)
        return sample
