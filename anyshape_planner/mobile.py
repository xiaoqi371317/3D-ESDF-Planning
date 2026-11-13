"""Grounded mecanum manipulator and a low, narrow corridor."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .articulated import ArticulatedRobot
from .articulated_collision import Configuration
from .collision import Pose
from .corridor import Corridor
from .geometry import Obstacle

MOBILE_MODEL = Path(__file__).with_name("models") / "mecanum_arm.urdf"
MOBILE_SCENE = "全向底盘折叠走廊"


@dataclass(frozen=True)
class MecanumDrive:
    wheel_radius: float = .12
    half_wheelbase: float = .44
    half_track: float = .283
    wheel_speed_limit: float = 10.0

    @property
    def lever(self):
        return self.half_wheelbase + self.half_track

    def wheel_speeds(self, vx, vy, yaw_rate):
        """Body-frame inverse kinematics for FL, FR, RL, RR (rad/s)."""
        turn = self.lever * yaw_rate
        return np.array((vx - vy - turn, vx + vy + turn,
                         vx + vy - turn, vx - vy + turn)) / self.wheel_radius

    def body_velocity(self, wheels):
        """Recover the planar body twist from FL, FR, RL, RR wheel speeds."""
        fl, fr, rl, rr = wheels
        return self.wheel_radius / 4 * np.array(
            (fl + fr + rl + rr, -fl + fr + rl - rr, (-fl + fr - rl + rr) / self.lever))

    def limit(self, rotation, linear, angular):
        local = rotation.apply(linear, inverse=True)
        peak = np.abs(self.wheel_speeds(*local[:2], angular[2])).max()
        scale = min(1.0, self.wheel_speed_limit / peak) if peak else 1.0
        return linear * scale, angular * scale

    def between(self, start, end, dt):
        linear = start.rotation.apply((end.position - start.position) / dt, inverse=True)
        yaw_rate = start.relative_rotation(end)[2] / dt
        return self.wheel_speeds(*linear[:2], yaw_rate)

    def duration_bound(self, edge):
        # |vx| + |vy| is at most sqrt(2) times the translational speed.
        travel = np.sqrt(2) * np.linalg.norm(edge.delta[:2]) + self.lever * abs(edge.rotvec[2])
        return 1.5 * travel / (self.wheel_radius * self.wheel_speed_limit)


@dataclass
class MobileCorridor(Corridor):
    drive: MecanumDrive = MecanumDrive()


def mobile_corridor(*, width=.78, height=.83):
    robot = ArticulatedRobot.from_urdf(MOBILE_MODEL, sample_pitch=.05)
    bounds = np.array([[-3.76, -1.50, -.12], [3.76, 1.50, 1.46]])
    half = width / 2
    obstacles = [Obstacle.box((-1.19, -1.52, -.14), (1.19, -half, 1.48)),
                 Obstacle.box((-1.19, half, -.14), (1.19, 1.52, 1.48)),
                 Obstacle.box((-1.19, -half, height), (1.19, half, 1.48))]
    nominal = robot.postures["nominal"]
    start = Configuration(Pose.at((-2.70, -.24, 0), Rotation.from_euler("z", -9, degrees=True)), nominal)
    goal = Configuration(Pose.at((2.70, .20, 0), Rotation.from_euler("z", 8, degrees=True)), nominal)
    return MobileCorridor(robot, bounds, obstacles, start, goal, half_length=1.19, half_width=half,
                          half_height=height / 2, name=MOBILE_SCENE, base_height=0.0, portal_offset=2.04)
