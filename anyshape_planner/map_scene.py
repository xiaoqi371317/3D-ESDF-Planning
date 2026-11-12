"""An obstacle map with two offset, orthogonal openings."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.spatial.transform import Rotation

from .articulated import ArticulatedRobot
from .articulated_collision import ArticulatedCollision, Configuration
from .collision import Pose
from .geometry import ESDF, Obstacle
from .mobile import MOBILE_MODEL, MecanumDrive

MAP_SCENE = "地图探索 · 双窄口转弯"


@dataclass
class MapScene:
    robot: ArticulatedRobot
    bounds: np.ndarray
    obstacles: list[Obstacle]
    start: Configuration
    goal: Configuration
    base_height: float = 0.
    resolution: float = .025
    drive: MecanumDrive = MecanumDrive()
    name: str = MAP_SCENE

    @cached_property
    def checker(self):
        return ArticulatedCollision(self.robot, ESDF.build(self.bounds, self.resolution, self.obstacles),
                                    base_height=self.base_height)


def map_scene():
    robot = ArticulatedRobot.from_urdf(MOBILE_MODEL, sample_pitch=.05)
    bounds = np.array([[-4.35, -2.65, -.12], [3.30, 4.30, 1.43]])
    # Two partitions form a left room, a turning room and an upper room.
    # Only obstacle volumes are passed to the map planner.
    obstacles = [
        Obstacle.box((-2.10, -2.70, -.15), (-.95, -1.58, 1.47)),
        Obstacle.box((-2.10, -.74, -.15), (-.95, 4.35, 1.47)),
        Obstacle.box((-2.10, -1.58, .89), (-.95, -.74, 1.47)),
        Obstacle.box((-.95, .70, -.15), (1.13, 1.85, 1.47)),
        Obstacle.box((1.91, .70, -.15), (3.35, 1.85, 1.47)),
        Obstacle.box((1.13, .70, .77), (1.91, 1.85, 1.47)),
    ]
    nominal = robot.postures["nominal"]
    start = Configuration(Pose.at((-3.24, -1.34, 0.), Rotation.from_euler("z", -8, degrees=True)), nominal)
    goal = Configuration(Pose.at((1.72, 3.23, 0.), Rotation.from_euler("z", 87, degrees=True)), nominal)
    return MapScene(robot, bounds, obstacles, start, goal)
