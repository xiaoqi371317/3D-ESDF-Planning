"""An obstacle map with two offset, orthogonal openings."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.spatial.transform import Rotation

from .articulated import ArticulatedRobot
from .articulated_collision import ArticulatedCollision, Configuration
from .box_clearance import UprightBoxClearance
from .collision import Pose
from .geometry import ESDF, Obstacle
from .mobile import MOBILE_MODEL, MecanumDrive

MAP_SCENE = "地图探索 · 双窄口转弯"
LOW_FIRST_MAP_SCENE = "地图探索 · 首口更低"
LOW_FIRST_OPENINGS = ((.86, .77), (.80, .88))
# Reference limits for the simulated lift (m/s²) and two rotary joints (rad/s²).
JOINT_ACCELERATION_LIMITS = (3., 12., 12.)
BOX_CLEARANCE_MARGIN = .008


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
                                    base_height=self.base_height, accelerate=True)

    @cached_property
    def trajectory_geometry(self):
        return UprightBoxClearance.for_scene(self, margin=BOX_CLEARANCE_MARGIN)


def map_scene(*, openings=((.84, .89), (.78, .77)), name=MAP_SCENE):
    (w1, h1), (w2, h2) = openings
    if not (0. < w1 < 3.08 and 0. < w2 < 3.66 and .30 < h1 < 1.47 and .30 < h2 < 1.47):
        raise ValueError("Openings must fit inside their partitions.")
    robot = ArticulatedRobot.from_urdf(MOBILE_MODEL, sample_pitch=.05)
    bounds = np.array([[-4.35, -2.65, -.12], [3.30, 4.30, 1.43]])
    y0, y1 = -1.16 - w1 / 2, -1.16 + w1 / 2
    x0, x1 = 1.52 - w2 / 2, 1.52 + w2 / 2
    # Two partitions form a left room, a turning room and an upper room.
    # Only obstacle volumes are passed to the map planner.
    obstacles = [
        Obstacle.box((-2.10, -2.70, -.15), (-.95, y0, 1.47)),
        Obstacle.box((-2.10, y1, -.15), (-.95, 4.35, 1.47)),
        Obstacle.box((-2.10, y0, h1), (-.95, y1, 1.47)),
        Obstacle.box((-.95, .70, -.15), (x0, 1.85, 1.47)),
        Obstacle.box((x1, .70, -.15), (3.35, 1.85, 1.47)),
        Obstacle.box((x0, .70, h2), (x1, 1.85, 1.47)),
    ]
    nominal = robot.postures["nominal"]
    start = Configuration(Pose.at((-3.24, -1.34, 0.), Rotation.from_euler("z", -8, degrees=True)), nominal)
    goal = Configuration(Pose.at((1.72, 3.23, 0.), Rotation.from_euler("z", 87, degrees=True)), nominal)
    return MapScene(robot, bounds, obstacles, start, goal, name=name)
