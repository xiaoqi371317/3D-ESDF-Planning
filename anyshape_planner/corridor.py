"""Corridor and rooms for the floating-base dual-arm benchmark."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from .articulated import ArticulatedRobot
from .articulated_collision import ArticulatedCollision, Configuration
from .collision import Pose
from .geometry import ESDF, Obstacle

MODEL = Path(__file__).with_name("models") / "dual_arm.urdf"


@dataclass
class Corridor:
    robot: ArticulatedRobot
    bounds: np.ndarray
    obstacles: list[Obstacle]
    start: Configuration
    goal: Configuration
    half_length: float = 1.05
    half_width: float = .37
    half_height: float = .31
    name: str = "双臂收缩走廊"

    @cached_property
    def checker(self):
        return ArticulatedCollision(self.robot, ESDF.build(self.bounds, .02, self.obstacles))

    @property
    def portals(self):
        return tuple(Pose.at((x, 0, 0)) for x in (-1.75, 1.75))


def corridor(robot: ArticulatedRobot | None = None) -> Corridor:
    robot = ArticulatedRobot.from_urdf(MODEL) if robot is None else robot
    if not {"nominal", "compact"} <= robot.postures.keys():
        raise ValueError("The corridor model needs nominal and compact postures in its .poses.json file.")
    bounds = np.array([[-3.4, -1.2, -1.04], [3.4, 1.2, 1.04]])
    obstacles = [Obstacle.box((-1.05, -1.22, -1.06), (1.05, -.37, 1.06)),
                 Obstacle.box((-1.05, .37, -1.06), (1.05, 1.22, 1.06)),
                 Obstacle.box((-1.05, -.37, -1.06), (1.05, .37, -.31)),
                 Obstacle.box((-1.05, -.37, .31), (1.05, .37, 1.06))]
    nominal = robot.postures["nominal"]
    return Corridor(robot, bounds, obstacles, Configuration(Pose.at((-2.46, -.17, 0)), nominal),
                    Configuration(Pose.at((2.46, .17, 0)), nominal))
