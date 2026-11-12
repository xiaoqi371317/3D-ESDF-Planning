"""Procedural window and clutter benchmarks, with dimensions in metres."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .collision import CollisionChecker, Pose
from .geometry import ESDF, Obstacle, Robot


@dataclass
class Scene:
    name: str
    bounds: np.ndarray
    obstacles: list[Obstacle]
    start: Pose
    goal: Pose
    robot: Robot
    portal: Pose | None = None
    resolution: float = 0.02

    @cached_property
    def checker(self) -> CollisionChecker:
        return CollisionChecker(self.robot, ESDF.build(self.bounds, self.resolution, self.obstacles))

    def sample_portal(self, rng: np.random.Generator) -> Pose:
        assert self.portal is not None
        offset = rng.normal(0, (0.64, 0.13, 0.15))
        rotation = Rotation.from_rotvec(rng.normal(0, 0.23, 3)) * self.portal.rotation
        return Pose.at(np.clip(self.portal.position + offset, *self.bounds), rotation)


def l_mesh() -> trimesh.Trimesh:
    polygon = np.array([[-.67, -.28], [.67, -.28], [.67, -.04],
                        [-.43, -.04], [-.43, .28], [-.67, .28]])
    triangles = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]])
    mesh = trimesh.creation.extrude_triangulation(polygon, triangles, height=.22)
    mesh.vertices = mesh.vertices[:, [2, 0, 1]] - np.array([.11, 0, 0])
    return mesh


def window(robot: Robot | None = None) -> Scene:
    bounds = np.array([[-2.90, -1.60, -1.60], [2.90, 1.60, 1.60]])
    # Four solid slabs span the grid. The only connection between the two
    # chambers is the 0.80 x 0.88 m window in the x-normal wall.
    obstacles = [
        Obstacle.box((-.15, -1.62, -1.62), (.15, -.40, 1.62)),
        Obstacle.box((-.15, .40, -1.62), (.15, 1.62, 1.62)),
        Obstacle.box((-.15, -.40, -1.62), (.15, .40, -.44)),
        Obstacle.box((-.15, -.40, .44), (.15, .40, 1.62)),
    ]
    return Scene("窗口穿越", bounds, obstacles, Pose.at((-2.04, -.14, -.16)),
                 Pose.at((2.04, .18, .14)), robot or Robot.from_mesh(l_mesh()),
                 Pose.at((0, 0, 0), Rotation.from_euler("z", 90, degrees=True)))


def clutter(seed: int = 42, robot: Robot | None = None) -> Scene:
    rng = np.random.default_rng(seed)
    bounds = np.array([[-2.90, -1.60, -1.60], [2.90, 1.60, 1.60]])
    obstacles = []
    for index in range(13):
        center = rng.uniform((-1.37, -1.15, -.95), (1.37, 1.15, .85))
        size = rng.uniform((.22, .25, .40), (.55, .56, 1.03))
        if index % 2:
            mesh = trimesh.creation.cylinder(radius=size[0] / 2, height=size[2], sections=24)
            mesh.apply_translation(center)
            obstacles.append(Obstacle(mesh))
        else:
            obstacles.append(Obstacle.box(center - size / 2, center + size / 2))
    return Scene("杂乱环境", bounds, obstacles, Pose.at((-2.04, -.30, -.25)),
                 Pose.at((2.04, .34, .29)), robot or Robot.from_mesh(l_mesh()))
