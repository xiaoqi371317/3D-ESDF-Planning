"""Optional PyTorch refinement of a collision-checked seed path."""

from collections.abc import Callable
from dataclasses import dataclass
from math import ceil

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.nn.functional import grid_sample

from .collision import CollisionChecker, Edge, Pose


@dataclass
class Refinement:
    path: list[Pose]
    accepted: bool
    initial_cost: float
    final_cost: float


def _rotate(vectors: torch.Tensor, rotvecs: torch.Tensor) -> torch.Tensor:
    axes = rotvecs[:, None, :].expand_as(vectors)
    theta = torch.linalg.vector_norm(rotvecs, dim=-1)[:, None, None]
    cross = torch.cross(axes, vectors, dim=-1)
    return (vectors + torch.sinc(theta / torch.pi) * cross
            + .5 * torch.sinc(theta / (2 * torch.pi)).square() * torch.cross(axes, cross, dim=-1))


class TorchESDF:
    def __init__(self, field):
        self.volume = torch.as_tensor(field.values.transpose(2, 1, 0).copy())[None, None]
        self.lower = torch.as_tensor(field.origin, dtype=self.volume.dtype)
        self.upper = torch.as_tensor(field.high, dtype=self.volume.dtype)
        self.correction = field.correction

    def interpolate(self, points):
        normalized = 2 * (points - self.lower) / (self.upper - self.lower) - 1
        return (grid_sample(self.volume, normalized[None, :, :, None], padding_mode="border",
                            align_corners=True)[0, 0, :, :, 0] - self.correction)

    def query(self, points):
        boundary = torch.minimum(points - self.lower, self.upper - points).amin(dim=-1)
        return torch.minimum(self.interpolate(points), boundary)


def refine(path: list[Pose], checker: CollisionChecker, iterations: int = 45,
           cancelled: Callable[[], bool] = lambda: False) -> Refinement:
    if len(path) < 2 or iterations < 0:
        raise ValueError("Refinement requires two poses and a nonnegative iteration count.")

    waypoints = [path[0]]
    for start, end in zip(path, path[1:]):
        edge = Edge(start, end, checker.robot.radius)
        count = max(2, ceil(edge.bound / .42))
        waypoints.extend(edge.at(i / count) for i in range(1, count))
        waypoints.append(end)

    def tensor(values):
        return torch.as_tensor(values, dtype=torch.float32)

    positions = tensor(np.stack([pose.position for pose in waypoints]))
    rotations = tensor(np.stack([pose.rotation.as_matrix() for pose in waypoints]))
    body_points = tensor(checker.robot.points)
    base_points = body_points @ rotations.transpose(1, 2)
    base_axes = rotations.transpose(1, 2)
    field = TorchESDF(checker.field)
    safe_distance = max(.045, checker.margin + .025)
    zeros = torch.zeros((1, 6))
    offsets = torch.zeros((len(waypoints) - 2, 6), requires_grad=True)

    def objective():
        # Endpoints are constant. Each free waypoint gets a translation and a
        # world-frame rotation vector relative to its initial orientation.
        changes = torch.cat((zeros, offsets, zeros))
        centers = positions + changes[:, :3]
        axes = _rotate(base_axes, changes[:, 3:])
        points = _rotate(base_points, changes[:, 3:]) + centers[:, None, :]
        clearance = field.query(points).amin(dim=1)
        states = torch.cat((centers, checker.robot.radius / np.sqrt(3) * axes.flatten(1)), dim=1)
        acceleration = states.diff(n=2, dim=0).square().sum(dim=1).mean()
        obstacle_cost = (safe_distance - clearance).relu().square().mean()
        return acceleration + 65 * obstacle_cost + .1 * offsets.square().mean()

    optimizer = torch.optim.Adam([offsets], lr=.007)
    initial_cost = float(objective().detach())
    best_cost = initial_cost
    best_path = path
    for iteration in range(iterations):
        if cancelled():
            break
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        optimizer.step()
        if (iteration + 1) % 5 and iteration + 1 < iterations:
            continue
        with torch.no_grad():
            candidate_cost = float(objective())
            if not np.isfinite(candidate_cost) or candidate_cost >= best_cost - 1e-7:
                continue
            delta = offsets.detach().numpy()
            candidate = [path[0]]
            for pose, change in zip(waypoints[1:-1], delta):
                candidate.append(Pose.at(pose.position + change[:3],
                                         Rotation.from_rotvec(change[3:]) * pose.rotation))
            candidate.append(path[-1])
            if all(checker.edge_is_free(a, b) for a, b in zip(candidate, candidate[1:])):
                best_cost = candidate_cost
                best_path = candidate
    return Refinement(best_path, best_path is not path, initial_cost, best_cost)
