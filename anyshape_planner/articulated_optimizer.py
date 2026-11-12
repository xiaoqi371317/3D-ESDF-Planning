"""Differentiable path refinement with environment and self clearance."""

from dataclasses import dataclass
from math import ceil

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .articulated_collision import ArticulatedCollision, Configuration
from .collision import Pose
from .optimizer import TorchESDF, _rotate


@dataclass
class ArticulatedRefinement:
    path: list[Configuration]
    accepted: bool
    initial_cost: float
    final_cost: float


def refine_articulated(path, checker: ArticulatedCollision, iterations=40, cancelled=lambda: False):
    if len(path) < 2 or iterations < 0:
        raise ValueError("Refinement requires two configurations and a nonnegative iteration count.")
    robot = checker.robot
    waypoints = [path[0]]
    for start, end in zip(path, path[1:]):
        edge = checker.edge(start, end)
        count = max(2, ceil(edge.bound / .37))
        waypoints.extend(edge.at(i / count) for i in range(1, count))
        waypoints.append(end)

    def tensor(values):
        return torch.as_tensor(values, dtype=torch.float32)

    positions = tensor(np.stack([state.position for state in waypoints]))
    axes = tensor(np.stack([state.rotation.as_matrix().T for state in waypoints]))
    angles = tensor(np.stack([state.joints for state in waypoints]))
    joint_weights = tensor(robot.joint_radii)
    continuous = torch.as_tensor(robot.continuous)
    bounded = ~continuous
    low, high = tensor(robot.limits).T
    nominal = tensor(robot.postures["nominal"])
    points = [tensor(link.geometry.points) for link in robot.links]
    centers, half = tensor(robot.centers), tensor(robot.half_extents)
    environment = TorchESDF(checker.field)
    local = [TorchESDF(field) for field in checker.local_fields]
    zeros = angles.new_zeros((1, 6 + angles.shape[1]))
    offsets = angles.new_zeros((len(waypoints) - 2, zeros.shape[1]), requires_grad=True)

    def shortest(delta):
        return torch.where(continuous, torch.atan2(delta.sin(), delta.cos()), delta)

    def objective():
        changes = torch.cat((zeros, offsets, zeros))
        base = positions + changes[:, :3]
        rotated_axes = _rotate(axes, changes[:, 3:6])
        joints = angles + changes[:, 6:]
        rotations, translations = robot.fk_torch(joints)
        link_points = torch.cat([cloud @ rotations[:, i].transpose(1, 2) + translations[:, i, None]
                                 for i, cloud in enumerate(points)], dim=1)
        world = link_points @ rotated_axes + base[:, None]
        env_clearance = environment.query(world).amin(dim=1)
        env_penalty = (max(.050, checker.margin + .025) - env_clearance).relu().square().mean()

        pair_clearances = []
        for a, b in checker.pairs:
            inverse_a = rotations[:, a].transpose(1, 2)
            relative = inverse_a @ rotations[:, b]
            origin = (inverse_a @ (translations[:, b] - translations[:, a])[..., None])[..., 0]
            center = (relative @ centers[b]) + origin
            extents = relative.abs() @ half[b]
            gap = ((center - centers[a]).abs() - half[a] - extents).amax(dim=1)
            b_in_a = points[b] @ relative.transpose(1, 2) + origin[:, None]
            a_in_b = (points[a] - origin[:, None]) @ relative
            distance = torch.minimum(local[a].interpolate(b_in_a).amin(dim=1),
                                     local[b].interpolate(a_in_b).amin(dim=1))
            pair_clearances.append(torch.maximum(gap, distance))
        self_penalty = angles.new_zeros(())
        if pair_clearances:
            closest = torch.stack(pair_clearances).amin(dim=0)
            self_penalty = (max(.030, checker.self_margin + .015) - closest).relu().square().mean()

        rigid = torch.cat((base, robot.radius / np.sqrt(3) * rotated_axes.flatten(1)), dim=1)
        acceleration = rigid.diff(n=2, dim=0).square().sum(dim=1).mean()
        joint_velocity = shortest(joints.diff(dim=0)) * joint_weights
        acceleration += joint_velocity.diff(dim=0).square().sum(dim=1).mean()
        violations = ((low - joints).relu().square() + (joints - high).relu().square()) * bounded
        preference = shortest(joints - nominal).square().sum(dim=1).mean()
        return (acceleration + 100 * env_penalty + 75 * self_penalty
                + 80 * violations.sum(dim=1).mean() + .008 * preference + .07 * offsets.square().mean())

    adam = torch.optim.Adam([offsets], lr=.006)
    initial_cost = float(objective().detach())
    best_cost, best = initial_cost, path
    for index in range(iterations):
        if cancelled():
            break
        adam.zero_grad()
        objective().backward()
        adam.step()
        if (index + 1) % 5 and index + 1 < iterations:
            continue
        with torch.no_grad():
            cost = float(objective())
            if not np.isfinite(cost) or cost >= best_cost - 1e-7:
                continue
            candidate = [path[0]]
            for state, change in zip(waypoints[1:-1], offsets.detach().numpy()):
                pose = Pose.at(state.position + change[:3], Rotation.from_rotvec(change[3:6]) * state.rotation)
                candidate.append(Configuration(pose, state.joints + change[6:]))
            candidate.append(path[-1])
            if all(checker.edge_is_free(a, b) for a, b in zip(candidate, candidate[1:])):
                best_cost, best = cost, candidate
    return ArticulatedRefinement(best, best is not path, initial_cost, best_cost)
