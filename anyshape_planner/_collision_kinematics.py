"""Compiled rigid transforms used in repeated articulated collision queries."""

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _forward(q, parents, children, dofs, prismatic, bases, origins, projections, skews, directions):
    count = len(parents) + 1
    rotations = np.empty((count, 3, 3))
    translations = np.empty((count, 3))
    rotations[0] = np.eye(3)
    translations[0] = 0.
    for index, (parent, child, dof) in enumerate(zip(parents, children, dofs)):
        position = origins[index]
        basis = bases[index]
        if dof >= 0:
            value = q[dof]
            if prismatic[index]:
                position = position + value * directions[index]
            else:
                cosine, sine = np.cos(value), np.sin(value)
                basis = cosine * basis + (1 - cosine) * projections[index] + sine * skews[index]
        for row in range(3):
            x, y, z = rotations[parent, row]
            translations[child, row] = x * position[0] + y * position[1] + z * position[2] + translations[parent, row]
            for col in range(3):
                rotations[child, row, col] = x * basis[0, col] + y * basis[1, col] + z * basis[2, col]
    return rotations, translations


class CollisionKinematics:
    def __init__(self, robot):
        joints = robot.joints
        self.reference = robot.fk
        self.joint_shape = robot.speeds.shape
        self.parents = np.array([joint.parent for joint in joints], dtype=np.intp)
        self.children = np.array([joint.child for joint in joints], dtype=np.intp)
        self.dofs = np.array([joint.dof if joint.dof is not None else -1 for joint in joints], dtype=np.intp)
        self.prismatic = np.array([joint.prismatic for joint in joints], dtype=bool)
        self.bases = np.array([joint.basis for joint in joints]).reshape(-1, 3, 3)
        self.origins = np.array([joint.offset for joint in joints]).reshape(-1, 3)
        axes = np.array([joint.axis for joint in joints]).reshape(-1, 3)
        x, y, z = axes.T
        zero = np.zeros(len(axes))
        cross = np.stack((zero, -z, y, z, zero, -x, -y, x, zero), axis=1).reshape(-1, 3, 3)
        self.directions = (self.bases @ axes[..., None])[..., 0]
        self.projections = self.directions[..., None] * axes[:, None, :]
        self.skews = self.bases @ cross

    def forward(self, q):
        if not isinstance(q, np.ndarray) or q.shape != self.joint_shape:
            return self.reference(q)
        return _forward(q, self.parents, self.children, self.dofs, self.prismatic, self.bases,
                        self.origins, self.projections, self.skews, self.directions)


@njit(cache=True, nogil=True)
def pair_geometry(indices, rotations, translations, centers, extents):
    left, right = indices
    count = len(left)
    relative = np.empty((count, 3, 3))
    offsets = np.empty((count, 3))
    gaps = np.empty(count)
    for index, (a, b) in enumerate(zip(left, right)):
        gap = -np.inf
        dx, dy, dz = translations[b] - translations[a]
        for row in range(3):
            x, y, z = rotations[a, :, row]
            offset = x * dx + y * dy + z * dz
            offsets[index, row] = offset
            center = offset - centers[a, row]
            extent = extents[a, row]
            for col in range(3):
                value = x * rotations[b, 0, col] + y * rotations[b, 1, col] + z * rotations[b, 2, col]
                relative[index, row, col] = value
                center += value * centers[b, col]
                extent += abs(value) * extents[b, col]
            gap = max(gap, abs(center) - extent)
        gaps[index] = gap
    return relative, offsets, gaps
