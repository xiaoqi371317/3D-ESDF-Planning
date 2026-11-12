"""URDF geometry and forward kinematics for a free-floating link tree."""

from dataclasses import dataclass
import json
from math import pi
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .geometry import Obstacle, Robot, load_mesh


def _vector(element, attribute, default):
    if element is None or attribute not in element.attrib:
        return np.array(default, dtype=float)
    values = np.array([float(value) for value in element.attrib[attribute].split()])
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"Expected three finite values for {attribute}.")
    return values


def _origin(element):
    origin = element.find("origin")
    return (_vector(origin, "xyz", (0, 0, 0)),
            Rotation.from_euler("xyz", _vector(origin, "rpy", (0, 0, 0))).as_matrix())


def _link_shape(element, directory, parts):
    meshes = []
    axis_aligned_box = False
    for part in parts:
        geometry = part.find("geometry")
        position, rotation = _origin(part)
        shape = geometry[0]
        if shape.tag == "box":
            mesh = trimesh.creation.box(extents=_vector(shape, "size", (0, 0, 0)))
            axis_aligned_box = len(parts) == 1 and np.allclose(rotation, np.eye(3))
        elif shape.tag == "cylinder":
            mesh = trimesh.creation.cylinder(radius=float(shape.attrib["radius"]),
                                             height=float(shape.attrib["length"]))
        elif shape.tag == "sphere":
            mesh = trimesh.creation.icosphere(radius=float(shape.attrib["radius"]))
        elif shape.tag == "mesh":
            filename = shape.attrib["filename"]
            if "://" in filename:
                raise ValueError(f"Use a filesystem path for URDF meshes: {filename}")
            mesh = load_mesh(directory / filename)
            mesh.apply_scale(_vector(shape, "scale", (1, 1, 1)))
        else:
            raise ValueError(f"Unsupported URDF geometry: {shape.tag}")
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation, position
        meshes.append(mesh.apply_transform(transform))
    if not meshes:
        raise ValueError(f"Link {element.attrib['name']} has no geometry.")
    return trimesh.util.concatenate(meshes), axis_aligned_box


@dataclass
class Link:
    name: str
    geometry: Robot
    obstacle: Obstacle
    visual: trimesh.Trimesh | None = None


@dataclass
class Joint:
    name: str
    parent: int
    child: int
    offset: np.ndarray
    basis: np.ndarray
    axis: np.ndarray
    dof: int | None
    prismatic: bool


class ArticulatedRobot:
    def __init__(self, links, joints, limits, speeds, continuous, postures):
        self.links = links
        self.joints = joints
        self.limits = np.asarray(limits, dtype=float).reshape(-1, 2)
        self.speeds = np.asarray(speeds, dtype=float)
        self.continuous = np.asarray(continuous, dtype=bool)
        self.prismatic = np.array([joint.prismatic for joint in joints if joint.dof is not None], dtype=bool)
        self.postures = {name: np.asarray(values, dtype=float) for name, values in postures.items()}
        for name, angles in self.postures.items():
            if not self.valid(angles):
                raise ValueError(f"Named posture {name} violates the joint limits.")

        count = len(links)
        distances = np.full((count, count), np.inf)
        np.fill_diagonal(distances, 0.0)
        ancestors = np.zeros((count, len(self.speeds)), dtype=bool)
        for joint in joints:
            offset = joint.offset
            if joint.prismatic:
                offset = offset + self.limits[joint.dof, :, None] * (joint.basis @ joint.axis)
            length = np.linalg.norm(offset, axis=-1).max()
            distances[joint.parent, joint.child] = distances[joint.child, joint.parent] = length
            ancestors[joint.child] = ancestors[joint.parent]
            if joint.dof is not None:
                ancestors[joint.child, joint.dof] = True
        for index in range(count):
            distances = np.minimum(distances, distances[:, index, None] + distances[None, index, :])
        # Using AABB corners also bounds the broad-phase projections in link frames.
        self.centers = np.array([link.geometry.mesh.bounds.mean(axis=0) for link in links])
        self.half_extents = np.array([link.geometry.mesh.extents / 2 for link in links])
        radii = np.linalg.norm(np.abs(self.centers) + self.half_extents, axis=1)
        reach = distances + radii[None, :]
        self.radius = float(reach[0].max())
        self.joint_radii = np.array([1.0 if joint.prismatic else reach[joint.child, ancestors[:, joint.dof]].max()
                                     for joint in joints if joint.dof is not None])
        self.ancestors = ancestors
        self.reach = reach
        self.adjacent = {tuple(sorted((joint.parent, joint.child))) for joint in joints}

    @classmethod
    def from_urdf(cls, path, sample_pitch: float = .05):
        path = Path(path)
        if not np.isfinite(sample_pitch) or sample_pitch <= 0:
            raise ValueError("Robot sample pitch must be positive and finite.")
        root = ET.parse(path).getroot()
        elements = {link.attrib["name"]: link for link in root.findall("link")}
        raw_joints = root.findall("joint")
        children = [joint.find("child").attrib["link"] for joint in raw_joints]
        roots = set(elements) - set(children)
        if len(roots) != 1 or len(set(children)) != len(children):
            raise ValueError("URDF must describe a single tree.")

        names = [roots.pop()]
        joints, limits, speeds, continuous = [], [], [], []
        pending = list(raw_joints)
        while pending:
            ready = [joint for joint in pending if joint.find("parent").attrib["link"] in names]
            if not ready:
                raise ValueError("URDF has a disconnected or cyclic joint tree.")
            for element in ready:
                pending.remove(element)
                parent = names.index(element.find("parent").attrib["link"])
                child_name = element.find("child").attrib["link"]
                if child_name not in elements or child_name in names:
                    raise ValueError(f"Invalid joint child {child_name}.")
                child = len(names)
                names.append(child_name)
                position, basis = _origin(element)
                axis = _vector(element.find("axis"), "xyz", (1, 0, 0))
                mode = element.attrib["type"]
                dof = None
                if mode in ("revolute", "continuous", "prismatic"):
                    limit = element.find("limit")
                    lo, hi = (-pi, pi) if mode == "continuous" else (
                        float(limit.attrib["lower"]), float(limit.attrib["upper"]))
                    speed = float(limit.attrib["velocity"])
                    norm = np.linalg.norm(axis)
                    if not np.isfinite([lo, hi, speed, norm]).all() or hi <= lo or min(speed, norm) <= 0:
                        raise ValueError(f"Invalid axis or limits for {element.attrib['name']}.")
                    axis /= norm
                    dof = len(limits)
                    limits.append((lo, hi))
                    speeds.append(speed)
                    continuous.append(mode == "continuous")
                elif mode != "fixed":
                    raise ValueError(f"Unsupported joint type {mode}.")
                if element.find("mimic") is not None:
                    raise ValueError("Mimic joints require an independent coordinate mapping.")
                joints.append(Joint(element.attrib["name"], parent, child, position, basis, axis, dof,
                                    mode == "prismatic"))

        links = []
        for name in names:
            element = elements[name]
            collisions, visuals = element.findall("collision"), element.findall("visual")
            mesh, box = _link_shape(element, path.parent, collisions or visuals)
            visual = _link_shape(element, path.parent, visuals)[0] if collisions and visuals else None
            if box:
                bounds = mesh.bounds
                axes = [np.linspace(lo, hi, max(2, int(np.ceil((hi - lo) / sample_pitch))) + 1)
                        for lo, hi in zip(*bounds)]
                points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
                geometry = Robot(mesh, points, float(np.linalg.norm(mesh.vertices, axis=1).max()))
            else:
                geometry = Robot.from_mesh(mesh, sample_pitch)
            links.append(Link(name, geometry, Obstacle(mesh, mesh.bounds if box else None), visual))

        posture_path = path.with_suffix(".poses.json")
        named = json.loads(posture_path.read_text(encoding="utf-8")) if posture_path.exists() else {}
        names_by_dof = [joint.name for joint in joints if joint.dof is not None]
        postures = {key: [angles[name] for name in names_by_dof] for key, angles in named.items()}
        return cls(links, joints, limits, speeds, continuous, postures)

    def valid(self, q):
        q = np.asarray(q)
        return (q.shape == self.speeds.shape and np.isfinite(q).all()
                and np.all(self.continuous | ((q >= self.limits[:, 0]) & (q <= self.limits[:, 1]))))

    def delta(self, start, end):
        delta = end - start
        return np.where(self.continuous, (delta + pi) % (2 * pi) - pi, delta)

    def clamp(self, q):
        return np.where(self.continuous, (q + pi) % (2 * pi) - pi, np.clip(q, *self.limits.T))

    def fk(self, q):
        """Return link-to-base rotation matrices and translations."""
        rotations = np.empty((len(self.links), 3, 3))
        translations = np.empty((len(self.links), 3))
        rotations[0], translations[0] = np.eye(3), 0.0
        for joint in self.joints:
            rotation, offset = joint.basis, joint.offset
            if joint.prismatic:
                offset = offset + q[joint.dof] * (joint.basis @ joint.axis)
            elif joint.dof is not None:
                rotation = rotation @ Rotation.from_rotvec(joint.axis * q[joint.dof]).as_matrix()
            parent = rotations[joint.parent]
            rotations[joint.child] = parent @ rotation
            translations[joint.child] = parent @ offset + translations[joint.parent]
        return rotations, translations

    def fk_torch(self, q):
        """Differentiable forward kinematics over an arbitrary batch of angles."""
        import torch

        def tensor(values):
            return torch.as_tensor(values, device=q.device, dtype=q.dtype)

        identity = torch.eye(3, device=q.device, dtype=q.dtype)
        rotations = [identity.expand(*q.shape[:-1], 3, 3)]
        translations = [q.new_zeros((*q.shape[:-1], 3))]
        for joint in self.joints:
            local = tensor(joint.basis)
            offset = tensor(joint.offset)
            if joint.prismatic:
                offset = offset + q[..., joint.dof, None] * tensor(joint.basis @ joint.axis)
            elif joint.dof is not None:
                x, y, z = joint.axis
                skew = tensor([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                angle = q[..., joint.dof, None, None]
                local = local @ (identity + angle.sin() * skew + (1 - angle.cos()) * (skew @ skew))
            parent = rotations[joint.parent]
            rotations.append(parent @ local)
            translations.append((parent @ offset[..., None])[..., 0] + translations[joint.parent])
        return torch.stack(rotations, dim=-3), torch.stack(translations, dim=-2)

    def frozen(self, q):
        rotations, translations = self.fk(q)
        parts, points = [], []
        for link, rotation, position in zip(self.links, rotations, translations):
            mesh = link.geometry.mesh.copy()
            mesh.vertices = mesh.vertices @ rotation.T + position
            parts.append(mesh)
            points.append(link.geometry.points @ rotation.T + position)
        mesh = trimesh.util.concatenate(parts)
        return Robot(mesh, np.vstack(points), float(np.linalg.norm(mesh.vertices, axis=1).max()))
