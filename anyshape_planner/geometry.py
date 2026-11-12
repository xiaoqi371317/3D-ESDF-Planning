"""Mesh sampling and a signed Euclidean distance field on a regular grid."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt, map_coordinates


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    mesh = trimesh.load_scene(path).to_mesh()
    if mesh.is_empty or not np.isfinite(mesh.vertices).all():
        raise ValueError(f"Mesh has no valid triangles: {path}")
    return mesh


@dataclass
class Robot:
    mesh: trimesh.Trimesh
    points: np.ndarray
    radius: float

    @classmethod
    def from_mesh(cls, mesh: trimesh.Trimesh, pitch: float = 0.075, seed: int = 0):
        if not mesh.is_watertight:
            raise ValueError("The robot mesh must be watertight to sample its interior.")
        if pitch <= 0:
            raise ValueError("Robot sampling pitch must be positive.")
        count = max(400, int(np.ceil(3.0 * mesh.area / pitch**2)))
        surface, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
        edge_points = []
        for start, end in mesh.vertices[mesh.edges_unique]:
            segments = max(1, int(np.ceil(2 * np.linalg.norm(end - start) / pitch)))
            edge_points.append(np.linspace(start, end, segments + 1)[1:-1])
        # Start the lattice half a pitch inside the AABB. Filter it against the
        # solid so no point from an exterior voxel is attached to the robot.
        axes = [np.arange(lo + min(pitch, hi - lo) / 2, hi, pitch)
                for lo, hi in zip(*mesh.bounds)]
        lattice = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        interior = lattice[mesh.contains(lattice)]
        points = np.vstack((mesh.vertices, surface, interior, *edge_points))
        return cls(mesh, points, float(np.linalg.norm(mesh.vertices, axis=1).max()))


@dataclass
class Obstacle:
    mesh: trimesh.Trimesh
    box_bounds: np.ndarray | None = None

    @classmethod
    def box(cls, lo, hi):
        bounds = np.array((lo, hi), dtype=float)
        mesh = trimesh.creation.box(extents=bounds[1] - bounds[0])
        mesh.apply_translation(bounds.mean(axis=0))
        return cls(mesh, bounds)


@dataclass
class ESDF:
    values: np.ndarray
    origin: np.ndarray
    resolution: float
    lipschitz: float

    @classmethod
    def build(cls, bounds, resolution: float, obstacles: list[Obstacle]):
        origin, high = np.asarray(bounds, dtype=float)
        if resolution <= 0 or np.any(high <= origin):
            raise ValueError("Invalid ESDF bounds or resolution.")
        shape = np.ceil((high - origin) / resolution - 1e-9).astype(int) + 1
        occupied = np.zeros(tuple(shape), dtype=bool)
        for obstacle in obstacles:
            if obstacle.box_bounds is not None:
                # Occupy every grid cell with positive-volume box overlap.
                lo, hi = (obstacle.box_bounds - origin) / resolution
                first = np.maximum(0, np.floor(lo + 0.5 + 1e-9).astype(int))
                stop = np.minimum(shape, np.ceil(hi + 0.5 - 1e-9).astype(int))
                if np.all(first < stop):
                    occupied[tuple(slice(a, b) for a, b in zip(first, stop))] = True
                continue

            voxels = obstacle.mesh.voxelized(resolution / 2)
            if obstacle.mesh.is_watertight:
                voxels = voxels.fill()
            indices = np.floor((voxels.points - origin) / resolution + 0.5).astype(int)
            indices = indices[np.all((indices >= 0) & (indices < shape), axis=1)]
            occupied[tuple(indices.T)] = True

        if occupied.any():
            outside = distance_transform_edt(~occupied, sampling=resolution)
            inside = distance_transform_edt(occupied, sampling=resolution)
            outside -= inside
            values = outside.astype(np.float32)
        else:
            values = np.full(tuple(shape), np.linalg.norm(high - origin), dtype=np.float32)

        # Each partial derivative of a trilinear cell is a convex combination
        # of its parallel edge slopes. Include the slope of the world boundary.
        slopes = [float(np.abs(np.diff(values, axis=axis)).max()) / resolution
                  for axis in range(3)]
        return cls(values, origin, resolution, max(1.0, float(np.linalg.norm(slopes))))

    @property
    def high(self):
        return self.origin + (np.array(self.values.shape) - 1) * self.resolution

    @property
    def correction(self) -> float:
        return float(np.sqrt(3.0) * self.resolution / 2.0)

    def query(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        coords = (points - self.origin) / self.resolution
        distance = map_coordinates(self.values, coords.T, order=1, mode="nearest",
                                   output=np.float64) - self.correction
        boundary = np.minimum(points - self.origin, self.high - points).min(axis=1)
        return np.minimum(distance, boundary)
