"""Minimum sampled ESDF distance with conservative group pruning."""

import numpy as np
from numba import njit


_BLOCK = 4


class SampleGroups:
    def __init__(self, clouds):
        points, limits, links, centers, extents = [], [0], [], [], []

        def add(cloud, link):
            low, high = cloud.min(axis=0), cloud.max(axis=0)
            if len(cloud) > 64:
                axis = (high - low).argmax()
                order = np.argsort(cloud[:, axis])
                middle = len(order) // 2
                add(cloud[order[:middle]], link)
                add(cloud[order[middle:]], link)
                return
            points.append(cloud)
            limits.append(limits[-1] + len(cloud))
            links.append(link)
            centers.append((low + high) / 2)
            extents.append((high - low) / 2)

        for link, cloud in enumerate(clouds):
            add(cloud, link)
        self.points = np.concatenate(points)
        self.limits = np.asarray(limits)
        self.links = np.asarray(links)
        self.centers = np.asarray(centers)
        self.extents = np.asarray(extents)


@njit(cache=True, inline="always")
def _coordinates(point, rotation, offset, axis):
    return (rotation[axis, 0] * point[0] + rotation[axis, 1] * point[1]
            + rotation[axis, 2] * point[2] + offset[axis])


@njit(cache=True, inline="always")
def _cell(x, last):
    x = max(0., min(x, last))
    lo = int(x)
    return lo, min(lo + 1, last), x - lo


@njit(cache=True, inline="always")
def _lerp(a, b, t):
    return (1 - t) * a + t * b


@njit(cache=True, inline="always")
def _interpolate(grid, x, y, z):
    a, b, u = _cell(x, grid.shape[0] - 1)
    c, d, v = _cell(y, grid.shape[1] - 1)
    e, f, w = _cell(z, grid.shape[2] - 1)
    lo = _lerp(_lerp(grid[a, c, e], grid[a, c, f], w), _lerp(grid[a, d, e], grid[a, d, f], w), v)
    hi = _lerp(_lerp(grid[b, c, e], grid[b, c, f], w), _lerp(grid[b, d, e], grid[b, d, f], w), v)
    return _lerp(lo, hi, u)


@njit(cache=True, inline="always")
def _block_index(coord, last):
    return int(max(0., min(coord, last))) // _BLOCK


@njit(cache=True, nogil=True)
def _minimum(grid, blocks, origin, high, resolution, correction,
             points, limits, links, centers, extents, rotations, offsets, world):
    best = np.inf
    first = np.empty(3, dtype=np.int64)
    last = np.empty(3, dtype=np.int64)
    for group, link in enumerate(links):
        rotation, offset = rotations[link], offsets[link]
        boundary = np.inf
        for axis in range(3):
            center = _coordinates(centers[group], rotation, offset, axis)
            radius = (abs(rotation[axis, 0]) * extents[group, 0]
                      + abs(rotation[axis, 1]) * extents[group, 1]
                      + abs(rotation[axis, 2]) * extents[group, 2])
            low, top = center - radius, center + radius
            if not np.isfinite(low) or not np.isfinite(top):
                return np.nan
            if world:
                boundary = min(boundary, low - origin[axis], high[axis] - top)
            size = grid.shape[axis] - 1
            # Include both trilinear vertices on each axis. Expand the range
            # slightly to cover roundoff in transforming the bounding box.
            first[axis] = _block_index((low - origin[axis]) / resolution - 1e-9, size)
            last[axis] = _block_index(np.ceil((top - origin[axis]) / resolution + 1e-9), size)
        lower = np.inf
        for i in range(first[0], last[0] + 1):
            for j in range(first[1], last[1] + 1):
                for k in range(first[2], last[2] + 1):
                    lower = min(lower, blocks[i, j, k])
        check_boundary = world and boundary < best + 1e-12
        check_field = lower - correction < best + 1e-12
        if not (check_boundary or check_field):
            continue
        for i in range(limits[group], limits[group + 1]):
            x = _coordinates(points[i], rotation, offset, 0)
            y = _coordinates(points[i], rotation, offset, 1)
            z = _coordinates(points[i], rotation, offset, 2)
            if check_boundary:
                best = min(best, x - origin[0], y - origin[1], z - origin[2],
                           high[0] - x, high[1] - y, high[2] - z)
            if check_field:
                value = _interpolate(grid, (x - origin[0]) / resolution, (y - origin[1]) / resolution,
                                     (z - origin[2]) / resolution)
                best = min(best, value - correction)
    return best


class ESDFMinimum:
    def __init__(self, field):
        self.field = field
        self.high = field.high
        self.correction = field.correction
        blocks = field.values
        for axis, size in enumerate(field.values.shape):
            blocks = np.minimum.reduceat(blocks, np.arange(0, size, _BLOCK), axis=axis)
        self.blocks = np.ascontiguousarray(blocks)

    def distance(self, groups, rotations, offsets, *, world=True):
        return _minimum(self.field.values, self.blocks, self.field.origin, self.high, self.field.resolution,
                        self.correction, groups.points, groups.limits, groups.links, groups.centers,
                        groups.extents, rotations, offsets, world)
