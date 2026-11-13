"""Local spline refinement with sparse obstacle planes pulled back by link Jacobians."""

from copy import copy

import numpy as np
from scipy.interpolate import BPoly
from scipy.optimize import minimize

from .planner import PlanningError


def jerk_quadrature(knots):
    """Three Gauss points integrate squared jerk of a quintic exactly."""
    u, weights = np.polynomial.legendre.leggauss(3)
    durations = np.diff(knots)
    times = knots[:-1, None] + durations[:, None] * (u + 1) / 2
    return times.ravel(), np.sqrt(durations[:, None] * weights / 2).ravel()


def refine_trajectory(seed, cancelled, *, deformation=True, activation=.055):
    """Optimize a fixed-time quintic, then leave acceptance to the certificate."""
    if cancelled():
        raise PlanningError("Planning cancelled.")
    if seed.geometry is None or len(seed.knots) <= 2:
        return None
    knots = seed.knots
    states = seed.spline(knots)
    velocities = seed.spline(knots, nu=1)
    robot, geometry = seed.checker.robot, seed.geometry
    durations = np.diff(knots)
    dofs = states.shape[1]
    controls = len(states) - 2
    weight = np.r_[1., 1., robot.radius, robot.joint_radii]

    # Both the state and its shared knot velocity enter the quintic linearly.
    # Fix the endpoint position, velocity and acceleration. Three Gauss
    # points integrate squared jerk exactly on each quintic piece.
    sample_times, factor = jerk_quadrature(knots)
    identity = np.pad(np.eye(controls), ((1, 1), (0, 0)))
    zeros = np.zeros_like(identity)
    position_basis = seed.controls(identity, zeros, durations)
    velocity_basis = seed.controls(zeros, identity, durations)
    basis = BPoly(np.concatenate((position_basis, velocity_basis), axis=2).swapaxes(0, 1), knots)
    jerk_basis = basis(sample_times, nu=3) * factor[:, None]
    jerk = seed.spline(sample_times, nu=3) * factor[:, None]
    jerk_weight = weight / max(np.linalg.norm(jerk * weight), 1e-12)
    hessian = jerk_basis.T @ jerk_basis
    linear = jerk_basis.T @ (jerk * jerk_weight**2)

    # Use trust bounds in physical units. For bounded joints, constrain both
    # the knot and the nearest Bernstein controls within the joint limits.
    offset = np.broadcast_to(.045 / weight, (controls, dofs)).copy()
    velocity_span = np.broadcast_to(.35 / weight, (controls, dofs)).copy()
    low = -np.vstack((offset, velocity_span))
    high = -low.copy()
    bounded = np.flatnonzero(~robot.continuous)
    limits = robot.limits[bounded]
    indices = bounded + 3
    time_offsets = np.column_stack((-2 * durations[:-1] / 5, np.zeros(controls), 2 * durations[1:] / 5))
    positions = states[1:-1, indices][:, None, :]
    joint_velocities = velocities[1:-1, indices][:, None, :]
    joint_controls = positions + time_offsets[:, :, None] * joint_velocities
    slack = np.maximum(0., np.minimum(joint_controls - limits[:, 0], limits[:, 1] - joint_controls)).min(axis=1)
    # Split available slack between offsets and tangent changes.
    room = slack / 2
    low[:controls, indices] = np.maximum(low[:controls, indices], -room)
    high[:controls, indices] = np.minimum(high[:controls, indices], room)
    velocity_room = room / np.abs(time_offsets).max(axis=1)[:, None]
    low[controls:, indices] = np.maximum(low[controls:, indices], -velocity_room)
    high[controls:, indices] = np.minimum(high[controls:, indices], velocity_room)
    if not deformation:
        low[:, 3:] = high[:, 3:] = 0.

    # Anchor only near obstacles along the current seed. The local model
    # linearizes box support along the chosen normal, with an exact continuous
    # collision check after the optimizer has proposed a candidate.
    times = (knots[:-1, None] + durations[:, None] * np.array([.25, .5, .75])).ravel()
    shapes = basis(times)
    rows, slacks = [], []
    if activation > 0.:
        for time, shape in zip(times, shapes):
            if cancelled():
                raise PlanningError("Planning cancelled.")
            for gap, gradient in geometry.planes(seed.at(time), activation):
                rows.append(np.outer(shape, gradient).ravel())
                # Reserve space for tracking error at close approaches.
                slacks.append(gap - max(.025, seed.environment_margin + .003))
    planes = np.asarray(rows).reshape(-1, 2 * controls * dofs)
    slacks = np.asarray(slacks)
    diagonal = np.broadcast_to(weight, low.shape) ** 2
    regularizer = 10. * diagonal
    plane_weight = 100.
    # Bias the mechanism toward its nominal shape wherever the local
    # separation planes permit. Constant fold angles have zero jerk.
    joint_preference = np.zeros_like(low)
    joint_preference[:controls, 3:] = robot.delta(robot.postures["nominal"], states[1:-1, 3:])
    posture_weight = np.zeros_like(low)
    posture_weight[:controls, 3:] = .25 * weight[3:]**2 / controls
    slope = linear + posture_weight * joint_preference

    def objective(flat):
        if cancelled():
            raise PlanningError("Planning cancelled.")
        value = flat.reshape(low.shape)
        smooth = (hessian @ value) * jerk_weight**2
        quadratic = smooth + (regularizer + posture_weight) * value
        violations = np.minimum(slacks + planes @ flat, 0.)
        cost = .5 * np.sum(value * quadratic) + np.sum(slope * value)
        cost += .5 * plane_weight * (violations @ violations)
        gradient = (quadratic + slope).ravel() + plane_weight * (planes.T @ violations)
        return cost, gradient

    zero = np.zeros(low.size)
    initial_cost = objective(zero)[0]
    solution = minimize(objective, zero, jac=True, method="L-BFGS-B",
                        bounds=list(zip(low.ravel(), high.ravel())),
                        options={"maxiter": 24, "maxfun": 40, "maxls": 8, "ftol": 1e-7})
    if not np.isfinite(solution.x).all() or not np.isfinite(solution.fun) or solution.fun >= initial_cost - 1e-5:
        return None
    shifts, tangents = np.split(solution.x.reshape(low.shape), 2)
    refined = states.copy()
    refined[1:-1] += shifts
    velocities[1:-1] += tangents
    candidate = copy(seed)
    candidate._set_polynomial(seed.controls(refined, velocities, durations), durations)
    candidate.optimization = {"active_planes": len(planes), "evaluations": solution.nfev,
                              "iterations": solution.nit}
    return candidate
