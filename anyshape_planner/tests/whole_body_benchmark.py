"""Paired planning and morphology ablations for the map scene."""

import argparse
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from ..articulated_collision import Configuration
from ..collision import Pose
from ..geometry import Obstacle
from ..map_motion import plan_map_motion
from ..map_planner import MapPlanner
from ..map_scene import JOINT_ACCELERATION_LIMITS, LOW_FIRST_OPENINGS, map_scene
from ..planner import PlanningError
from ..trajectory import TrajectoryPlayback
from ..whole_body_optimizer import jerk_quadrature, refine_trajectory
from .mobile_benchmark import reference


METHODS = ("esdf_v1_5", "geometric", "base_only", "whole_body", "no_obstacle_planes")
CASES = (
    ("default", ((.84, .89), (.78, .77))),
    ("low_first", LOW_FIRST_OPENINGS),
    ("low_second", ((.86, .87), (.77, .751))),
    ("wide", ((1.40, 1.14), (1.37, 1.15))),
)
OFFSETS = ((0., 0., 0.), (-.12, -.17, 0.), (.10, .10, 0.), (-.08, .15, 0.))


def _jerk(trajectory, robot):
    times, weights = jerk_quadrature(trajectory.knots)
    units = np.r_[1., 1., robot.radius, robot.joint_radii]
    integrals = np.sum((trajectory.spline(times, nu=3) * units * weights[:, None])**2, axis=0)
    return float(integrals.sum()), float(integrals[3:].sum())


def _trajectory_metrics(scene, trajectory, minimum, check_fcl):
    robot = scene.robot
    times = np.linspace(0., trajectory.duration, 2401)
    q = trajectory.spline(times)
    velocity = trajectory.spline(times, nu=1)
    acceleration = trajectory.spline(times, nu=2)
    np.testing.assert_allclose(q[0, :2], trajectory.start.position[:2], rtol=0, atol=1e-12)
    np.testing.assert_allclose(q[-1, :2], trajectory.goal.position[:2], rtol=0, atol=1e-12)
    np.testing.assert_allclose(q[[0, -1], 2],
                               [trajectory.start.rotation.as_rotvec()[2], trajectory.goal.rotation.as_rotvec()[2]],
                               atol=1e-12)
    np.testing.assert_allclose(q[[0, -1], 3:], [trajectory.start.joints, trajectory.goal.joints], atol=1e-12)
    assert np.all(q[:, 3:] >= robot.limits[:, 0] - 1e-9)
    assert np.all(q[:, 3:] <= robot.limits[:, 1] + 1e-9)
    assert np.all(np.abs(velocity[:, 3:]) <= robot.speeds + 1e-9)
    assert np.all(np.abs(acceleration[:, 3:]) <= np.asarray(JOINT_ACCELERATION_LIMITS) + 1e-9)
    wheels = float(np.abs(trajectory.wheel_speeds(times)).max())
    assert wheels <= scene.drive.wheel_speed_limit
    knots = trajectory.knots[1:-1]
    for order in (0, 1, 2):
        np.testing.assert_allclose(trajectory.spline(np.nextafter(knots, -np.inf), nu=order),
                                   trajectory.spline(knots, nu=order), atol=1e-8)
    jerk, joint_jerk = _jerk(trajectory, robot)
    min_geometric = float(np.min([
        scene.trajectory_geometry.link_clearances(state, robot.fk(state.joints)).min()
        for state in map(trajectory.at, times[::12])
    ]))

    samples = 0
    if check_fcl:
        exact = reference(scene)
        brute = np.full(2, np.inf)
        for state in trajectory.dense_states():
            assert scene.checker.valid(state)
            assert exact.check(state) == (False, [])
            brute = np.minimum(brute, trajectory.clearances(state))
            samples += 1
        np.testing.assert_allclose(minimum, brute, rtol=0, atol=1e-10)
        playback = TrajectoryPlayback(trajectory)
        while not playback.done:
            playback.step(.025)
        assert playback.terminal_error == (0., 0.)

    crossings = []
    for axis, i in enumerate((2, 5)):
        box = scene.obstacles[i].box_bounds
        center = box.mean(axis=0)
        side = 1 - axis
        inside = (np.abs(q[:, axis] - center[axis]) < .08) & (np.abs(q[:, side] - center[side]) < .4)
        assert inside.any()
        crossing = np.median(q[inside], axis=0)
        crossings.append({"width_m": float(box[1, side] - box[0, side]), "height_m": float(box[0, 2]),
                          "lift_m": float(crossing[3]), "shoulder_deg": float(np.rad2deg(crossing[4])),
                          "elbow_deg": float(np.rad2deg(crossing[5])),
                          "yaw_deg": float(np.rad2deg(crossing[2]))})
    return {"trajectory_s": trajectory.duration, "jerk_integral": jerk, "joint_jerk_integral": joint_jerk,
            "interior_stops": int(np.count_nonzero(np.linalg.norm(trajectory.spline(knots, nu=1)[:, :2],
                                                                 axis=1) < .001)),
            "peak_wheel_rad_s": wheels, "peak_joint_acceleration": np.abs(acceleration[:, 3:]).max(axis=0).tolist(),
            "certified_min_environment_m": float(minimum[0]), "certified_min_self_m": float(minimum[1]),
            "box_clearance_m_at_201_common_time_samples": min_geometric,
            "fcl_samples_at_5mm": samples, "crossings": crossings,
            "refinement": getattr(trajectory, "optimization", None),
            "geometry_fallback": trajectory.geometry is None}


def _run_case(scene, start, goal, method, check_fcl):
    geometry = None if method == "esdf_v1_5" else scene.trajectory_geometry
    optimizer = None
    if method in ("base_only", "whole_body", "no_obstacle_planes"):
        optimizer = partial(refine_trajectory, deformation=method != "base_only",
                            activation=-np.inf if method == "no_obstacle_planes" else .055)
    planner = MapPlanner(scene.checker, geometry=geometry)
    started = perf_counter()
    try:
        plan, trajectory, minimum = plan_map_motion(
            planner, start, goal, scene.drive, optimize=optimizer, joint_acceleration=JOINT_ACCELERATION_LIMITS)
    except PlanningError as error:
        return {"success": False, "seconds": perf_counter() - started, "error": str(error)}
    elapsed = perf_counter() - started
    metrics = _trajectory_metrics(scene, trajectory, minimum, check_fcl)
    return {"success": True, "seconds": elapsed, "search": plan.search, **metrics,
            "geometry_fallback": geometry is not None and trajectory.geometry is None}


def _locked_and_sealed(scene):
    outcomes = {}
    for name, fixed in (("no_lift", (0,)), ("no_fold", (1, 2)), ("rigid_nominal", (0, 1, 2))):
        solver = MapPlanner(scene.checker, geometry=scene.trajectory_geometry, fixed_joints=fixed)
        try:
            plan_map_motion(solver, scene.start, scene.goal, scene.drive)
        except PlanningError as error:
            outcomes[name] = str(error)
        else:
            raise AssertionError(f"The default narrow map unexpectedly accepted {name}.")
    ceiling = scene.obstacles[5].box_bounds
    fill = Obstacle.box((*ceiling[0, :2], -.15), (*ceiling[1, :2], ceiling[1, 2]))
    sealed = replace(scene, obstacles=[*scene.obstacles, fill])
    try:
        plan_map_motion(MapPlanner(sealed.checker, geometry=sealed.trajectory_geometry),
                        sealed.start, sealed.goal, sealed.drive, optimize=refine_trajectory)
    except PlanningError as error:
        outcomes["sealed"] = str(error)
    else:
        raise AssertionError("The planner traversed a sealed partition.")
    return outcomes


def run(targets, repetitions, check_fcl):
    trials = []
    for case, openings in CASES:
        scene = map_scene(openings=openings)
        assert scene.trajectory_geometry is not None
        if case == "default":
            locked = _locked_and_sealed(scene)
        for index, offset in enumerate(OFFSETS[:targets]):
            goal = Configuration(Pose.at(scene.goal.position + np.asarray(offset), scene.goal.rotation),
                                 scene.goal.joints)
            for repetition in range(repetitions):
                order = METHODS if (index + repetition) % 2 == 0 else METHODS[::-1]
                for method in order:
                    result = _run_case(scene, scene.start, goal, method, check_fcl and method in
                                       ("esdf_v1_5", "whole_body") and repetition == 0)
                    trials.append({"scene": case, "target": index, "repeat": repetition, "method": method, **result})
            result = next(row for row in reversed(trials) if row["method"] == "whole_body")
            print(f"{case} target {index}: whole body {result['seconds']:.3f} s, success={result['success']}",
                  flush=True)
    default = next(row for row in trials if row["scene"] == "default" and row["target"] == 0
                   and row["method"] == "whole_body")
    assert default["success"] and default["interior_stops"] == 0 and default["refinement"]
    swapped = next(row for row in trials if row["scene"] == "low_first" and row["target"] == 0
                   and row["method"] == "whole_body")
    assert swapped["success"] and swapped["crossings"][0]["lift_m"] + .05 < swapped["crossings"][1]["lift_m"]
    assert default["crossings"][0]["lift_m"] > default["crossings"][1]["lift_m"] + .05

    summary = {}
    for method in METHODS:
        rows = [row for row in trials if row["method"] == method]
        successes = [row for row in rows if row["success"]]
        times = [row["seconds"] for row in rows]
        summary[method] = {"trials": len(rows), "successes": len(successes),
                           "p50_s": float(np.median(times)), "p95_s": float(np.quantile(times, .95)),
                           "refinement_accepted": sum(row["refinement"] is not None for row in successes),
                           "geometry_fallbacks": sum(row["geometry_fallback"] for row in successes),
                           "fcl_samples": sum(row["fcl_samples_at_5mm"] for row in successes)}
    return {"platform": platform.platform(), "python": platform.python_version(), "summary": summary,
            "locked_and_sealed": locked, "trials": trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=int, choices=range(1, len(OFFSETS) + 1), default=len(OFFSETS))
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--skip-fcl", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    report = run(args.targets, args.repetitions, not args.skip_fcl)
    print(json.dumps(report["summary"], indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
