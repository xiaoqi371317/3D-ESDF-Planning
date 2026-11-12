"""Verify the map-only planner in the two-opening turn scene."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ..geometry import Obstacle
from ..map_planner import MapPlanner
from ..map_scene import map_scene
from ..planner import PlanningError
from ..trajectory import PlanarTrajectory, TrajectoryPlayback
from .mobile_benchmark import reference


def run():
    scene = map_scene()
    field = scene.checker
    solver = MapPlanner(field)
    clock = perf_counter()
    plan = solver.plan(scene.start.base, scene.goal.base)
    planning_seconds = perf_counter() - clock
    assert plan.guide is not None and plan.search["checked_edges"] > 0
    print(f"Map search accepted {len(plan.path)} knots in {planning_seconds:.2f} s.", flush=True)
    clock = perf_counter()
    trajectory, clearance = PlanarTrajectory.fit(plan.path, field, scene.drive)
    smoothing_seconds = perf_counter() - clock
    assert trajectory.tangent_scale > 0

    fcl = reference(scene)
    samples = 0
    for state in trajectory.dense_states(.005):
        assert field.valid(state) and fcl.check(state) == (False, [])
        samples += 1

    times = np.linspace(0, trajectory.duration, 2401)
    q = trajectory.spline(times)
    velocity = trajectory.spline(times, nu=1)
    internal = trajectory.knots[1:-1]
    for order in (1, 2):
        np.testing.assert_allclose(trajectory.spline(np.nextafter(internal, -np.inf), nu=order),
                                   trajectory.spline(internal, nu=order), atol=1e-8)
    assert np.all(np.abs(velocity[:, 3:]) <= scene.robot.speeds + 1e-9)
    wheels = float(np.abs(trajectory.wheel_speeds(times)).max())
    assert wheels < scene.drive.wheel_speed_limit

    # The test reads the two obstacle slabs after planning to measure the
    # crossing poses. Neither opening is an argument to MapPlanner.
    openings, records = [scene.obstacles[i].box_bounds for i in (2, 5)], []
    for axis, box in enumerate(openings):
        center = box.mean(axis=0)
        side = 1 - axis
        crossing = (np.abs(q[:, axis] - center[axis]) < .10) & (np.abs(q[:, side] - center[side]) < .32)
        assert np.any(crossing)
        pose = np.median(q[crossing], axis=0)
        records.append({"width_m": float(box[1, side] - box[0, side]), "height_m": float(box[0, 2]),
                        "base_xy_m": pose[:2].tolist(), "yaw_deg": float(np.rad2deg(pose[2])),
                        "lift_m": float(pose[3]), "shoulder_deg": float(np.rad2deg(pose[4])),
                        "elbow_deg": float(np.rad2deg(pose[5]))})
    turn = float(abs((records[1]["yaw_deg"] - records[0]["yaw_deg"] + 180) % 360 - 180))
    assert turn >= 70 and records[0]["lift_m"] - records[1]["lift_m"] >= .04

    playback = TrajectoryPlayback(trajectory)
    while not playback.done:
        playback.step(.025)
    assert playback.terminal_error == (0., 0.)

    ceiling = openings[1]
    closed = Obstacle.box((*ceiling[0, :2], -.15), (*ceiling[1, :2], ceiling[1, 2]))
    blocked = replace(scene, obstacles=[*scene.obstacles, closed])
    try:
        MapPlanner(blocked.checker).plan(blocked.start, blocked.goal)
    except PlanningError as error:
        assert "没有连通" in str(error)
    else:
        raise AssertionError("Planner crossed a sealed partition.")

    speeds = np.linalg.norm(trajectory.spline(internal, nu=1)[:, :2], axis=1)
    return {"planning_seconds": planning_seconds, "smoothing_seconds": smoothing_seconds,
            "search": plan.search, "trajectory_knots": len(plan.path), "trajectory_seconds": trajectory.duration,
            "tangent_scale": trajectory.tangent_scale, "base_turn_deg": turn, "openings": records,
            "interior_knots_with_motion": int(np.count_nonzero(speeds > .001)),
            "interior_knots": len(speeds), "min_environment_m": float(clearance[0]),
            "min_self_m": float(clearance[1]), "fcl_samples_at_5mm": samples,
            "peak_wheel_rad_s": wheels, "wheel_limit_rad_s": scene.drive.wheel_speed_limit,
            "continuous_velocity_and_acceleration": True, "sealed_opening_rejected": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.dumps(run(), indent=2)
    print(data)
    if args.output:
        args.output.write_text(data, encoding="utf-8")


if __name__ == "__main__":
    main()
