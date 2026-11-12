"""Verify geometry-dependent posture selection and executed spline motion."""

import argparse
import json
from pathlib import Path

import numpy as np

from ..articulated_planner import ArticulatedPlanner
from ..mobile import mobile_corridor
from ..trajectory import PlanarTrajectory, TrajectoryPlayback
from .mobile_benchmark import reference


def plan(scene):
    return ArticulatedPlanner(scene.checker, scene.bounds, scene.portals, 0,
                              adaptive_posture=True).plan(scene.start, scene.goal)


def posture_and_spline():
    scene = mobile_corridor()
    planned = plan(scene)
    posture = planned.passage_joints
    assert posture is not None and planned.method == "ESDF 搜索通行姿态"
    assert np.linalg.norm(posture - scene.robot.postures["compact"]) > .1
    assert posture[0] < scene.robot.postures["nominal"][0] - .05
    trajectory, minimum = PlanarTrajectory.fit(planned.path, scene.checker, scene.drive)
    assert trajectory.tangent_scale > 0
    exact = reference(scene)
    for state in trajectory.dense_states():
        assert scene.checker.valid(state) and exact.check(state) == (False, [])

    times = np.linspace(0, trajectory.duration, 1601)
    velocities = trajectory.spline(times, nu=1)
    np.testing.assert_allclose(velocities[[0, -1]], 0, atol=1e-10)
    assert np.all(np.abs(velocities[:, 3:]) <= scene.robot.speeds + 1e-9)
    assert np.abs(trajectory.wheel_speeds(times)).max() < scene.drive.wheel_speed_limit
    assert np.linalg.norm(velocities[:, :2], axis=1).max() < 1.0
    internal = trajectory.knots[1:-1]
    for derivative in (1, 2):
        before = trajectory.spline(np.nextafter(internal, -np.inf), nu=derivative)
        after = trajectory.spline(internal, nu=derivative)
        np.testing.assert_allclose(before, after, atol=1e-9)
    interior_speed = np.linalg.norm(trajectory.spline(internal, nu=1)[:, :2], axis=1)
    assert np.min(interior_speed) > .20
    # Measure the difference between the executed curve and the waypoint chords.
    linear_y = np.interp(trajectory.spline(times)[:, 0], [p.position[0] for p in planned.path],
                         [p.position[1] for p in planned.path])
    deviation = np.abs(trajectory.spline(times)[:, 1] - linear_y).max()
    assert deviation > .01

    follower = TrajectoryPlayback(trajectory)
    wheel_peak = 0.0
    while not follower.done:
        before = follower.samples[-1]
        after = follower.step(.025)
        wheel_peak = max(wheel_peak, np.abs(scene.drive.between(before.pose.base, after.pose.base,
                                                               after.time - before.time)).max())
    assert wheel_peak < scene.drive.wheel_speed_limit and follower.terminal_error == (0., 0.)

    low = mobile_corridor(height=.77)
    low_plan = plan(low)
    assert low_plan.passage_joints is not None and low_plan.method == "ESDF 搜索通行姿态"
    assert low_plan.passage_joints[0] < posture[0] - .035
    low_trajectory, _ = PlanarTrajectory.fit(low_plan.path, low.checker, low.drive)
    low_exact = reference(low)
    assert all(low_exact.check(state) == (False, []) for state in low_trajectory.dense_states())

    wide = mobile_corridor(width=2.64, height=1.17)
    wide_plan = plan(wide)
    assert wide_plan.method == "直接连接"
    assert all(np.array_equal(state.joints, wide.robot.postures["nominal"]) for state in wide_plan.path)
    return dict(default_passage_joints_m_rad=posture.tolist(), low_passage_joints_m_rad=low_plan.passage_joints.tolist(),
                wide_opening_keeps_nominal=True, trajectory_seconds=trajectory.duration,
                interior_base_speeds_m_s=interior_speed.tolist(), max_curve_deviation_m=float(deviation),
                minimum_environment_m=float(minimum[0]), minimum_self_m=float(minimum[1]),
                peak_executed_wheel_rad_s=float(wheel_peak), continuous_velocity_and_acceleration=True,
                fcl_5mm_both_openings=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(posture_and_spline(), indent=2)
    print(report)
    if args.output:
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
