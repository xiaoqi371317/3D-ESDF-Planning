"""Mecanum and joint tracking with an actuator delay and lateral slip."""

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..articulated_collision import Configuration
from ..collision import Pose
from ..controller import mecanum_tracking_command
from ..map_motion import plan_map_motion
from ..map_planner import MapPlanner
from ..map_scene import JOINT_ACCELERATION_LIMITS, LOW_FIRST_OPENINGS, map_scene
from ..whole_body_optimizer import refine_trajectory
from .mobile_benchmark import reference


def _simulate(scene, trajectory, slip, delay_steps, dt=.02):
    robot, drive = scene.robot, scene.drive
    pose = trajectory.start
    wheels = np.zeros(4)
    joint_velocity = np.zeros(len(robot.speeds))
    joint_acceleration = np.asarray(JOINT_ACCELERATION_LIMITS)
    command_queue = deque([(wheels, joint_velocity)] * delay_steps)
    exact = reference(scene)
    environment, self_clearance = np.inf, np.inf
    position_error, yaw_error = 0., 0.
    joint_error = np.zeros_like(joint_velocity)
    peak_wheels = 0.
    samples = 0

    for step in range(1, int(np.ceil((trajectory.duration + 1.) / dt)) + 1):
        time = step * dt
        command_queue.append(mecanum_tracking_command(trajectory, pose, time - dt))
        wheel_command, joint_command = command_queue.popleft()
        # A first-order actuator on each wheel. Integrate the body velocity
        # using the yaw at the middle of the timestep.
        wheels = .35 * wheels + .65 * wheel_command
        vx, vy, yaw_rate = drive.body_velocity(wheels)
        yaw = pose.rotation.as_rotvec()[2]
        mid = Rotation.from_euler("z", yaw + .5 * yaw_rate * dt)
        position = pose.position + dt * mid.apply((vx, vy * (1 - slip), 0.))
        joint_velocity += np.clip(joint_command - joint_velocity, -dt * joint_acceleration, dt * joint_acceleration)
        joints = robot.clamp(pose.joints + dt * joint_velocity)
        pose = Configuration(Pose.at(position, Rotation.from_euler("z", yaw + yaw_rate * dt)), joints)
        target = trajectory.at(time)
        position_error = max(position_error, np.linalg.norm(pose.position - target.position))
        yaw_error = max(yaw_error, abs(pose.base.relative_rotation(target.base)[2]))
        joint_error = np.maximum(joint_error, np.abs(robot.delta(pose.joints, target.joints)))
        peak_wheels = max(peak_wheels, np.abs(wheels).max())
        assert exact.check(pose) == (False, []), f"FCL collision at {time:.3f} s."
        env, own = trajectory.clearances(pose)
        environment, self_clearance = min(environment, env), min(self_clearance, own)
        samples += 1

    terminal_position = float(np.linalg.norm(pose.position - trajectory.goal.position))
    terminal_angle = abs(pose.base.relative_rotation(trajectory.goal.base)[2])
    terminal_joints = np.abs(robot.delta(pose.joints, trajectory.goal.joints))
    assert peak_wheels <= drive.wheel_speed_limit + 1e-9
    assert environment > 0. and self_clearance > 0.
    assert terminal_position < .002 and terminal_angle < .002 and terminal_joints.max() < .002
    return {"delay_s": delay_steps * dt, "lateral_slip": slip, "samples_at_50_hz": samples,
            "peak_position_error_m": float(position_error), "peak_yaw_error_deg": float(np.rad2deg(yaw_error)),
            "peak_joint_error": joint_error.tolist(), "peak_wheel_rad_s": float(peak_wheels),
            "min_environment_m": float(environment), "min_self_m": float(self_clearance),
            "terminal_position_m": terminal_position, "terminal_yaw_deg": float(np.rad2deg(terminal_angle)),
            "terminal_joint_error": terminal_joints.tolist()}


def run():
    results = {}
    for name, scene in (("default", map_scene()), ("low_first", map_scene(openings=LOW_FIRST_OPENINGS))):
        twist = np.array([.2, -.1, .3])
        np.testing.assert_allclose(scene.drive.body_velocity(scene.drive.wheel_speeds(*twist)), twist, atol=1e-14)
        _, trajectory, _ = plan_map_motion(
            MapPlanner(scene.checker, geometry=scene.trajectory_geometry), scene.start, scene.goal, scene.drive,
            optimize=refine_trajectory, joint_acceleration=JOINT_ACCELERATION_LIMITS)
        results[name] = [_simulate(scene, trajectory, slip, delay) for slip, delay in ((0., 0), (.05, 2), (.10, 3))]
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.dumps(run(), indent=2)
    print(data)
    if args.output:
        args.output.write_text(data + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
