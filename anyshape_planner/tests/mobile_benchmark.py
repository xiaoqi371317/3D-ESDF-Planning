"""Test grounded planning, retraction and arm folding in the low corridor."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import fcl
import numpy as np
from scipy.spatial.transform import Rotation

from ..articulated_collision import Configuration
from ..articulated_planner import ArticulatedPlanner
from ..collision import Pose
from ..controller import Follower
from ..mobile import MOBILE_MODEL, mobile_corridor
from ..planner import PlanningError
from .articulated_benchmark import FCLReference


def reference(scene):
    drive = scene.drive
    return FCLReference(scene, {link.name: fcl.Cylinder(drive.wheel_radius, link.geometry.mesh.extents[2])
                                for link in scene.robot.links if link.name.startswith("wheel_")})


def random_state(scene, rng):
    x, y = rng.uniform(scene.bounds[0, :2], scene.bounds[1, :2])
    pose = Pose.at((x, y, 0), Rotation.from_euler("z", rng.uniform(-np.pi, np.pi)))
    return Configuration(pose, rng.uniform(*scene.robot.limits.T))


def kinematics(scene):
    import torch
    from yourdfpy import URDF

    robot = scene.robot
    independent = URDF.load(str(MOBILE_MODEL), load_meshes=False)
    rng = np.random.default_rng(120)
    angles = rng.uniform(*robot.limits.T, size=(80, len(robot.speeds)))
    coordinates = torch.tensor(angles, dtype=torch.float64, requires_grad=True)
    rotations, translations = robot.fk_torch(coordinates)
    translations[..., -1, 2].sum().backward()
    np.testing.assert_allclose(coordinates.grad[:, robot.prismatic].numpy(), 1, atol=1e-12)
    names = [joint.name for joint in robot.joints if joint.dof is not None]
    maximum = 0.0
    for i, q in enumerate(angles):
        independent.update_cfg(dict(zip(names, q)))
        exact = np.array([independent.get_transform(link.name) for link in robot.links])
        r, p = robot.fk(q)
        maximum = max(maximum, np.max(np.abs(r - exact[:, :3, :3])), np.max(np.abs(p - exact[:, :3, 3])))
        np.testing.assert_allclose(rotations[i].detach().numpy(), r, atol=1e-12)
        np.testing.assert_allclose(translations[i].detach().numpy(), p, atol=1e-12)
    assert maximum < 1e-12
    bad_pose = Pose.at((0, 0, .01), Rotation.from_euler("x", .08))
    assert not scene.checker.valid(Configuration(bad_pose, robot.postures["compact"]))
    worst_ratio = 0.0
    for _ in range(400):
        edge = scene.checker.edge(random_state(scene, rng), random_state(scene, rng))
        a, b = sorted(rng.uniform(0, 1, 2))
        points = [scene.checker.world_points(edge.at(fraction)) for fraction in (a, b)]
        motion = max(np.linalg.norm(p - q, axis=1).max() for p, q in zip(*points))
        worst_ratio = max(worst_ratio, motion / ((b - a) * edge.bound))
    assert worst_ratio <= 1 + 1e-10
    return dict(fk_samples=len(angles), max_fk_error=float(maximum), displacement_trials=400,
                maximum_displacement_ratio=float(worst_ratio), off_plane_pose_rejected=True)


def collision(scene, exact):
    rng = np.random.default_rng(709)
    total = 10_000
    counts = np.zeros((2, 2), dtype=int)
    seconds = 0.0
    for _ in range(total):
        state = random_state(scene, rng)
        start = perf_counter()
        predicted = not scene.checker.is_free(state)
        seconds += perf_counter() - start
        environment, pairs = exact.check(state)
        counts[int(environment or bool(pairs)), int(predicted)] += 1
    free, false_positive, false_negative, hit = (int(x) for x in counts.ravel())
    result = dict(samples=total, agreement=(free + hit) / total, false_positive=false_positive,
                  false_negative=false_negative, poses_per_second=total / seconds)
    print(json.dumps({"collision": result}), flush=True)
    assert result["agreement"] >= .99 and false_negative / total <= .001, result
    return result


def ablations(scene, exact):
    robot = scene.robot
    nominal, compact = (robot.postures[name] for name in ("nominal", "compact"))
    lowered = np.where(robot.prismatic, compact, nominal)
    folded = np.where(robot.prismatic, nominal, compact)
    assert not scene.checker.edge_is_free(scene.start, scene.goal)
    shapes = [robot.frozen(q) for q in (nominal, lowered, folded, compact)]
    wide, down, folded_up, narrow = [shape.mesh for shape in shapes]
    # At x=0 every vertex lies within the corridor's length, for any yaw.
    assert all(np.linalg.norm(mesh.vertices[:, :2], axis=1).max() < scene.half_length
               for mesh in (wide, down, folded_up, narrow))
    # The base length and perpendicular span from the chassis side to the
    # straight arm tip cannot both project inside the corridor width.
    chassis = robot.links[0].geometry.mesh
    arm_span = wide.bounds[1, 1] - chassis.bounds[0, 1]
    width = 2 * scene.half_width
    projection_ratio = (width / chassis.extents[0])**2 + (width / arm_span)**2
    assert projection_ratio < 1
    assert folded_up.bounds[1, 2] > scene.opening_z[1] > narrow.bounds[1, 2]
    assert narrow.extents[1] < width and abs(narrow.bounds[0, 2]) < 1e-9
    for yaw in np.linspace(-np.pi, np.pi, 73, endpoint=False):
        base = Pose.at((0, 0, 0), Rotation.from_euler("z", yaw))
        for q in (nominal, lowered, folded):
            state = Configuration(base, q)
            assert exact.check(state)[0] and not scene.checker.is_free(state)
    center = Configuration(Pose.at((0, 0, 0)), compact)
    assert scene.checker.is_free(center) and exact.check(center) == (False, [])
    return dict(fully_locked_blocked=True, retraction_only_blocked=True, folding_only_blocked=True,
                corridor_width_m=width, corridor_height_m=scene.opening_z[1],
                deployed_width_m=float(wide.extents[1]), deployed_height_m=float(wide.bounds[1, 2]),
                folded_width_m=float(narrow.extents[1]), folded_height_m=float(narrow.bounds[1, 2]),
                retraction_m=float((nominal - compact)[robot.prismatic][0]), yaw_checks=73)


def inspect_path(scene, plan, exact):
    env_min = self_min = np.inf
    robot = scene.robot
    compact = robot.postures["compact"]
    inside = 0
    for state in scene.checker.dense_states(plan.path, .005):
        assert scene.checker.valid(state)
        fk = robot.fk(state.joints)
        environment = scene.checker.link_clearances(state, fk).min()
        self_distance = scene.checker.self_clearance(state.joints, fk)
        assert exact.check(state, fk) == (False, [])
        env_min, self_min = min(env_min, environment), min(self_min, self_distance)
        if abs(state.position[0]) <= scene.half_length:
            inside += 1
            delta = np.abs(robot.delta(state.joints, compact))
            assert delta[robot.prismatic].max() < .06 and delta[~robot.prismatic].max() < .45
    assert inside and env_min > scene.checker.margin and self_min > scene.checker.self_margin
    return dict(environment_clearance_m=float(env_min), self_clearance_m=float(self_min))


def plan_trials(scene, exact):
    records = []
    for seed in range(12):
        rng = np.random.default_rng(seed)

        def jitter(state):
            position = state.position + (*rng.uniform(-.018, .018, 2), 0)
            rotation = Rotation.from_euler("z", rng.uniform(-.035, .035)) * state.rotation
            return Configuration(Pose.at(position, rotation), state.joints)

        clock = perf_counter()
        try:
            plan = ArticulatedPlanner(scene.checker, scene.bounds, scene.portals, seed).plan(
                jitter(scene.start), jitter(scene.goal), max_iterations=420)
        except PlanningError as error:
            records.append(dict(seed=seed, error=str(error)))
            continue
        record = dict(seed=seed, seconds=perf_counter() - clock, iterations=plan.iterations,
                      **inspect_path(scene, plan, exact))
        records.append(record)
        print(json.dumps(record), flush=True)
    successes = [entry for entry in records if "error" not in entry]
    assert len(successes) == len(records), records
    # Put the compact exit closer to the ceiling so unfolding directly fails
    # and the local joint-and-base RRT must find the transition.
    portals = tuple(Pose.at((x, 0, 0)) for x in (-1.90, 1.90))
    clock = perf_counter()
    tight = ArticulatedPlanner(scene.checker, scene.bounds, portals, 0).plan(scene.start, scene.goal)
    tight_seconds = perf_counter() - clock
    assert tight.iterations > 0
    return dict(successes=len(successes), trials=len(records),
                median_s=float(np.median([r["seconds"] for r in successes])),
                min_environment_m=min(r["environment_clearance_m"] for r in successes),
                min_self_m=min(r["self_clearance_m"] for r in successes),
                tight_transition=dict(seconds=tight_seconds, iterations=tight.iterations,
                                      **inspect_path(scene, tight, exact)), seeds=records)


def tracking(scene, plan, exact):
    robot, drive = scene.robot, scene.drive
    follower = Follower(plan.path, scene.checker, drive=drive)
    for _ in range(2000):
        if follower.done:
            break
        follower.step()
    assert follower.done
    wheel_peak, lateral_peak = 0.0, 0.0
    for before, after in zip(follower.samples, follower.samples[1:]):
        dt = after.time - before.time
        a, b = before.pose, after.pose
        assert scene.checker.valid(b) and exact.check(b) == (False, [])
        assert np.all(np.abs(robot.delta(a.joints, b.joints)) / dt <= robot.speeds + 1e-8)
        wheel_peak = max(wheel_peak, np.abs(drive.between(a.base, b.base, dt)).max())
        local = a.rotation.apply((b.position - a.position) / dt, inverse=True)
        lateral_peak = max(lateral_peak, abs(local[1]))
    assert wheel_peak <= drive.wheel_speed_limit + 1e-8 and lateral_peak > .05
    position, yaw = follower.terminal_error
    errors = np.abs(robot.delta(follower.pose.joints, follower.goal.joints))
    assert position < .01 and yaw < np.deg2rad(1)
    return dict(wheel_speed_peak_rad_s=float(wheel_peak), wheel_speed_limit_rad_s=drive.wheel_speed_limit,
                lateral_speed_peak_m_s=float(lateral_peak), terminal_position_m=position,
                terminal_yaw_deg=float(np.rad2deg(yaw)), terminal_lift_m=float(errors[robot.prismatic].max()),
                terminal_arm_deg=float(np.rad2deg(errors[~robot.prismatic].max())),
                simulation_seconds=follower.time)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=("all", "collision"), default="all")
    args = parser.parse_args()
    scene = mobile_corridor()
    exact = reference(scene)
    results = {"collision": collision(scene, exact)}
    if args.stage == "all":
        results["kinematics"] = kinematics(scene)
        results["ablations"] = ablations(scene, exact)
        results["planning"] = plan_trials(scene, exact)
        path = ArticulatedPlanner(scene.checker, scene.bounds, scene.portals, 0).plan(scene.start, scene.goal)
        results["tracking"] = tracking(scene, path, exact)
    print(json.dumps({k: v for k, v in results.items() if k != "planning"}), flush=True)
    if args.output:
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
