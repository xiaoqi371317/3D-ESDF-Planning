"""Acceptance benchmarks for the floating-base dual-arm demo."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch
import xml.etree.ElementTree as ET

import fcl
import numpy as np
from scipy.spatial.transform import Rotation

from ..articulated import ArticulatedRobot
from ..articulated_collision import ArticulatedEdge, Configuration
from ..articulated_planner import ArticulatedPlanner
from ..collision import CollisionChecker, Pose
from ..controller import Follower
from ..corridor import MODEL, corridor
from ..planner import PlanningError, RRTConnect


def _random_state(scene, rng):
    return Configuration(Pose.at(rng.uniform(*scene.bounds), Rotation.random(random_state=rng)),
                         rng.uniform(*scene.robot.limits.T))


def kinematics(scene):
    import torch
    from yourdfpy import URDF

    robot = scene.robot
    independent = URDF.load(str(MODEL), load_meshes=False)
    rng = np.random.default_rng(13)
    samples = rng.uniform(*robot.limits.T, size=(100, len(robot.speeds)))
    joints = [joint.name for joint in robot.joints if joint.dof is not None]
    angles = torch.tensor(samples, dtype=torch.float64, requires_grad=True)
    torch_rot, torch_pos = robot.fk_torch(angles)
    (torch_rot.square().sum() + torch_pos[..., 1].sum()).backward()
    assert torch.isfinite(angles.grad).all() and angles.grad.abs().max() > 0
    maximum = 0.0
    for index, q in enumerate(samples):
        independent.update_cfg(dict(zip(joints, q)))
        matrices = np.array([independent.get_transform(link.name) for link in robot.links])
        rotations, translations = robot.fk(q)
        maximum = max(maximum, np.max(np.abs(matrices[:, :3, :3] - rotations)),
                      np.max(np.abs(matrices[:, :3, 3] - translations)))
        np.testing.assert_allclose(torch_rot[index].detach().numpy(), rotations, atol=1e-12)
        np.testing.assert_allclose(torch_pos[index].detach().numpy(), translations, atol=1e-12)
    assert maximum < 1e-12
    with TemporaryDirectory() as directory:
        path = Path(directory) / "test.urdf"
        tree = ET.parse(MODEL)
        geometry = tree.find("./link/collision/geometry")
        geometry.remove(geometry[0])
        ET.SubElement(geometry, "mesh", {"filename": "torso.stl", "scale": "1 1 1"})
        robot.links[0].geometry.mesh.export(path.with_name("torso.stl"))
        tree.find("./joint").set("type", "continuous")
        tree.write(path)
        test_model = ArticulatedRobot.from_urdf(path)
        np.testing.assert_allclose(test_model.links[0].geometry.mesh.bounds,
                                   robot.links[0].geometry.mesh.bounds, atol=1e-7)
        start, end = (Configuration(scene.start.base, np.array([value, 0.0, 0.0, 0.0]))
                      for value in (np.pi - .02, -np.pi + .02))
        np.testing.assert_allclose(ArticulatedEdge(start, end, test_model).joint_delta,
                                   (.04, 0, 0, 0), atol=1e-12)
    nominal, compact = (robot.frozen(robot.postures[name]) for name in ("nominal", "compact"))
    assert compact.mesh.extents[1] < 2 * scene.half_width < nominal.mesh.extents[1]
    return dict(samples=len(samples), maximum_fk_error=float(maximum), torch_gradients=True,
                link_count=len(robot.links), degrees_of_freedom=6 + len(joints),
                compact_width_m=float(compact.mesh.extents[1]), nominal_width_m=float(nominal.mesh.extents[1]),
                external_mesh_and_continuous_wrap=True)


class FCLReference:
    def __init__(self, scene):
        self.scene = scene
        self.robot = scene.robot
        self.request = fcl.CollisionRequest()
        self.links = [fcl.CollisionObject(fcl.Box(*link.geometry.mesh.extents))
                      for link in self.robot.links]
        self.walls = []
        for obstacle in scene.obstacles:
            lo, hi = obstacle.box_bounds
            self.walls.append(fcl.CollisionObject(fcl.Box(*(hi - lo)), fcl.Transform((lo + hi) / 2)))

    def collide(self, a, b):
        return bool(fcl.collide(a, b, self.request, fcl.CollisionResult()))

    def check(self, state, fk=None, include_self=True):
        local_r, local_t = self.robot.fk(state.joints) if fk is None else fk
        rotations = state.rotation.as_matrix() @ local_r
        centers = state.rotation.apply(local_t) + state.position
        centers += (rotations @ self.robot.centers[..., None])[..., 0]
        half = (np.abs(rotations) @ self.robot.half_extents[..., None])[..., 0]
        for link, rotation, center in zip(self.links, rotations, centers):
            link.setTransform(fcl.Transform(rotation, center))
        out = np.any(centers - half <= self.scene.bounds[0]) or np.any(centers + half >= self.scene.bounds[1])
        environment = out or any(self.collide(link, wall) for link in self.links for wall in self.walls)
        self_pairs = [pair for pair in self.scene.checker.pairs
                      if self.collide(self.links[pair[0]], self.links[pair[1]])] if include_self else []
        return bool(environment), self_pairs


def collision(scene, include_self):
    rng = np.random.default_rng(2601)
    checker = scene.checker
    exact = FCLReference(scene)
    trials = 10_000
    env_seconds = self_seconds = 0.0
    kinds = ("environment", "self", "combined") if include_self else ("environment",)
    counts = {name: np.zeros((2, 2), dtype=int) for name in kinds}
    pair_counts = dict.fromkeys(checker.pairs, 0)
    for _ in range(trials):
        state = _random_state(scene, rng)
        clock = perf_counter()
        fk = scene.robot.fk(state.joints)
        estimated_env = checker.link_clearances(state, fk).min() <= checker.margin
        env_seconds += perf_counter() - clock
        exact_env, colliding = exact.check(state, fk, include_self)
        counts["environment"][int(exact_env), int(estimated_env)] += 1
        if include_self:
            clock = perf_counter()
            estimated_self = checker.self_clearance(state.joints, fk) <= checker.self_margin
            self_seconds += perf_counter() - clock
            counts["self"][int(bool(colliding)), int(estimated_self)] += 1
            counts["combined"][int(exact_env or bool(colliding)), int(estimated_env or estimated_self)] += 1
            for pair in colliding:
                pair_counts[pair] += 1
    results = {}
    for name, counts_for_kind in counts.items():
        free, false_positive, false_negative, hit = (int(i) for i in counts_for_kind.ravel())
        agreement = (free + hit) / trials
        elapsed = {"environment": env_seconds, "self": self_seconds,
                   "combined": env_seconds + self_seconds}[name]
        results[name] = dict(poses=trials, agreement=agreement, agree_free=free, agree_collision=hit,
                             false_positive=false_positive, false_negative=false_negative,
                             poses_per_second=trials / elapsed)
    if include_self:
        results["acm"] = dict(ignored_adjacent=len(checker.allowed), checked_pairs=len(checker.pairs),
                              persistent_non_adjacent=[list(pair) for pair, n in pair_counts.items()
                                                       if n >= .995 * trials])
        assert results["self"]["agree_collision"] > 0
        assert not results["acm"]["persistent_non_adjacent"]
    print(json.dumps(results), flush=True)
    for name in kinds:
        record = results[name]
        assert record["agreement"] >= .99 and record["false_negative"] / trials <= .001, record
    return results


def motion_bounds(scene):
    checker = scene.checker
    rng = np.random.default_rng(4210)
    maximum = maximum_self = 0.0
    trials = 1000
    for _ in range(trials):
        edge = checker.edge(_random_state(scene, rng), _random_state(scene, rng))
        a, b = sorted(rng.uniform(0, 1, size=2))
        points_a, points_b = (checker.world_points(edge.at(fraction)) for fraction in (a, b))
        actual = max(np.linalg.norm(p - q, axis=1).max() for p, q in zip(points_a, points_b))
        bound = (b - a) * edge.bound
        assert actual <= bound + 1e-10
        maximum = max(maximum, actual / bound)
        q1, q2 = (edge.at(fraction).joints for fraction in (a, b))
        fk1, fk2 = (scene.robot.fk(q) for q in (q1, q2))
        bounds = (b - a) * (checker.relative_rates @ np.abs(edge.joint_delta))
        for (i, j), rate in zip(checker.pairs, bounds):
            change = abs(checker._pair_distance(i, j, *fk1) - checker._pair_distance(i, j, *fk2))
            assert change <= rate + 1e-10
            if rate > 0:
                maximum_self = max(maximum_self, change / rate)
    compact = scene.robot.postures["compact"]
    assert checker.edge_is_free(scene.start, Configuration(scene.start.base, compact))
    assert checker.is_free(scene.start) and checker.is_free(scene.goal)
    assert not checker.edge_is_free(scene.start, scene.goal)
    return dict(trials=trials, maximum_point_motion_ratio=maximum, maximum_pair_field_ratio=maximum_self,
                colliding_corridor_edge_rejected=True)


def make_planner(scene, seed=0):
    return ArticulatedPlanner(scene.checker, scene.bounds, scene.portals, seed)


def locked_nominal(scene):
    nominal = scene.robot.frozen(scene.robot.postures["nominal"])
    diagonal = np.hypot(2 * scene.half_width, 2 * scene.half_height)
    body_length = scene.robot.links[0].geometry.mesh.extents[0]
    arm_span = nominal.mesh.extents[1]
    # At x=0 all the vertices lie within the tunnel's length. The projections
    # of two orthogonal segments onto its cross section cannot both be small
    # enough when the sum of these squared ratios is below one.
    ratio = (diagonal / body_length)**2 + (diagonal / arm_span)**2
    assert nominal.radius < scene.half_length and ratio < 1
    rigid = CollisionChecker(nominal, scene.checker.field, scene.checker.margin)
    try:
        RRTConnect(rigid, scene.bounds, seed=0).plan(scene.start.base, scene.goal.base, max_iterations=120)
    except PlanningError:
        return dict(impossible=True, projection_ratio=float(ratio), rrt_iterations=120)
    raise AssertionError("Nominal-only planner crossed the corridor.")


def verify_path(scene, path, exact, measure_posture=True):
    checker = scene.checker
    compact, nominal = (scene.robot.postures[name] for name in ("compact", "nominal"))
    env_min = self_min = np.inf
    inside, outside = [], []
    for state in checker.dense_states(path, step=.005):
        assert scene.robot.valid(state.joints)
        fk = scene.robot.fk(state.joints)
        env_min = min(env_min, checker.link_clearances(state, fk).min())
        self_min = min(self_min, checker.self_clearance(state.joints, fk))
        env_collision, self_collision = exact.check(state, fk)
        assert not env_collision and not self_collision
        x = abs(state.position[0])
        if x <= scene.half_length:
            inside.append(np.linalg.norm(scene.robot.delta(state.joints, compact)))
        elif x >= scene.half_length + scene.robot.radius:
            outside.append(np.linalg.norm(scene.robot.delta(state.joints, nominal)))
    assert env_min > checker.margin and self_min > checker.self_margin
    result = dict(environment_clearance_m=float(env_min), self_clearance_m=float(self_min))
    if measure_posture:
        assert inside and outside
        compact_error, nominal_error = float(np.max(inside)), float(np.mean(outside))
        assert compact_error < .30 and nominal_error < .75, (compact_error, nominal_error)
        result.update(inside_compact_max_rad=compact_error, outside_nominal_mean_rad=nominal_error)
    return result


def planning(scene):
    exact = FCLReference(scene)
    ablation = locked_nominal(scene)
    records = []
    for seed in range(20):
        rng = np.random.default_rng(seed)

        def jitter(endpoint):
            position = endpoint.position + rng.uniform((-.018, -.023, -.02), (.018, .023, .02))
            rotation = Rotation.from_euler("z", rng.uniform(-.06, .06))
            return Configuration(Pose.at(position, rotation), endpoint.joints)

        start, goal = jitter(scene.start), jitter(scene.goal)
        solver = make_planner(scene, seed)
        clock = perf_counter()
        try:
            plan = solver.plan(start, goal)
        except PlanningError as error:
            records.append(dict(seed=seed, error=str(error), seconds=perf_counter() - clock))
            continue
        record = dict(seed=seed, seconds=perf_counter() - clock, iterations=plan.iterations,
                      waypoints=len(plan.path), method=plan.method)
        record.update(verify_path(scene, plan.path, exact))
        records.append(record)
        print(json.dumps(record), flush=True)

    successes = [record for record in records if "error" not in record]
    assert len(successes) >= 18, records
    clock = perf_counter()
    full = make_planner(scene, 0).plan(scene.start, scene.goal, force_full=True)
    full_seconds = perf_counter() - clock
    assert full.iterations > 0
    full_record = dict(iterations=full.iterations, seconds=full_seconds, waypoints=len(full.path),
                       **verify_path(scene, full.path, exact, measure_posture=False))
    return dict(trials=20, successes=len(successes), locked_nominal=ablation,
                median_planning_s=float(np.median([r["seconds"] for r in successes])),
                min_environment_m=min(r["environment_clearance_m"] for r in successes),
                min_self_m=min(r["self_clearance_m"] for r in successes),
                full_search=full_record, seed_results=records)


def optimization(scene):
    from ..articulated_optimizer import refine_articulated

    path = make_planner(scene).plan(scene.start, scene.goal).path
    clock = perf_counter()
    result = refine_articulated(path, scene.checker)
    seconds = perf_counter() - clock
    assert result.accepted and result.final_cost < result.initial_cost
    assert result.path[0] is path[0] and result.path[-1] is path[-1]
    measurements = verify_path(scene, result.path, FCLReference(scene))
    with patch.object(scene.checker, "edge_is_free", return_value=False) as reject:
        fallback = refine_articulated(path, scene.checker, iterations=10)
    assert reject.called and fallback.path is path and not fallback.accepted
    return dict(initial_cost=result.initial_cost, final_cost=result.final_cost, seconds=seconds,
                waypoints=len(result.path), unsafe_candidate_rejected=True, **measurements)


def finish_tracking(scene, follower):
    for _ in range(3000):
        if follower.done:
            break
        follower.step()
    assert follower.done
    robot = scene.robot
    exact = FCLReference(scene)
    linear_max = angular_max = 0.0
    joint_max = np.zeros_like(robot.speeds)
    for sample in follower.samples:
        assert robot.valid(sample.pose.joints)
        assert sample.clearance > scene.checker.margin and sample.self_clearance > scene.checker.self_margin
        environment, pairs = exact.check(sample.pose)
        assert not environment and not pairs
    for a, b in zip(follower.samples, follower.samples[1:]):
        dt = b.time - a.time
        linear_max = max(linear_max, np.linalg.norm(b.pose.position - a.pose.position) / dt)
        angular_max = max(angular_max, np.linalg.norm(a.pose.base.relative_rotation(b.pose.base)) / dt)
        joint_max = np.maximum(joint_max, np.abs(robot.delta(a.pose.joints, b.pose.joints)) / dt)
    assert linear_max <= follower.linear_limit + 1e-9 and angular_max <= follower.angular_limit + 1e-9
    assert np.all(joint_max <= robot.speeds + 1e-9)
    position, angle = follower.terminal_error
    assert position < .01 and angle < np.deg2rad(1) and follower.terminal_joint_error < np.deg2rad(1)
    return dict(position_error_m=position, rotation_error_deg=float(np.rad2deg(angle)),
                max_joint_error_deg=float(np.rad2deg(follower.terminal_joint_error)),
                peak_joint_speed_rad_s=joint_max.tolist(), peak_linear_speed_m_s=float(linear_max),
                peak_angular_speed_rad_s=float(angular_max), steps=len(follower.samples),
                simulation_seconds=follower.time,
                min_environment_m=min(s.clearance for s in follower.samples),
                min_self_m=min(s.self_clearance for s in follower.samples))


def control(scene):
    original = make_planner(scene).plan(scene.start, scene.goal)
    nominal = finish_tracking(scene, Follower(original.path, scene.checker))
    follower = Follower(original.path, scene.checker)
    while follower.pose.position[0] < -.4:
        follower.step()
        assert not follower.done
    goal = Configuration(Pose.at((2.42, -.15, .08), Rotation.from_euler("z", -12, degrees=True)),
                         scene.goal.joints)
    clock = perf_counter()
    plan = make_planner(scene, 5).plan(follower.pose, goal)
    seconds = perf_counter() - clock
    assert plan.path[0] is follower.pose
    retarget = finish_tracking(scene, Follower(plan.path, scene.checker))
    return dict(original=nominal, retarget={"planning_seconds": seconds, **retarget})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "fk", "environment", "collision", "bounds", "plan", "optimizer", "control"),
                        default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    scene = corridor()
    stages = {"fk": kinematics, "environment": lambda s: collision(s, False),
              "collision": lambda s: collision(s, True), "bounds": motion_bounds, "plan": planning,
              "optimizer": optimization, "control": control}
    results = {}
    for name in ("fk", "collision", "bounds", "plan", "optimizer", "control") if args.stage == "all" else (args.stage,):
        results[name] = stages[name](scene)
        print(json.dumps({"stage": name, "result": results[name]}), flush=True)
    if args.output:
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
