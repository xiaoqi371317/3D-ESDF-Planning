"""Reproducible acceptance benchmarks. Run with python -m anyshape_planner.tests.benchmark."""

import argparse
import json
import tempfile
from math import ceil
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import fcl
import numpy as np
from scipy.spatial.transform import Rotation

from ..collision import CollisionChecker, Edge, Pose
from ..controller import Follower
from ..geometry import ESDF, Obstacle, Robot, load_mesh
from ..planner import PlanningError, RRTConnect
from ..scenes import clutter, l_mesh, window


class FCLWindow:
    """Compare the demo L as two solid boxes, including full containment."""

    def __init__(self, scene):
        self.bounds = scene.bounds
        self.vertices = scene.robot.mesh.vertices
        self.request = fcl.CollisionRequest()
        self.walls = []
        for obstacle in scene.obstacles:
            lo, hi = obstacle.box_bounds
            self.walls.append(fcl.CollisionObject(fcl.Box(*(hi - lo)),
                                                  fcl.Transform((lo + hi) / 2)))
        self.parts = [(np.array((0, 0, -.16)), fcl.CollisionObject(fcl.Box(.22, 1.34, .24))),
                      (np.array((0, -.55, .12)), fcl.CollisionObject(fcl.Box(.22, .24, .32)))]

    def collides(self, pose: Pose) -> bool:
        vertices = pose.rotation.apply(self.vertices) + pose.position
        if np.any(vertices <= self.bounds[0]) or np.any(vertices >= self.bounds[1]):
            return True
        rotation = pose.rotation.as_matrix()
        for center, part in self.parts:
            part.setTransform(fcl.Transform(rotation, rotation @ center + pose.position))
            if any(fcl.collide(part, wall, self.request, fcl.CollisionResult())
                   for wall in self.walls):
                return True
        return False


def collision_benchmark(scene) -> dict:
    checker = scene.checker
    exact = FCLWindow(scene)
    rng = np.random.default_rng(2026)
    positions = rng.uniform(*scene.bounds, size=(10_000, 3))
    rotations = Rotation.random(len(positions), random_state=rng)
    counts = dict(agree_free=0, agree_collision=0, false_positive=0, false_negative=0)
    for position, rotation in zip(positions, rotations):
        pose = Pose.at(position, rotation)
        actual = exact.collides(pose)
        predicted = not checker.is_free(pose)
        key = {(True, True): "agree_collision", (False, False): "agree_free",
               (False, True): "false_positive", (True, False): "false_negative"}
        counts[key[actual, predicted]] += 1
    accuracy = (counts["agree_free"] + counts["agree_collision"]) / len(positions)
    print(json.dumps({"agreement": accuracy, **counts}), flush=True)
    assert accuracy >= .99 and counts["false_negative"] == 0, counts

    assert checker.is_free(scene.start) and checker.is_free(scene.goal)
    assert not checker.edge_is_free(scene.start, scene.goal)
    assert not checker.is_free(Pose.at((3.0, 0, 0)))
    assert not checker.is_free(Pose.at((0, 0, 0))) and checker.is_free(scene.portal)
    # A tiny box inside the L, with no surface intersections.
    obstacle = Obstacle.box((-.038, -.28, -.192), (.038, -.16, -.128))
    embedded = CollisionChecker(scene.robot, ESDF.build([[-1, -1, -1], [1, 1, 1]],
                                                       .02, [obstacle]))
    assert not embedded.is_free(Pose.at((0, 0, 0)))
    with tempfile.TemporaryDirectory() as directory:
        for extension in ("obj", "stl"):
            path = Path(directory) / f"robot.{extension}"
            l_mesh().export(path)
            assert Robot.from_mesh(load_mesh(path)).radius > .65
    return {"poses": len(positions), "agreement": accuracy, **counts,
            "fixed_orientation_crossing_blocked": True, "embedded_obstacle_detected": True,
            "out_of_bounds_blocked": True, "obj_and_stl_round_trip": True}


def make_planner(scene, seed):
    return RRTConnect(scene.checker, scene.bounds, seed,
                      scene.sample_portal if scene.portal is not None else None)


def verify_with_fcl(scene, path, exact):
    for start, end in zip(path, path[1:]):
        edge = Edge(start, end, scene.robot.radius)
        count = max(1, ceil(edge.bound / .005))
        assert all(not exact.collides(edge.at(i / count)) for i in range(count + 1))


def planning_benchmark(scene, trials: int, exact: FCLWindow | None = None) -> dict:
    records = []
    for seed in range(trials):
        planner = make_planner(scene, seed)
        start = perf_counter()
        try:
            plan = planner.plan(scene.start, scene.goal)
        except PlanningError as error:
            records.append(dict(seed=seed, error=str(error), seconds=perf_counter() - start))
            continue
        seconds = perf_counter() - start
        minimum = scene.checker.dense_clearance(plan.path, step=.005)
        assert minimum > scene.checker.margin
        record = dict(seed=seed, seconds=seconds, iterations=plan.iterations,
                      waypoints=len(plan.path), dense_clearance_m=minimum)
        if exact is not None:
            verify_with_fcl(scene, plan.path, exact)
            record["peak_rotation_deg"] = float(np.rad2deg(max(
                np.linalg.norm(scene.start.relative_rotation(pose)) for pose in plan.path)))
            assert record["peak_rotation_deg"] > 15
        records.append(record)
        print(json.dumps({"scene": scene.name, **record}, ensure_ascii=True), flush=True)
    successes = [record for record in records if "error" not in record]
    return {"trials": trials, "successes": len(successes), "seed_results": records,
            "median_planning_s": float(np.median([r["seconds"] for r in records])),
            "min_dense_clearance_m": min((r["dense_clearance_m"] for r in successes), default=None)}


def goal_modes(scene) -> dict:
    goal = Pose.at(scene.portal.position)
    planner = make_planner(scene, 31)
    assert not scene.checker.is_free(goal)
    plan = planner.plan(scene.start, goal, free_orientation=True, position_tolerance=.05)
    assert np.linalg.norm(plan.path[-1].position - goal.position) <= .05 + 1e-10
    assert scene.checker.dense_clearance(plan.path) > scene.checker.margin
    goal = Pose.at((scene.bounds[0, 0] + .09, 0, 0))
    assert not scene.checker.is_free(goal)
    plan = planner.plan(scene.start, goal, position_tolerance=.14)
    assert np.linalg.norm(plan.path[-1].position - goal.position) <= .14 + 1e-10
    assert np.linalg.norm(plan.path[-1].relative_rotation(goal)) < 1e-10
    return {"free_orientation": True, "position_tolerance_m": .14}


def finish_tracking(follower: Follower) -> dict:
    for _ in range(3000):
        if follower.done:
            break
        follower.step()
    assert follower.done
    position, angle = follower.terminal_error
    angle_deg = float(np.rad2deg(angle))
    minimum = min(sample.clearance for sample in follower.samples)
    assert position < .01 and angle_deg < 1 and minimum > follower.checker.margin
    for a, b in zip(follower.samples, follower.samples[1:]):
        dt = b.time - a.time
        assert np.linalg.norm(b.pose.position - a.pose.position) / dt <= follower.linear_limit + 1e-9
        assert np.linalg.norm(a.pose.relative_rotation(b.pose)) / dt <= follower.angular_limit + 1e-9
    return {"terminal_position_m": position, "terminal_angle_deg": angle_deg,
            "minimum_clearance_m": minimum, "simulation_time_s": follower.time,
            "samples": len(follower.samples)}


def tracking_benchmark(scene) -> dict:
    records = []
    for seed in (0, 4, 9):
        plan = make_planner(scene, seed).plan(scene.start, scene.goal)
        records.append({"seed": seed, **finish_tracking(Follower(plan.path, scene.checker))})
    plan = make_planner(scene, 0).plan(scene.start, scene.goal)
    follower = Follower(plan.path, scene.checker)
    while follower.pose.position[0] < -.55:
        follower.step()
        assert not follower.done
    start = follower.pose
    new_goal = Pose.at((1.84, -.36, .31), Rotation.from_euler("x", -25, degrees=True))
    clock = perf_counter()
    new_plan = make_planner(scene, 33).plan(start, new_goal)
    seconds = perf_counter() - clock
    assert new_plan.path[0] is start
    assert scene.checker.dense_clearance(new_plan.path) > scene.checker.margin
    retarget = {"planning_s": seconds, **finish_tracking(Follower(new_plan.path, scene.checker))}
    result = {"fixed_goal": records, "retarget": retarget}
    print(json.dumps(result), flush=True)
    return result


def optimizer_benchmark(scene) -> dict:
    from ..optimizer import refine

    path = make_planner(scene, 0).plan(scene.start, scene.goal).path
    clock = perf_counter()
    result = refine(path, scene.checker)
    seconds = perf_counter() - clock
    assert result.accepted and result.final_cost < result.initial_cost
    assert result.path[0] is path[0] and result.path[-1] is path[-1]
    minimum = scene.checker.dense_clearance(result.path)
    assert minimum > scene.checker.margin
    verify_with_fcl(scene, result.path, FCLWindow(scene))
    with patch.object(scene.checker, "edge_is_free", return_value=False) as reject:
        fallback = refine(path, scene.checker, iterations=10)
    assert reject.called and not fallback.accepted and fallback.path is path
    summary = {"initial_cost": result.initial_cost, "final_cost": result.final_cost,
               "refinement_s": seconds, "dense_clearance_m": minimum,
               "unsafe_candidate_rejected": True,
               "tracking": finish_tracking(Follower(result.path, scene.checker))}
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "core", "collision", "plan", "control", "optimizer"],
                        default="core", help="Run core benchmarks or include PyTorch with 'all'.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    scene = window()
    results = {}
    if args.stage in ("all", "core", "collision"):
        results["collision"] = collision_benchmark(scene)
    if args.stage in ("all", "core", "plan"):
        results["window"] = planning_benchmark(scene, 20, FCLWindow(scene))
        assert results["window"]["successes"] >= 18
        results["goal_modes"] = goal_modes(scene)
        results["clutter"] = planning_benchmark(clutter(), 10)
        assert results["clutter"]["successes"] > 0
    if args.stage in ("all", "core", "control"):
        results["tracking"] = tracking_benchmark(scene)
    if args.stage in ("all", "optimizer"):
        results["optimizer"] = optimizer_benchmark(scene)
    if args.output:
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
