"""Search, refinement and certified fallback for map-based mobile manipulation."""

from .map_planner import MapPlanner
from .planner import PlanningError
from .trajectory import PlanarTrajectory


def plan_map_motion(planner, start, goal, drive, *, optimize=None, joint_acceleration=None,
                    free_orientation=False, position_tolerance=0., cancelled=lambda: False):
    """Return the waypoint plan, executable trajectory and its two clearances."""
    checker = planner.checker

    def search(solver):
        return solver.plan(start, goal, free_orientation=free_orientation,
                           position_tolerance=position_tolerance, cancelled=cancelled)

    try:
        plan = search(planner)
        trajectory, minimum = PlanarTrajectory.fit(
            plan.path, checker, drive, cancelled, geometry=planner.geometry,
            optimize=optimize, joint_acceleration=joint_acceleration)
    except PlanningError:
        if planner.geometry is None or cancelled():
            raise
        fallback = MapPlanner(checker, planner.seed, fixed_joints=planner.fixed_joints.nonzero()[0])
        plan = search(fallback)
        trajectory, minimum = PlanarTrajectory.fit(
            plan.path, checker, drive, cancelled, joint_acceleration=joint_acceleration)
    return plan, trajectory, minimum
