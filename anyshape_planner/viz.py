"""Local viser demo for rigid bodies and articulated floating-base robots."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from importlib.util import find_spec
from queue import Empty, SimpleQueue
from threading import Event
from time import perf_counter

import numpy as np
import viser
from scipy.spatial.transform import Rotation

from .articulated import ArticulatedRobot
from .articulated_collision import Configuration
from .articulated_planner import ArticulatedPlanner
from .collision import Pose
from .controller import Follower, TrackingError
from .corridor import Corridor, corridor
from .geometry import Robot
from .mobile import MOBILE_SCENE, MobileCorridor, mobile_corridor
from .planner import PlanningError, RRTConnect
from .scenes import clutter, window
from .trajectory import PlanarTrajectory, TrajectoryPlayback
from .viz_trajectory import JointTrajectoryPlots


def _move(handle, pose: Pose):
    handle.position = pose.position
    handle.wxyz = pose.rotation.as_quat(scalar_first=True)


class Demo:
    def __init__(self, server: viser.ViserServer, scene_name: str, robot: Robot,
                 seed: int, articulated: ArticulatedRobot | None = None):
        self.server = server
        self.robot = robot
        self.custom_articulated = articulated
        self.articulated = False
        self.mobile = False
        self.scenes = {"窗口穿越": window(robot)}
        self.events = SimpleQueue()
        self.workers = ThreadPoolExecutor(max_workers=1)
        self.seed = seed
        self.pending = None
        self.cancel = Event()
        self.following = None
        self.path = None
        self.trajectory = None
        self.path_handle = None
        self.obstacle_handles = []
        self.body_parts = []
        self.goal_parts = []
        self.auto_execute = False
        self.last_plan_summary = ""
        self.last_ui_update = 0.0

        server.scene.set_up_direction("+z")
        server.scene.world_axes.visible = False
        self.floor = server.scene.add_grid("/floor", width=7.1, height=3.7,
                                           cell_size=.25, section_size=1, cell_color=(100, 117, 139),
                                           section_color=(149, 165, 184), plane_opacity=.19,
                                           plane_color=(40, 56, 77), fade_distance=9)
        self.body = server.scene.add_frame("/robot", show_axes=False)
        self.target = server.scene.add_transform_controls("/target", scale=.53, depth_test=False)

        gui = server.gui
        gui.set_panel_label("3D ESDF 路径规划")
        gui.configure_theme(dark_mode=True, control_width="medium", show_logo=False,
                            show_share_button=False, brand_color=(80, 196, 168))
        self.chooser = gui.add_dropdown("场景", ("窗口穿越", "杂乱环境", "双臂收缩走廊", MOBILE_SCENE),
                                        initial_value=scene_name)
        self.instructions = gui.add_markdown("")
        self.free_orientation = gui.add_checkbox("目标姿态自由", False)
        self.tolerance = gui.add_number("位置容差（m）", 0.0, min=0.0, max=.3, step=.01)
        self.goal_posture = gui.add_dropdown("目标关节姿态", ("展开", "收缩"), initial_value="展开", visible=False)
        self.optimize = gui.add_checkbox("ESDF 轨迹优化", False, disabled=find_spec("torch") is None,
                                         hint="可选 PyTorch 优化，保留通过连续碰撞检查的改进。")
        plan_button = gui.add_button("规划", color="blue")
        self.execute_button = gui.add_button("执行", color="teal", disabled=True)
        go_button = gui.add_button("规划并执行", color="teal")
        reset_button = gui.add_button("重置机器人与目标")
        self.status = gui.add_markdown("")
        self.joint_plots = JointTrajectoryPlots(gui)

        self.chooser.on_update(lambda event: self.events.put(("scene", event.target.value)))
        plan_button.on_click(lambda _: self.events.put(("plan", False)))
        go_button.on_click(lambda _: self.events.put(("plan", True)))
        self.execute_button.on_click(lambda _: self.events.put(("execute", None)))
        reset_button.on_click(lambda _: self.events.put(("reset", None)))
        self.target.on_drag_end(lambda _: self.events.put(("target", None)))
        self.free_orientation.on_update(lambda _: self.events.put(("target", None)))
        self.tolerance.on_update(lambda _: self.events.put(("target", None)))
        self.optimize.on_update(lambda _: self.events.put(("target", None)))

        @self.goal_posture.on_update
        def on_goal_posture(event):
            if event.client_id is not None:
                self.events.put(("target", None))

        @server.on_client_connect
        def set_camera(client):
            self._camera(client)

        self._change_scene(scene_name)

    def _camera(self, client):
        if self.mobile:
            client.camera.position = (3.8, -6.7, 3.55)
            client.camera.look_at = (.69, .10, .38)
        elif self.articulated:
            client.camera.position = (3.65, -6.9, 4.6)
            client.camera.look_at = (.38, .10, -.10)
        else:
            client.camera.position = (4.2, -6.0, 3.45)
            client.camera.look_at = (.45, .12, -.13)
        client.camera.fov = np.deg2rad(38)

    def _mesh(self, name, mesh, color, **options):
        return self.server.scene.add_mesh_simple(name, mesh.vertices, mesh.faces, color=color,
                                                 flat_shading=not options.get("wireframe", False),
                                                 **options)

    def _report(self, title: str, details: str = ""):
        self.status.content = f"**{title}**\n\n{details}" if details else f"**{title}**"

    def _load_robot_parts(self):
        for handle in [*self.body_parts, *self.goal_parts]:
            handle.remove()
        if self.articulated:
            meshes = [(link.name, link.visual if link.visual is not None else link.geometry.mesh)
                      for link in self.scene.robot.links]
        else:
            meshes = [("body", self.robot.mesh)]
        self.body_parts, self.goal_parts = [], []
        for index, (name, mesh) in enumerate(meshes):
            color = (255, 168, 52) if index == 0 else (255, 208, 96)
            if self.mobile:
                if name == "base":
                    color = (60, 144, 190)
                elif name.startswith("wheel_"):
                    color = (91, 107, 124)
                else:
                    color = (255, 172, 53)
            self.body_parts.append(self._mesh(f"/robot/link_{index}", mesh, color))
            self.goal_parts.append(self._mesh(f"/target/link_{index}", mesh, (239, 93, 188),
                                              wireframe=True, opacity=.55))

    def _move_links(self, handles, q):
        rotations, translations = self.scene.robot.fk(q)
        for handle, rotation, position in zip(handles, rotations, translations):
            _move(handle, Pose.at(position, Rotation.from_matrix(rotation)))

    def _show_robot(self):
        _move(self.body, self.pose)
        if self.articulated:
            self._move_links(self.body_parts, self.pose.joints)

    def _goal_angles(self):
        key = "compact" if self.goal_posture.value == "收缩" else "nominal"
        return self.scene.robot.postures[key]

    def _target_pose(self):
        rotation = Rotation.from_quat(np.array(self.target.wxyz), scalar_first=True)
        pose = Pose.at(np.array(self.target.position), rotation)
        if self.mobile:
            pose = pose.on_plane(self.scene.base_height)
            _move(self.target, pose)
        return pose

    def _stop(self):
        self.cancel.set()
        self.pending = None
        self.following = None
        self.path = None
        self.trajectory = None
        self.joint_plots.clear()
        self.execute_button.disabled = True
        if self.path_handle is not None:
            self.path_handle.remove()
            self.path_handle = None

    def _reset(self):
        self._stop()
        self.pose = self.scene.start
        self._show_robot()
        _move(self.target, self.scene.goal)
        if self.articulated:
            self.goal_posture.value = "展开"
            self._move_links(self.goal_parts, self._goal_angles())
        details = ("蓝色是轮式底盘，橙色是机械臂，粉色线框是目标。" if self.mobile else
                   "橙色是机器人，粉色线框是目标位姿。")
        self._report("等待规划", details)

    def _change_scene(self, name: str):
        self._stop()
        self._report("正在构建 ESDF")
        if name not in self.scenes:
            if name == MOBILE_SCENE:
                self.scenes[name] = mobile_corridor()
            elif name == "双臂收缩走廊":
                self.scenes[name] = corridor(self.custom_articulated)
            else:
                self.scenes[name] = clutter(robot=self.robot)
        self.scene = self.scenes[name]
        self.checker = self.scene.checker
        self.articulated = isinstance(self.scene, Corridor)
        self.mobile = isinstance(self.scene, MobileCorridor)
        self.goal_posture.visible = self.articulated
        self.instructions.content = ("拖动手柄设定地面位置和偏航角。执行中松开手柄会重新规划。" if self.mobile else
                                     "拖动手柄设置目标位姿。执行中松开手柄，机器人会从当前位置重新规划。")
        self.floor.position = (0, 0, self.scene.base_height if self.mobile else self.scene.bounds[0, 2] - .02)
        self._load_robot_parts()
        for client in self.server.get_clients().values():
            self._camera(client)
        for handle in self.obstacle_handles:
            handle.remove()
        is_window = not self.articulated and self.scene.portal is not None
        opacity = ((.14, .40, .15) if self.mobile else (.16, .37, .32, .10)) if self.articulated else (
            (.46 if is_window else .88,) * len(self.scene.obstacles))
        self.obstacle_handles = [
            self._mesh(f"/obstacles/{i}", obstacle.mesh, (104, 137, 175),
                       opacity=opacity[i],
                       side="double", cast_shadow=not (is_window or self.articulated))
            for i, obstacle in enumerate(self.scene.obstacles)
        ]
        if self.articulated:
            x, y = self.scene.half_length, self.scene.half_width
            low, high = self.scene.opening_z
            rings = np.array([[[i, -y, low], [i, y, low], [i, y, high], [i, -y, high]] for i in (-x, x)])
            edges = np.stack((rings, np.roll(rings, -1, axis=1)), axis=2).reshape(-1, 2, 3)
            lines = np.concatenate((edges, rings.transpose(1, 0, 2)))
            self.obstacle_handles.append(self.server.scene.add_line_segments(
                "/obstacles/corridor_rim", lines, (114, 202, 250), thickness=2, thickness_units="screen"))
        elif is_window:
            corners = np.array([[.152, -.4, -.44], [.152, .4, -.44],
                                [.152, .4, .44], [.152, -.4, .44]])
            edge_lines = np.stack((corners, np.roll(corners, -1, axis=0)), axis=1)
            self.obstacle_handles.append(self.server.scene.add_line_segments(
                "/obstacles/window_rim", edge_lines, (116, 207, 254),
                thickness=2, thickness_units="screen"))
        self._reset()

    def _solve(self, planner, start, goal, free, tolerance, optimize, drive, cancelled):
        clock = perf_counter()
        plan = planner.plan(start, goal, free_orientation=free,
                            position_tolerance=tolerance, cancelled=cancelled)
        original = plan
        checker = planner.checker
        articulated = isinstance(planner, ArticulatedPlanner)
        margins = (checker.margin, checker.self_margin) if articulated else (checker.margin,)

        def dense(path):
            if articulated:
                return checker.dense_clearances(path)
            return (checker.dense_clearance(path, step=.005),)

        def clear(values):
            return all(distance > margin for distance, margin in zip(values, margins))

        refinement_status = ""
        minimum = None
        if optimize and not cancelled():
            if articulated:
                from .articulated_optimizer import refine_articulated as refine
            else:
                from .optimizer import refine
            result = refine(plan.path, checker, cancelled=cancelled)
            candidate_minimum = dense(result.path) if result.accepted else None
            if candidate_minimum is not None and clear(candidate_minimum):
                plan = replace(plan, path=result.path)
                minimum = candidate_minimum
                refinement_status = " · 优化通过"
            else:
                refinement_status = " · 保留原路径"
        trajectory = None
        if drive is not None:
            try:
                trajectory, minimum = PlanarTrajectory.fit(plan.path, checker, drive, cancelled)
            except PlanningError:
                if plan is original or cancelled():
                    raise
                plan = original
                refinement_status = " · 回退原路点"
                trajectory, minimum = PlanarTrajectory.fit(plan.path, checker, drive, cancelled)
        elif minimum is None:
            minimum = dense(plan.path)
        if not clear(minimum):
            raise PlanningError("Dense collision verification failed.")
        return plan, minimum, perf_counter() - clock, refinement_status, trajectory

    def _plan(self, execute: bool, retarget: bool = False):
        self._stop()
        self.cancel = Event()
        self.auto_execute = execute
        goal = self._target_pose()
        if self.articulated:
            planner = ArticulatedPlanner(self.checker, self.scene.bounds, self.scene.portals, self.seed,
                                         adaptive_posture=self.mobile)
            goal = Configuration(goal, self._goal_angles())
            detail = "SE(2) × 关节空间" if self.mobile else "SE(3) × 关节空间"
        else:
            guide = self.scene.sample_portal if self.scene.portal is not None else None
            planner = RRTConnect(self.checker, self.scene.bounds, self.seed, guide)
            detail = "SE(3) RRT-Connect"
        self._report("正在重新规划" if retarget else "正在规划", detail)
        self.pending = self.workers.submit(self._solve, planner, self.pose, goal,
                                           self.free_orientation.value, self.tolerance.value,
                                           self.optimize.value, self.scene.drive if self.mobile else None,
                                           self.cancel.is_set)

    def _clearance_text(self, environment, self_distance=None):
        if self_distance is None:
            return f"**{100 * environment:.2f} cm**"
        self_text = f"**{100 * self_distance:.2f} cm**" if np.isfinite(self_distance) else "无检测对"
        return f"环境 **{100 * environment:.2f} cm** · 自碰撞 {self_text}"

    def _show_plan(self, plan, minimum, seconds: float, refinement_status: str, trajectory):
        self.path = plan.path
        self.trajectory = trajectory
        positions = np.array([pose.position for pose in plan.path])
        timing = ""
        if trajectory is not None:
            times = np.linspace(0, trajectory.duration, 301)
            positions = np.column_stack((trajectory.spline(times)[:, :2], np.full(len(times), .018)))
            self.joint_plots.show(trajectory, times)
            mode = "C² 连续轨迹" if trajectory.tangent_scale > 0 else "C² 保守停点"
            timing = f"  \n{mode} · 执行 {trajectory.duration:.2f} s"
        lines = np.stack((positions[:-1], positions[1:]), axis=1)
        self.path_handle = self.server.scene.add_line_segments(
            "/path", lines, (60, 244, 181), thickness=4, thickness_units="screen")
        method = f" · {plan.method}" if self.articulated else ""
        self.last_plan_summary = (f"{len(plan.path)} 个路点 · {seconds:.2f} s{method}{refinement_status}  \n"
                                  f"5 mm 复检净空 {self._clearance_text(*minimum)}{timing}")
        if self.auto_execute:
            self._execute()
        else:
            self.execute_button.disabled = False
            self._report("路径已就绪", self.last_plan_summary)

    def _execute(self):
        if self.path is not None:
            self.execute_button.disabled = True
            self.following = (TrajectoryPlayback(self.trajectory) if self.trajectory is not None else
                              Follower(self.path, self.checker))
            self.peak_wheel_speed = 0.0
            self.path = None
            self.last_ui_update = 0.0

    def _tick(self, dt):
        if self.pending is not None and self.pending.done():
            future, self.pending = self.pending, None
            try:
                self._show_plan(*future.result())
            except PlanningError as error:
                self._report("规划未找到可执行路径", str(error))
        if self.following is None:
            return
        previous = self.following.samples[-1]
        try:
            sample = self.following.step(dt)
        except TrackingError as error:
            self.following = None
            self._report("执行已停止", str(error))
            return
        self.pose = sample.pose
        self._show_robot()
        wheels = ""
        if self.mobile:
            drive = self.scene.drive
            speeds = drive.between(previous.pose.base, sample.pose.base, sample.time - previous.time)
            self.peak_wheel_speed = max(self.peak_wheel_speed, np.abs(speeds).max())
            wheels = f"  \n峰值轮速 **{self.peak_wheel_speed:.2f} / {drive.wheel_speed_limit:.0f} rad/s**"
        if self.following.done:
            if self.trajectory is not None:
                self.joint_plots.set_time(sample.time, self.trajectory.duration)
            position, angle = self.following.terminal_error
            minimum = min(state.clearance for state in self.following.samples)
            minimum_self = min(state.self_clearance for state in self.following.samples) if self.articulated else None
            joints = ""
            if self.articulated:
                robot = self.scene.robot
                error = np.abs(robot.delta(self.pose.joints, self.following.goal.joints))
                degrees = np.rad2deg(error[~robot.prismatic].max(initial=0.0))
                joints = f"  \n旋转关节最大误差 **{degrees:.2f}°**"
                if robot.prismatic.any():
                    joints += f" · 直线关节 **{1000 * error[robot.prismatic].max():.2f} mm**"
            label = "执行净空" if self.trajectory is not None else "跟踪净空"
            self._report("已到达目标", f"{self.last_plan_summary}  \n"
                         f"{label} {self._clearance_text(minimum, minimum_self)}  \n"
                         f"终点误差 **{1000 * position:.2f} mm / {np.rad2deg(angle):.2f}°**{joints}{wheels}")
            self.following = None
        elif sample.time - self.last_ui_update >= .16:
            self.last_ui_update = sample.time
            if self.trajectory is not None:
                self.joint_plots.set_time(sample.time, self.trajectory.duration)
            clearance = self._clearance_text(sample.clearance, sample.self_clearance if self.articulated else None)
            angles = ""
            if self.articulated:
                values = [f"{value:.2f} m" if linear else f"{np.rad2deg(value):.0f}°"
                          for value, linear in zip(self.pose.joints, self.scene.robot.prismatic)]
                angles = "  \n关节 " + " / ".join(values)
            self._report("执行中", f"{self.last_plan_summary}  \n"
                         f"当前净空 {clearance} · {sample.time:.1f} s{angles}{wheels}")

    def run(self):
        dt = .025
        deadline = perf_counter()
        try:
            while True:
                try:
                    kind, value = self.events.get(timeout=max(0.0, deadline - perf_counter()))
                except Empty:
                    pass
                else:
                    if kind == "scene":
                        self._change_scene(value)
                    elif kind == "reset":
                        self._reset()
                    elif kind == "plan":
                        self._plan(value)
                    elif kind == "execute":
                        self._execute()
                    elif kind == "target":
                        self._target_pose()
                        if self.articulated:
                            self._move_links(self.goal_parts, self._goal_angles())
                        if self.following is not None or (self.pending is not None and self.auto_execute):
                            self._plan(True, retarget=True)
                        else:
                            self._stop()
                            self._report("目标已更新", "点击规划生成新路径。")
                now = perf_counter()
                if now >= deadline:
                    self._tick(dt)
                    deadline = now + dt
        finally:
            self.cancel.set()
            self.workers.shutdown(cancel_futures=True)
