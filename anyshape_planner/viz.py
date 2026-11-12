"""Local viser demo with a draggable SE(3) goal and live replanning."""

from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec
from queue import Empty, SimpleQueue
from threading import Event
from time import perf_counter

import numpy as np
import viser
from scipy.spatial.transform import Rotation

from .collision import Pose
from .controller import Follower, TrackingError
from .geometry import Robot
from .planner import Plan, PlanningError, RRTConnect
from .scenes import clutter, window


def _move(handle, pose: Pose):
    handle.position = pose.position
    handle.wxyz = pose.rotation.as_quat(scalar_first=True)


class Demo:
    def __init__(self, server: viser.ViserServer, scene_name: str, robot: Robot,
                 seed: int):
        self.server = server
        self.robot = robot
        self.scenes = {"窗口穿越": window(robot)}
        self.events = SimpleQueue()
        self.workers = ThreadPoolExecutor(max_workers=1)
        self.seed = seed
        self.pending = None
        self.cancel = Event()
        self.following = None
        self.path = None
        self.path_handle = None
        self.obstacle_handles = []
        self.auto_execute = False
        self.last_plan_summary = ""
        self.last_ui_update = 0.0

        server.scene.set_up_direction("+z")
        server.scene.world_axes.visible = False
        server.scene.add_grid("/floor", width=6.2, height=3.7, position=(0, 0, -1.62),
                              cell_size=.25, section_size=1, cell_color=(100, 117, 139),
                              section_color=(149, 165, 184), plane_opacity=.19,
                              plane_color=(40, 56, 77), fade_distance=9)
        self.body = self._mesh("/robot", self.robot.mesh, (255, 178, 49))
        self.target = server.scene.add_transform_controls("/target", scale=.53, depth_test=False)
        self._mesh("/target/preview", self.robot.mesh, (239, 93, 188), wireframe=True, opacity=.55)

        gui = server.gui
        gui.set_panel_label("6-DoF 刚体路径规划")
        gui.configure_theme(dark_mode=True, control_width="medium", show_logo=False,
                            show_share_button=False, brand_color=(80, 196, 168))
        self.chooser = gui.add_dropdown("场景", ("窗口穿越", "杂乱环境"), initial_value=scene_name)
        gui.add_markdown("拖动手柄设置目标位姿。执行中松开手柄，机器人会从当前位置重新规划。")
        self.free_orientation = gui.add_checkbox("目标姿态自由", False)
        self.tolerance = gui.add_number("位置容差（m）", 0.0, min=0.0, max=.3, step=.01)
        self.optimize = gui.add_checkbox("ESDF 轨迹优化", False, disabled=find_spec("torch") is None,
                                         hint="可选 PyTorch 优化，保留通过连续碰撞检查的改进。")
        plan_button = gui.add_button("规划", color="blue")
        self.execute_button = gui.add_button("执行", color="teal", disabled=True)
        go_button = gui.add_button("规划并执行", color="teal")
        reset_button = gui.add_button("重置机器人与目标")
        self.status = gui.add_markdown("")

        self.chooser.on_update(lambda event: self.events.put(("scene", event.target.value)))
        plan_button.on_click(lambda _: self.events.put(("plan", False)))
        go_button.on_click(lambda _: self.events.put(("plan", True)))
        self.execute_button.on_click(lambda _: self.events.put(("execute", None)))
        reset_button.on_click(lambda _: self.events.put(("reset", None)))
        self.target.on_drag_end(lambda _: self.events.put(("target", None)))
        self.free_orientation.on_update(lambda _: self.events.put(("target", None)))
        self.tolerance.on_update(lambda _: self.events.put(("target", None)))
        self.optimize.on_update(lambda _: self.events.put(("target", None)))

        @server.on_client_connect
        def set_camera(client):
            client.camera.position = (4.2, -6.0, 3.45)
            client.camera.look_at = (.45, .12, -.13)
            client.camera.fov = np.deg2rad(38)

        self._change_scene(scene_name)

    def _mesh(self, name, mesh, color, **options):
        return self.server.scene.add_mesh_simple(name, mesh.vertices, mesh.faces, color=color,
                                                 flat_shading=not options.get("wireframe", False),
                                                 **options)

    def _report(self, title: str, details: str = ""):
        self.status.content = f"**{title}**\n\n{details}" if details else f"**{title}**"

    def _stop(self):
        self.cancel.set()
        self.pending = None
        self.following = None
        self.path = None
        self.execute_button.disabled = True
        if self.path_handle is not None:
            self.path_handle.remove()
            self.path_handle = None

    def _reset(self):
        self._stop()
        self.pose = self.scene.start
        _move(self.body, self.pose)
        _move(self.target, self.scene.goal)
        self._report("等待规划", "橙色是机器人，粉色线框是目标位姿。")

    def _change_scene(self, name: str):
        self._stop()
        self._report("正在构建 ESDF")
        if name not in self.scenes:
            self.scenes[name] = clutter(robot=self.robot)
        self.scene = self.scenes[name]
        self.checker = self.scene.checker
        for handle in self.obstacle_handles:
            handle.remove()
        is_window = self.scene.portal is not None
        self.obstacle_handles = [
            self._mesh(f"/obstacles/{i}", obstacle.mesh, (104, 137, 175),
                       opacity=.46 if is_window else .88, side="double",
                       cast_shadow=not is_window)
            for i, obstacle in enumerate(self.scene.obstacles)
        ]
        if is_window:
            corners = np.array([[.152, -.4, -.44], [.152, .4, -.44],
                                [.152, .4, .44], [.152, -.4, .44]])
            edge_lines = np.stack((corners, np.roll(corners, -1, axis=0)), axis=1)
            self.obstacle_handles.append(self.server.scene.add_line_segments(
                "/obstacles/window_rim", edge_lines, (116, 207, 254),
                thickness=2, thickness_units="screen"))
        self._reset()

    def _solve(self, planner, start, goal, free, tolerance, optimize, cancelled):
        clock = perf_counter()
        plan = planner.plan(start, goal, free_orientation=free,
                            position_tolerance=tolerance, cancelled=cancelled)
        refinement_status = ""
        minimum = None
        if optimize and not cancelled():
            from .optimizer import refine

            result = refine(plan.path, planner.checker, cancelled=cancelled)
            candidate_minimum = (planner.checker.dense_clearance(result.path, step=.005)
                                 if result.accepted else -np.inf)
            if candidate_minimum > planner.checker.margin:
                plan = Plan(result.path, plan.iterations)
                minimum = candidate_minimum
                refinement_status = " · 优化通过"
            else:
                refinement_status = " · 保留原路径"
        if minimum is None:
            minimum = planner.checker.dense_clearance(plan.path, step=.005)
        if minimum <= planner.checker.margin:
            raise PlanningError("Dense collision verification failed.")
        return plan, minimum, perf_counter() - clock, refinement_status

    def _plan(self, execute: bool, retarget: bool = False):
        self._stop()
        self.cancel = Event()
        self.auto_execute = execute
        rotation = Rotation.from_quat(np.array(self.target.wxyz), scalar_first=True)
        goal = Pose.at(np.array(self.target.position), rotation)
        guide = self.scene.sample_portal if self.scene.portal is not None else None
        planner = RRTConnect(self.checker, self.scene.bounds, self.seed, guide)
        self._report("正在重新规划" if retarget else "正在规划", "SE(3) RRT-Connect")
        self.pending = self.workers.submit(self._solve, planner, self.pose, goal,
                                           self.free_orientation.value, self.tolerance.value,
                                           self.optimize.value, self.cancel.is_set)

    def _show_plan(self, plan: Plan, minimum: float, seconds: float, refinement_status: str):
        self.path = plan.path
        positions = np.array([pose.position for pose in plan.path])
        lines = np.stack((positions[:-1], positions[1:]), axis=1)
        self.path_handle = self.server.scene.add_line_segments(
            "/path", lines, (60, 244, 181), thickness=4, thickness_units="screen")
        self.last_plan_summary = (f"{len(plan.path)} 个路点 · {seconds:.2f} s{refinement_status}  \n"
                                  f"5 mm 复检最小净空 **{100 * minimum:.2f} cm**")
        if self.auto_execute:
            self._execute()
        else:
            self.execute_button.disabled = False
            self._report("路径已就绪", self.last_plan_summary)

    def _execute(self):
        if self.path is not None:
            self.execute_button.disabled = True
            self.following = Follower(self.path, self.checker)
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
        try:
            sample = self.following.step(dt)
        except TrackingError as error:
            self.following = None
            self._report("执行已停止", str(error))
            return
        self.pose = sample.pose
        _move(self.body, self.pose)
        if self.following.done:
            position, angle = self.following.terminal_error
            minimum = min(state.clearance for state in self.following.samples)
            self._report("已到达目标", f"{self.last_plan_summary}  \n"
                         f"跟踪最小净空 **{100 * minimum:.2f} cm**  \n"
                         f"终点误差 **{1000 * position:.2f} mm / {np.rad2deg(angle):.2f}°**")
            self.following = None
        elif sample.time - self.last_ui_update >= .16:
            self.last_ui_update = sample.time
            self._report("执行中", f"{self.last_plan_summary}  \n"
                         f"当前净空 **{100 * sample.clearance:.2f} cm** · {sample.time:.1f} s")

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
