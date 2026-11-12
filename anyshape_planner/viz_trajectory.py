"""Time plots and CSV export for the trajectory the simulator executes."""

import csv
from io import StringIO

import numpy as np


class JointTrajectoryPlots:
    def __init__(self, gui):
        self.folder = gui.add_folder("关节轨迹", visible=False)
        self.plots = []
        self.data = b""
        with self.folder:
            self.clock = gui.add_markdown("")
            self.download = gui.add_button("下载轨迹 CSV")

        @self.download.on_click
        def download(event):
            content = self.data
            if event.client is not None and content:
                event.client.send_file_download("planned-trajectory.csv", content, save_immediately=True)

        self.gui = gui

    def clear(self):
        self.folder.visible = False
        for plot in self.plots:
            plot.remove()
        self.plots.clear()
        self.data = b""

    def show(self, trajectory, times):
        self.clear()
        robot = trajectory.checker.robot
        states = trajectory.spline(times)
        speeds = trajectory.spline(times, nu=1)
        wheels = trajectory.wheel_speeds(times)
        joints = [joint for joint in robot.joints if joint.dof is not None]
        names = {"lift": "升降 lift", "shoulder": "肩关节 shoulder", "elbow": "肘关节 elbow"}
        grid = {"stroke": "#394653"}
        time_scale = {"x": {"time": False, "range": (0., trajectory.duration)}}
        with self.folder:
            for joint in joints:
                index = joint.dof + 3
                unit = "m" if joint.prismatic else "°"
                multiplier = 1.0 if joint.prismatic else 180 / np.pi
                self.plots.append(self.gui.add_uplot(
                    (times, multiplier * states[:, index], multiplier * speeds[:, index]),
                    ({}, {"label": f"位置 ({unit})", "stroke": "#5ce1b7", "width": 2},
                     {"label": f"速度 ({unit}/s)", "stroke": "#ffb65e", "width": 1.5,
                      "scale": "speed", "dash": (5, 4)}),
                    title=names.get(joint.name, joint.name), height=124,
                    scales=time_scale,
                    axes=({"scale": "x", "stroke": "#a6b1be", "grid": grid},
                          {"scale": "y", "size": 41, "stroke": "#5ce1b7", "grid": grid},
                          {"scale": "speed", "side": 1, "size": 36, "stroke": "#ffb65e",
                           "grid": {"show": False}})))
            self.plots.append(self.gui.add_uplot(
                (times, *wheels.T),
                ({}, *({"label": label, "stroke": color, "width": 1.5} for label, color in
                       zip(("左前", "右前", "左后", "右后"), ("#5ce1b7", "#ffb65e", "#83bfff", "#ee8dc7")))),
                title="四轮角速度 (rad/s)", height=124, scales=time_scale,
                axes=({"scale": "x", "stroke": "#a6b1be", "grid": grid},
                      {"scale": "y", "size": 36, "stroke": "#a6b1be", "grid": grid})))
        stream = StringIO(newline="")
        writer = csv.writer(stream)
        coordinates = ["base_x_m", "base_y_m", "base_yaw_rad"]
        coordinates.extend(f"{joint.name}_{'m' if joint.prismatic else 'rad'}" for joint in joints)
        writer.writerow(["time_s", *coordinates, *(name + "_per_s" for name in coordinates),
                         "wheel_fl_rad_s", "wheel_fr_rad_s", "wheel_rl_rad_s", "wheel_rr_rad_s"])
        writer.writerows(np.column_stack((times, states, speeds, wheels)))
        self.data = stream.getvalue().encode("utf-8")
        self.set_time(0.0, trajectory.duration)
        self.folder.visible = True

    def set_time(self, time, duration):
        self.clock.content = f"横轴为时间（s） · **{time:.2f} / {duration:.2f} s**"
