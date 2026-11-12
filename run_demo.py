"""Run the local, browser-based rigid or articulated planning demo."""

import argparse
import webbrowser

import viser

from anyshape_planner.articulated import ArticulatedRobot
from anyshape_planner.corridor import MODEL
from anyshape_planner.geometry import Robot, load_mesh
from anyshape_planner.scenes import l_mesh
from anyshape_planner.viz import Demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = {"window": "窗口穿越", "clutter": "杂乱环境", "corridor": "双臂收缩走廊"}
    parser.add_argument("--scene", choices=names, default="window")
    parser.add_argument("--robot", help="Watertight OBJ or STL robot, with vertices in metres")
    parser.add_argument("--urdf", help="Articulated robot URDF with a matching .poses.json file")
    parser.add_argument("--sample-pitch", type=float, help="Robot sample spacing in metres")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--open", action="store_true", help="Open the demo in the default browser")
    args = parser.parse_args()
    pitch = args.sample_pitch
    robot = Robot.from_mesh(load_mesh(args.robot) if args.robot else l_mesh(), .075 if pitch is None else pitch)
    articulated = None
    if args.urdf or (args.scene == "corridor" and pitch is not None):
        articulated = ArticulatedRobot.from_urdf(args.urdf or MODEL, .05 if pitch is None else pitch)
    server = viser.ViserServer(host="127.0.0.1", port=args.port, verbose=False)
    try:
        demo = Demo(server, names[args.scene], robot, args.seed, articulated)
        url = f"http://127.0.0.1:{server.get_port()}"
        print(f"Demo ready at {url}", flush=True)
        if args.open:
            webbrowser.open(url)
        demo.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
