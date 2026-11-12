# 6-DoF 刚体路径规划演示

用 Python、ESDF 和 SE(3) RRT-Connect 规划网格机器人的位姿路径，通过 viser 在浏览器里拖动目标并执行运动学仿真。默认场景中，L 形机器人在初始姿态下沿窗口短边展开 1.34 m，窗口宽 0.80 m、高 0.88 m。墙面覆盖工作空间截面，机器人需要旋转才能穿过窗口。

![L 形机器人旋转穿过窗口](assets/window-demo.gif)

## 启动

在项目目录中使用 Python 3.11 或更新版本。

```powershell
python -m pip install -r requirements.txt
python run_demo.py --open
```

打开 [本地演示](http://127.0.0.1:8080/)，点击 **规划** 查看绿色路径，然后点击 **执行**。橙色实体是机器人，粉色线框是目标位姿。拖动目标手柄的轴或圆环可修改位置或姿态。执行中拖动并松开手柄时，机器人从当前位置重新规划并继续执行。可在场景下拉框切换到随机盒体和圆柱组成的杂乱环境。

自定义机器人需要封闭的 OBJ 或 STL 网格，顶点坐标单位为米，变换以网格原点为参考。可以通过 `--sample-pitch` 调整机器人采样间距。工作空间和默认起终点由 `scenes.py` 定义。

```powershell
python run_demo.py --robot path/to/robot.stl --sample-pitch 0.05 --open
```

安装可选依赖后，勾选 **ESDF 轨迹优化** 可使用 PyTorch 优化路点。优化固定两端位姿，候选路径须通过连续碰撞检查和 5 mm 复检。

```powershell
python -m pip install -r requirements-optimizer.txt
```

## 复现测试

```powershell
python -m anyshape_planner.tests.benchmark --stage all --output demo_results.json
```

`--stage core` 运行碰撞、规划和控制测试。`collision`、`plan`、`control`、`optimizer` 可分别运行单个阶段。

| 验收项 | 测试结果 |
| --- | --- |
| 窗口场景 10,000 个随机位姿与 FCL 对拍 | 99.30% 一致，70 个保守误报，0 个漏检 |
| 窗口穿越 | 20/20 成功，按刚体点位移上界 5 mm 重采样的最小 ESDF 净空为 6.46 mm，重采样位姿也通过 FCL 检查 |
| 杂乱环境 | 固定障碍布局，10 个规划种子全部成功 |
| 运动学跟踪 | 三次固定目标与一次改目标，终点误差均小于 0.50 mm 和 0.04° |
| PyTorch 优化 | 窗口种子 0 的目标函数从 0.00654 降至 0.00177，FCL 密采样通过，碰撞检查拒绝候选时回退 |

浏览器中实测了分开规划与执行、拖动目标手柄触发重新规划、切换杂乱场景，以及启用 PyTorch 优化。窗口演示终点误差为 0.48 mm 和 0.03°，拖动目标后到达新目标的误差为 0.35 mm 和 0.02°。

逐次耗时和验收数据保存在 [demo_results.json](demo_results.json)。窗口测试采用窗口中心附近的位姿采样偏置，同时保留全空间随机采样。连续推进使用三线性场的 Lipschitz 上界。净空基于离散 ESDF 与有限机器人采样点，换用更细的几何特征时，需要重新验证体素和采样分辨率。对拍的 0 漏检只描述上述测试样本。
