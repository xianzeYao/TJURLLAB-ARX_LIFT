"""
手工选点 → Pure Pursuit 跟踪 → 底盘控制。

运行:
    python3 Demo/nav_try.py

交互:
- 左键添加路径点，右键撤销
- 按 e 执行当前批次；按 r 刷新获取新帧；按 q 退出（不执行）
- 使用 `camera_h` RGB + 对齐深度，分辨率 640x480
"""
from __future__ import annotations

import time

import numpy as np

from point2pos_utils import load_cam2ref, load_intrinsics

import sys
sys.path.append("../ARX_Realenv/ROS2")  # noqa: E402
from arx_ros2_env import ARXRobotEnv  # noqa: E402

from nav_try_utils import (
    pick_pixels,
    pixels_to_path,
    follow_path_online,
)


def main():
    arx = None
    try:
        # 初始化机器人环境
        arx = ARXRobotEnv(
            duration_per_step=1.0 / 20.0,
            min_steps_per_action=60,
            min_steps_gripper=20,
            max_v_xyz=0.1,
            max_v_rpy=0.1,
            camera_type="all",
            camera_view=("camera_h",),
            img_size=(640, 480),
        )
        time.sleep(4.0)
        arx.reset()
        arx.step_lift(19.0)

        K = load_intrinsics()
        T_cam2ref = load_cam2ref()

        while True:
            # 每轮获取最新 RGB+深度
            frames = arx.node.get_camera(
                target_size=(640, 480), return_status=False)
            color = frames.get("camera_h_color")
            depth = frames.get("camera_h_aligned_depth_to_color")
            if color is None or depth is None:
                print("未获取到彩色或深度图像，退出。")
                break

            res = pick_pixels(color)
            # unpack with backward compatibility
            if len(res) == 4:
                pixels, confirmed, refresh_flag, quit_flag = res
            elif len(res) == 3:
                pixels, confirmed, quit_flag = res
                refresh_flag = False
            elif len(res) == 2:
                pixels, confirmed = res
                refresh_flag = False
                quit_flag = False
            else:
                print(f"pick_pixels 返回长度异常: {len(res)}，跳过本批次。")
                continue

            if quit_flag:
                print("收到退出指令，不执行，结束程序。")
                break
            if refresh_flag:
                print("刷新图像，重新点选。")
                continue
            if not confirmed:
                print("本批次取消，继续等待下一次点选。")
                continue
            if len(pixels) < 2:
                print("路径点数量不足，跳过本批次。")
                continue

            path_xy = pixels_to_path(pixels, depth, K, T_cam2ref)
            if len(path_xy) < 2:
                print("有效路径点不足，跳过本批次。")
                continue

            # 在线纯跟踪，使用实测三轮角速度闭环积分
            follow_path_online(
                arx,
                path_xy,
                lookahead=0.12,
                v_max=0.12,
                v_min=0.06,
                omega_max=0.35,
                dt=0.05,
                use_meas_wheel=False,  # False=用下发指令积分（更平滑）；True=用实测轮速积分
            )
    finally:
        if arx is not None:
            try:
                arx.close()
            except Exception as exc:
                print(f"关闭环境时出错: {exc}")


if __name__ == "__main__":
    main()
