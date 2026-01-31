"""
导航/跟踪的工具函数。

- 像素点转基坐标路径
- Pure Pursuit 轨迹跟踪求解 (离散为底盘命令)
- 简单的 OpenCV 交互式点选
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from point2pos_utils import pixel_to_base_point

# ---- 三轮全向：几何模型 & 标定矩阵（可选） ----
# 轮序：temp_float_data[1]=后轮(1)，[2]=右前(2)，[3]=左前(3)
M_WHEEL2CMD = np.array([
    [0.0,       0.115389, -0.115389],
    [0.134163, -0.067077, -0.067086],
    [0.137048,  0.122361,  0.171939],
], dtype=np.float64)


def wheels_radps_to_cmd(w1: float, w2: float, w3: float) -> tuple[float, float, float]:
    """轮角速度(rad/s) -> 估计底盘指令 (chx, chy, chz)，基于标定矩阵。"""
    w = np.array([w1, w2, w3], dtype=np.float64)
    cmd = M_WHEEL2CMD @ w
    return float(cmd[0]), float(cmd[1]), float(cmd[2])


def wheels_radps_to_body_geom(chassis_r: float, chassis_L: float, w1: float, w2: float, w3: float):
    """
    轮角速度(rad/s) -> 车体速度 (vx, vy, wz)，基于理想几何模型。
    假设驱动方向角：后轮0°，右前120°，左前240°；坐标系 x前 y左 wz逆时针。
    """
    v1 = chassis_r * w1
    v2 = chassis_r * w2
    v3 = chassis_r * w3
    vx = (2.0 / 3.0) * v1 - (1.0 / 3.0) * v2 - (1.0 / 3.0) * v3
    vy = (1.0 / np.sqrt(3.0)) * (v2 - v3)
    wz = (1.0 / (3.0 * chassis_L)) * (v1 + v2 + v3)
    return vx, vy, wz


def follow_path_online(
    arx,
    path_xy: Sequence[Tuple[float, float]],
    lookahead: float = 0.12,
    v_max: float = 0.12,
    v_min: float = 0.06,
    omega_max: float = 0.35,
    dt: float = 0.05,
    use_meas_wheel: bool = False,
    chassis_r: float = 0.15,
    chassis_L: float = 0.376,
):
    """
    在线纯跟踪：每周期重算曲率并下发一次指令。

    use_meas_wheel=False: 用“下发指令”推积分（更平滑，接近早期行为）。
    use_meas_wheel=True : 用实测轮速→cmd→速度积分（更贴真实）。
    """
    if len(path_xy) < 2:
        print("路径点不足，终止。")
        return
    pose = (0.0, 0.0, 0.0)  # x, y, theta
    while True:
        # --- 轮速读取（仅在用实测时需要） ---
        if use_meas_wheel:
            status = arx.node.get_robot_status()
            base = status.get("base") if status else None
            if base is None:
                print("未获得底盘状态，停止。")
                break
            w1 = float(base.temp_float_data[1])
            w2 = float(base.temp_float_data[2])
            w3 = float(base.temp_float_data[3])
        else:
            w1 = w2 = w3 = 0.0

        # --- 使用当前姿态选 lookahead ---
        x_t, y_t, dist = get_lookahead_point(pose, path_xy, lookahead)
        if dist < 0.05:
            break

        curvature = 2.0 * y_t / (dist * dist + 1e-6)
        omega = 1.2 * curvature
        v = v_max * math.exp(-abs(omega))
        v = max(min(v, v_max), v_min)
        omega = max(min(omega, omega_max), -omega_max)
        chx = math.sqrt(max(v, 0.0) / 0.24)
        chz = omega / (2 * math.pi / 20.6)

        arx.step_base(chx, 0.0, chz, dt)

        # --- 积分姿态 ---
        if use_meas_wheel:
            # 标定矩阵反解 → 指令 → 速度
            chx_m, chy_m, chz_m = wheels_radps_to_cmd(w1, w2, w3)
            vx_m = 0.24 * math.copysign(chx_m * chx_m, chx_m)
            vy_m = 0.24 * math.copysign(chy_m * chy_m, chy_m)
            wz_m = chz_m * (2 * math.pi / 20.6)
        else:
            vx_m = 0.24 * math.copysign(chx * chx, chx)
            vy_m = 0.0  # 当前未下发横向指令
            wz_m = chz * (2 * math.pi / 20.6)

        x, y, th = pose
        x += vx_m * dt
        y += vy_m * dt
        th += wz_m * dt
        if th > math.pi:
            th -= 2 * math.pi
        elif th < -math.pi:
            th += 2 * math.pi
        pose = (x, y, th)

    # stop once
    arx.step_base(0.0, 0.0, 0.0, 0.1)


def get_lookahead_point(
    pose: Tuple[float, float, float],
    path_xy: Sequence[Tuple[float, float]],
    lookahead: float,
) -> Tuple[float, float, float]:
    """在机器人坐标系下找到满足前视距离的目标点。"""
    x_r, y_r, theta_r = pose
    for x_g, y_g in path_xy:
        dx = x_g - x_r
        dy = y_g - y_r
        x_t = math.cos(-theta_r) * dx - math.sin(-theta_r) * dy
        y_t = math.sin(-theta_r) * dx + math.cos(-theta_r) * dy
        dist = math.hypot(x_t, y_t)
        if dist >= lookahead and x_t > 0:
            return x_t, y_t, dist
    # fallback 到终点
    x_g, y_g = path_xy[-1]
    dx = x_g - x_r
    dy = y_g - y_r
    x_t = math.cos(-theta_r) * dx - math.sin(-theta_r) * dy
    y_t = math.sin(-theta_r) * dx + math.cos(-theta_r) * dy
    dist = math.hypot(x_t, y_t)
    return x_t, y_t, dist


def pure_pursuit_plan(
    path_xy: Sequence[Tuple[float, float]],
    lookahead: float = 0.12,
    v_max: float = 0.12,
    v_min: float = 0.06,
    omega_max: float = 0.35,
    dt: float = 0.05,
) -> List[Tuple[float, float, float]]:
    """根据路径生成离散底盘指令 (chx, chz, duration)。"""
    if len(path_xy) < 2:
        raise ValueError("路径点至少需要 2 个。")
    pose = (0.0, 0.0, 0.0)
    cmds: List[Tuple[float, float, float]] = []
    while True:
        x_t, y_t, dist = get_lookahead_point(pose, path_xy, lookahead)
        if dist < 0.05:
            break
        curvature = 2.0 * y_t / (dist * dist + 1e-6)
        omega = 1.2 * curvature
        v = v_max * math.exp(-abs(omega))
        v = max(min(v, v_max), v_min)
        omega = max(min(omega, omega_max), -omega_max)
        chx = math.sqrt(max(v, 0.0) / 0.24)
        chz = omega / (2 * math.pi / 20.6)
        if cmds and abs(cmds[-1][0] - chx) < 1e-3 and abs(cmds[-1][1] - chz) < 1e-3:
            last_chx, last_chz, dur = cmds[-1]
            cmds[-1] = (last_chx, last_chz, dur + dt)
        else:
            cmds.append((chx, chz, dt))
        x_r, y_r, theta_r = pose
        x_r += v * math.cos(theta_r) * dt
        y_r += v * math.sin(theta_r) * dt
        theta_r += omega * dt
        if theta_r > math.pi:
            theta_r -= 2 * math.pi
        elif theta_r < -math.pi:
            theta_r += 2 * math.pi
        pose = (x_r, y_r, theta_r)
    return cmds


def pick_pixels(color: np.ndarray) -> tuple[List[Tuple[int, int]], bool, bool, bool]:
    """OpenCV窗口手工点选路径点。

    左键添加，右键撤销；按 e 执行当前批次；按 r 刷新成最新图；按 q 退出（不执行）。
    返回 (points, confirmed, refresh_flag, quit_flag)
    """
    disp = color.copy()
    clicks: List[Tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        nonlocal disp
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and clicks:
            clicks.pop()
        disp = color.copy()
        for idx, (u, v) in enumerate(clicks, 1):
            cv2.circle(disp, (u, v), 5, (0, 0, 255), -1)
            cv2.putText(
                disp, str(idx), (u + 6, v - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
            )

    win = "nav_try_pixels"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    print("左键添加，右键撤销；e 执行，r 刷新重取图，q 退出（不执行）。")
    confirmed = False
    refresh_flag = False
    quit_flag = False
    while True:
        cv2.imshow(win, disp)
        key = cv2.waitKey(10) & 0xFF
        if key in (ord("e"), ord("E")):
            confirmed = True
            break
        if key in (ord("r"), ord("R")):
            refresh_flag = True
            break
        if key in (ord("q"), ord("Q")):
            quit_flag = True
            break
    cv2.destroyWindow(win)
    return clicks, confirmed, refresh_flag, quit_flag


def pixels_to_path(
    pixels: Sequence[Tuple[int, int]],
    depth: np.ndarray,
    K: np.ndarray,
    T_cam2ref: np.ndarray,
) -> List[Tuple[float, float]]:
    """像素路径 -> 基坐标系 (x, y) 序列。"""
    path: List[Tuple[float, float]] = []
    for px in pixels:
        try:
            xy = pixel_to_base_point(px, depth, K, T_cam2ref)
            path.append((float(xy[0]), float(xy[1])))
        except Exception as exc:
            print(f"像素 {px} 转换失败: {exc}")
    return path


def execute_commands(arx, cmds: Sequence[Tuple[float, float, float]]):
    """按序发送底盘命令，结束后补一次停止。"""
    for chx, chz, duration in cmds:
        arx.step_base(chx, 0.0, chz, duration)
    arx.step_base(0.0, 0.0, 0.0, 0.1)


__all__ = [
    "get_lookahead_point",
    "pure_pursuit_plan",
    "pick_pixels",
    "pixels_to_path",
    "execute_commands",
    "follow_path_online",
    "wheels_radps_to_cmd",
]
