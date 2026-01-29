import numpy as np
import sys
import termios
import tty
from typing import List, Tuple
import math

def depth_to_meters(raw_depth: float) -> float:
    """兼容毫米与米的深度值。"""
    if not np.isfinite(raw_depth) or raw_depth <= 0:
        raise ValueError(f"无效深度值: {raw_depth}")
    if raw_depth > 10.0:
        return float(raw_depth) / 1000.0
    return float(raw_depth)

def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def path_to_actions(
    path: List[Tuple[int, int]],
    init_yaw: float = 0.0,
):
    actions = []
    cur_yaw = init_yaw

    for i in range(1, len(path)):
        x0, y0 = path[i - 1]
        x1, y1 = path[i]

        # dx = (x1 - x0) * cell_size
        # dy = (y1 - y0) * cell_size
        dx = x1 - x0
        dy = y1 - y0

        target_yaw = math.atan2(dy, dx)
        d_yaw = target_yaw - cur_yaw

        # 归一化到 [-pi, pi]
        while d_yaw > math.pi:
            d_yaw -= 2 * math.pi
        while d_yaw < -math.pi:
            d_yaw += 2 * math.pi

        dist = math.hypot(dx, dy)

        if abs(d_yaw) > 1e-3:
            actions.append(("rotate", -d_yaw))
            cur_yaw = target_yaw

        if dist > 1e-3:
            actions.append(("forward", dist))

    return actions

Action = Tuple[str, float]

def merge_forward_actions(actions: List[Action]) -> List[Action]:
    merged = []
    forward_acc = 0.0

    for act, val in actions:
        if act == "forward":
            forward_acc += val
        else:
            # 遇到非 forward，先把累计的 forward 放进去
            if forward_acc > 0:
                merged.append(("forward", forward_acc))
                forward_acc = 0.0
            merged.append((act, val))

    # 结尾如果是 forward
    if forward_acc > 0:
        merged.append(("forward", forward_acc))

    return merged