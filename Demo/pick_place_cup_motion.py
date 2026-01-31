import numpy as np
from typing import Dict, Optional

CLOSE = 0.0
OPEN = -3.4
GRIPPER_OFFSET = 0.15
GRIPPER_CUP = -2.0
Z_CUP = 0.08


def make_pick_move_action(pt_ref: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """张开夹爪偏移到目标附近，保持不动。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    return {
        "left": np.array(
            [base[0] - GRIPPER_OFFSET-0.02, base[1], base[2], 0, 0, 0, OPEN],
            dtype=np.float32,
        ),
        "right": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    }


def make_pick_robust_action(pt_ref: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """执行向前移动，准备鲁棒夹取位置，保持不动。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    return {
        "left": np.array(
            [base[0] - GRIPPER_OFFSET + 0.05, base[1], base[2], 0, 0, 0, OPEN],
            dtype=np.float32,
        ),
        "right": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    }


def make_close_action(pt_ref: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """执行夹紧动作，保持不动。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    return {
        "left": np.array(
            [base[0] - GRIPPER_OFFSET + 0.05, base[1],
                base[2], 0, 0, 0, GRIPPER_CUP],
            dtype=np.float32,
        ),
        "right": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    }


def make_pick_stop_action(pt_ref: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """回撤一点抓回位置，抬高保证一个重力对抗，保持不动。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    return {
        "left": np.array(
            [base[0] - GRIPPER_OFFSET, base[1],
             base[2] + Z_CUP, 0, 0, 0, GRIPPER_CUP],
            dtype=np.float32,
        ),
        "right": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    }


def make_pick_back_action(pt_ref: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """夹住回到初始位置，保持不动。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    return {
        "left": np.array([0, 0, 0, 0, 0, 0, GRIPPER_CUP], dtype=np.float32),
        "right": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    }


def make_place_move_action(pt_ref: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """保持抓取偏移到放置目标附近，保持不动。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    return {
        "left": np.array(
            [base[0] - GRIPPER_OFFSET, base[1],
             base[2] + Z_CUP+0.05, 0, 0, 0, GRIPPER_CUP],
            dtype=np.float32,
        ),
        "right": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    }


def make_place_robust_action(pt_ref: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """执行向前移动，准备鲁棒放置位置。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    return {
        "left": np.array(
            [base[0] - GRIPPER_OFFSET-0.01, base[1],
             base[2] + Z_CUP+0.05, 0, 0, 0, GRIPPER_CUP],
            dtype=np.float32,
        ),
        "right": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    }


def make_down_action(pt_ref: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """下降到放置位置，保持不动。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    return {
        "left": np.array(
            [base[0] - GRIPPER_OFFSET-0.01, base[1],
             base[2] + Z_CUP-0.02, 0, 0, 0, GRIPPER_CUP],
            dtype=np.float32,
        ),
        "right": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    }


def make_open_action(pt_ref: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """夹爪张开放置"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    return {
        "left": np.array(
            [base[0] - GRIPPER_OFFSET-0.01, base[1], base[2] + Z_CUP-0.02, 0, 0, 0, OPEN], dtype=np.float32),
        "right": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)}


def make_place_stop_action(pt_ref: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """回撤一点放置位置，保持不动。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    return {
        "left": np.array(
            [base[0] - GRIPPER_OFFSET - 0.06, base[1],
             base[2] + Z_CUP, 0, 0, 0, OPEN],
            dtype=np.float32,
        ),
        "right": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    }


def make_release_action(pt_ref: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """夹爪home位置张开放置"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    return {
        "left": np.array([0, 0, 0, 0, 0, 0, OPEN], dtype=np.float32),
        "right": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)}


def build_pick_cup_sequence(pt_ref: Optional[np.ndarray]):
    """返回抓取动作序列（右臂），不执行。"""
    return [
        make_pick_move_action(pt_ref),
        make_pick_robust_action(pt_ref),
        make_close_action(pt_ref),
        make_pick_stop_action(pt_ref),
        make_pick_back_action(pt_ref),
    ]


def build_place_cup_sequence(pt_ref: Optional[np.ndarray]):
    """返回放置动作序列（右臂），不执行。"""
    return [
        make_place_move_action(pt_ref),
        make_place_robust_action(pt_ref),
        make_down_action(pt_ref),
        make_open_action(pt_ref),
        make_place_stop_action(pt_ref),
    ]
