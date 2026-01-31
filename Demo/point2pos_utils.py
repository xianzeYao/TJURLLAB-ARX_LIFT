"""
将 2D 像素点 + 对齐深度转换为基坐标系下的 6 维末端姿态。（部分做clip）

默认使用的相机的标定文件：
- 内参: instrics_right4camerah.json
- 外参: final_extrinsics_cam_h_right.json (T_cam2ref)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_INTRINSICS = WORKSPACE / "ARX_Realenv/Tools/instrinsics_camerah.json"
DEFAULT_EXTRINSICS = WORKSPACE / \
    "ARX_Realenv/Tools/new_extrinsics_cam_h_left.json"

BIAS_REF2CAM = np.array([0.48, 0.0, 0.0, 0.0])


def depth_to_meters(raw_depth: float) -> float:
    """将深度值转换为米单位，兼容毫米与米输入。"""
    if not np.isfinite(raw_depth) or raw_depth <= 0:
        raise ValueError(f"无效深度值: {raw_depth}")
    if raw_depth > 10.0:
        return float(raw_depth) / 1000.0
    return float(raw_depth)


def load_intrinsics(path: Path | str | None = None) -> np.ndarray:
    """读取 3x3 相机内参矩阵。"""
    intr_path = Path(path) if path else DEFAULT_INTRINSICS
    data = json.loads(intr_path.read_text())
    K = np.asarray(data["camera_matrix"], dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"内参矩阵形状异常: {K.shape}")
    return K


def load_cam2ref(path: Path | str | None = None) -> np.ndarray:
    """读取 4x4 齐次矩阵 T_cam2ref。"""
    ext_path = Path(path) if path else DEFAULT_EXTRINSICS
    payload = json.loads(ext_path.read_text())
    T = np.asarray(payload.get("T_cam2ref"), dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"外参矩阵形状异常: {T.shape}")
    return T


def pixel_to_ref_point(
    pixel: Tuple[int, int],
    depth_image: np.ndarray,
    K: np.ndarray,
    T_cam2ref: np.ndarray,
) -> np.ndarray:
    """像素 + 深度 -> 基坐标系 3D 点。"""
    u, v = pixel
    H, W = depth_image.shape
    if not (0 <= u < W and 0 <= v < H):
        raise ValueError(f"像素越界: {(u, v)} not in [0,{W})x[0,{H})")
    z = depth_to_meters(float(depth_image[v, u]))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    cam_point = np.array([x_cam, y_cam, z, 1.0], dtype=np.float64)
    ref_point = T_cam2ref @ cam_point
    return ref_point[:3]


def pixel_to_base_point(
    pixel: Tuple[int, int],
    depth_image: np.ndarray,
    K: np.ndarray,
    T_cam2ref: np.ndarray,
) -> np.ndarray:
    """像素 + 深度 -> 基坐标系 3D 点。"""
    u, v = pixel
    H, W = depth_image.shape
    if not (0 <= u < W and 0 <= v < H):
        raise ValueError(f"像素越界: {(u, v)} not in [0,{W})x[0,{H})")
    z = depth_to_meters(float(depth_image[v, u]))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    cam_point = np.array([x_cam, y_cam, z, 1.0], dtype=np.float64)
    base_point = T_cam2ref @ cam_point + BIAS_REF2CAM
    return base_point[:2]


__all__ = [
    "load_intrinsics",
    "load_cam2ref",
    "depth_to_meters",
    "pixel_to_ref_point",
    "pixel_to_base_point",
]
