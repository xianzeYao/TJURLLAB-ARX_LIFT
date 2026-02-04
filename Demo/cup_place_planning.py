"""
Cup place planning demo.

逻辑要求：
- pick prompt 固定为 "left side cups not on coaster"。
- 前 3 次 place：使用单点 text prompt "the coaster with the smallest"。
- 第 4 次及之后的 place：使用双点 prompt
    1. Point to top of the cup that has no cup on it.
    2. Point to top of the another cup that has no cup on it.
    - Output the pixel coordinates of the center points.
  取两个点的深度，UV 取平均用于可视化，3D 点取两个 ref 点的均值作为放杯位。

其余交互、动作序列参考 cup_pick_planning.py。
"""
from __future__ import annotations

import textwrap
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from pick_place_cup_motion import build_pick_cup_sequence, build_place_cup_sequence
from point2pos_utils import load_cam2ref, load_intrinsics, pixel_to_ref_point
from arx_pointing import predict_multi_points_from_rgb, predict_point_from_rgb

import sys

sys.path.append("../ARX_Realenv/ROS2")  # noqa
from arx_ros2_env import ARXRobotEnv  # noqa


COLOR_TOPIC = "/camera_h_namespace/camera_h/color/image_rect_raw"
DEPTH_TOPIC = "/camera_h_namespace/camera_h/aligned_depth_to_color/image_raw"


class FrameBuffer(Node):
    """订阅彩色与对齐深度，缓存最新帧。"""

    def __init__(self):
        super().__init__("cup_place_planning")
        self.bridge = CvBridge()
        self.latest_color: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.create_subscription(Image, COLOR_TOPIC, self._on_color, 5)
        self.create_subscription(Image, DEPTH_TOPIC, self._on_depth, 5)
        self.get_logger().info(f"订阅彩色: {COLOR_TOPIC}, 对齐深度: {DEPTH_TOPIC}")

    def _on_color(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.lock:
            self.latest_color = img

    def _on_depth(self, msg: Image):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32)
        with self.lock:
            self.latest_depth = depth

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self.lock:
            return self.latest_color, self.latest_depth


def _filter_valid_points(
    uv_list: List[Tuple[int, int]],
    depth: np.ndarray,
    K: np.ndarray,
    T_cam2ref: np.ndarray,
) -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
    """过滤深度无效的像素点并转换为 ref 坐标系 3D 点。"""
    valid_uvs: List[Tuple[int, int]] = []
    pt_refs: List[np.ndarray] = []
    for uv in uv_list:
        raw_depth = depth[uv[1], uv[0]]
        if np.isnan(raw_depth) or raw_depth == 0:
            print(f"预测像素 {uv} 深度无效({raw_depth})，跳过该点")
            continue
        pt_refs.append(pixel_to_ref_point(uv, depth, K, T_cam2ref))
        valid_uvs.append(uv)
    return valid_uvs, pt_refs


def main():
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
    arx.reset()
    arx.step_lift(13.0)

    window_node = FrameBuffer()
    K = load_intrinsics()
    T_cam2ref = load_cam2ref()

    current_pick_uv: Optional[Tuple[int, int]] = None
    current_pick_ref: Optional[np.ndarray] = None

    predicted_place_uv: Optional[Tuple[int, int]] = None
    place_pt_ref: Optional[np.ndarray] = None
    attachment_uvs: Optional[List[Tuple[int, int]]] = None

    pick_prompt = "left side cup's top"
    place_prompt_single = "the coaster with the smallest number"
    place_prompt_multi = """1. Point to top of the cup that has no cup on it.
                            2. Point to top of the another cup near it the most.
                            - Output the pixel coordinates of the center points."""
    current_place_prompt = place_prompt_single

    try:
        win = "cup_place_planning"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        i = 0  # 偶数轮 pick，奇数轮 place
        while rclpy.ok():
            rclpy.spin_once(window_node, timeout_sec=0.05)
            color, depth = window_node.get_frames()
            if color is None or depth is None:
                cv2.waitKey(1)
                continue

            need_pick = i % 2 == 0
            need_place = i % 2 == 1

            if (need_pick and current_pick_ref is None) or (need_place and place_pt_ref is None):
                if need_pick:
                    # 保持所有 pick 高度为 13.0
                    arx.step_lift(13.0)
                    u, v = predict_point_from_rgb(
                        color,
                        text_prompt=pick_prompt,
                        assume_bgr=False,
                    )
                    current_pick_uv = (int(round(u)), int(round(v)))
                    raw_depth = depth[current_pick_uv[1], current_pick_uv[0]]
                    if np.isnan(raw_depth) or raw_depth == 0:
                        print(
                            f"pick 像素 {current_pick_uv} 深度无效({raw_depth})，按 r 重新预测"
                        )
                        current_pick_uv = None
                        current_pick_ref = None
                        continue
                    current_pick_ref = pixel_to_ref_point(
                        current_pick_uv, depth, K, T_cam2ref
                    )
                    print(
                        f"pick 预测像素 {current_pick_uv} -> ref {current_pick_ref.tolist()}，按 e 执行抓取"
                    )
                else:
                    color_latest, depth_latest = window_node.get_frames()
                    if color_latest is None or depth_latest is None:
                        cv2.waitKey(1)
                        continue

                    # 通过 i 来切换单点 / 多点放置逻辑
                    if i in (1, 3, 5):  # 前三次放置
                        u, v = predict_point_from_rgb(
                            color_latest,
                            text_prompt=place_prompt_single,
                            assume_bgr=False,
                        )
                        predicted_place_uv = (int(round(u)), int(round(v)))
                        attachment_uvs = None
                        raw_depth = depth_latest[predicted_place_uv[1],
                                                 predicted_place_uv[0]]
                        if np.isnan(raw_depth) or raw_depth == 0:
                            print(
                                f"预测像素 {predicted_place_uv} 深度无效({raw_depth})，按 r 重新预测"
                            )
                            predicted_place_uv = None
                            place_pt_ref = None
                            continue
                        place_pt_ref = pixel_to_ref_point(
                            predicted_place_uv, depth_latest, K, T_cam2ref
                        )
                        current_place_prompt = place_prompt_single
                    else:
                        # 第 4/5 次放置预测前升降到 1.0，第 6 次升降到 20.0
                        if i == 7 or i == 9:
                            arx.step_lift(17.0)
                        if i == 11:
                            arx.step_lift(20.0)
                        raw_uvs, raw_text = predict_multi_points_from_rgb(
                            color_latest,
                            text_prompt="",
                            all_prompt=place_prompt_multi,
                            assume_bgr=False,
                            return_raw=True,
                        )
                        raw_uvs = raw_uvs[:2]  # 取前两个点
                        print(f"place 模型回答: {raw_text}")
                        uv_ints = [(int(round(u)), int(round(v)))
                                   for (u, v) in raw_uvs]
                        valid_uvs, valid_refs = _filter_valid_points(
                            uv_ints, depth_latest, K, T_cam2ref
                        )
                        if len(valid_uvs) < 2:
                            print("需要至少 2 个有效放置参考点，按 r 重新预测")
                            predicted_place_uv = None
                            place_pt_ref = None
                            attachment_uvs = None
                            continue
                        attachment_uvs = valid_uvs[:2]
                        predicted_place_uv = tuple(
                            np.mean(np.array(attachment_uvs),
                                    axis=0).round().astype(int)
                        )
                        place_pt_ref = np.mean(
                            np.stack(valid_refs[:2]), axis=0)
                        current_place_prompt = place_prompt_multi

                    print(
                        f"place 预测像素 {predicted_place_uv} -> ref {place_pt_ref.tolist()}，按 e 执行放置"
                    )

            # 可视化
            disp = color.copy()
            if current_pick_uv:
                cv2.circle(disp, current_pick_uv, 5, (0, 0, 255), -1)
            if predicted_place_uv and i % 2 == 1:
                cv2.circle(disp, predicted_place_uv, 5, (0, 0, 255), -1)
            if attachment_uvs:
                for uv in attachment_uvs:
                    cv2.circle(disp, uv, 5, (255, 0, 0), -1)

            curr_prompt = pick_prompt if i % 2 == 0 else current_place_prompt
            prompt_lines = textwrap.wrap(f"prompt: {curr_prompt}", width=32)
            for idx, line in enumerate(prompt_lines):
                cv2.putText(
                    disp,
                    line,
                    (10, 25 + idx * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                current_pick_uv = None
                current_pick_ref = None
                predicted_place_uv = None
                place_pt_ref = None
                attachment_uvs = None
                current_place_prompt = place_prompt_single
                continue

            if key == ord("e"):
                if i % 2 == 0 and current_pick_ref is not None:
                    print(
                        f"执行 pick 点 {current_pick_uv} -> {current_pick_ref.tolist()}"
                    )
                    seq = build_pick_cup_sequence(current_pick_ref, arm="left")
                    for act in seq:
                        arx.step(act)
                    current_pick_uv = None
                    current_pick_ref = None
                    i += 1
                elif i % 2 == 1 and place_pt_ref is not None:
                    seq = build_place_cup_sequence(place_pt_ref, arm="left")
                    for act in seq:
                        arx.step(act)
                    arx._go_to_initial_pose()
                    predicted_place_uv = None
                    place_pt_ref = None
                    attachment_uvs = None
                    i += 1

            if key in (27, ord("q")):
                break
    finally:
        window_node.destroy_node()
        cv2.destroyAllWindows()
        arx.close()


if __name__ == "__main__":
    main()
