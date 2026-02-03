from __future__ import annotations

import argparse
import textwrap
import threading
import time
import ast
import re
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
    def __init__(self):
        super().__init__("place_planning2")
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


def _extract_numbered_sentences(raw: Optional[str]) -> List[str]:
    """提取形如 '1. xxx' / '2) xxx' / '3- xxx' 的编号句子（行内行间都可）。"""
    if not raw:
        return []
    # 去掉代码块包裹
    raw_clean = re.sub(
        r"```(?:json|python)?\n?(.*?)\n?```", r"\1", raw, flags=re.DOTALL
    )
    steps: List[str] = []

    # 行内 / 行间匹配：1. xxx 2) yyy 3- zzz
    inline_matches = re.finditer(
        r"(\d+)[\.\)\-]\s*(.+?)(?=(?:\d+[\.\)\-])|$)",
        raw_clean,
        flags=re.DOTALL,
    )
    for m in inline_matches:
        steps.append(m.group(2).strip())

    # 去重保持顺序
    seen = set()
    uniq_steps = []
    for s in steps:
        if s not in seen:
            seen.add(s)
            uniq_steps.append(s)
    return uniq_steps


def do_replan(color_img: np.ndarray, pick_prompt: str) -> List[str]:
    raw_result = predict_multi_points_from_rgb(
        color_img,
        text_prompt="",
        all_prompt=pick_prompt,
        assume_bgr=False,
        return_raw=True,
    )
    if isinstance(raw_result, tuple):
        _, pick_answer_text = raw_result
    else:
        pick_answer_text = None

    pick_plan = _extract_numbered_sentences(pick_answer_text)
    if not pick_plan:
        do_replan(color_img=color_img, pick_prompt=pick_prompt)  # 递归重试
    return pick_plan


def main():
   
    try:
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
        arx.step_lift(17.0)

        window_node = FrameBuffer()
        K = load_intrinsics()
        T_cam2ref = load_cam2ref()

        planned = True
        step_idx = 0
        plan_steps: List[str] = []

        pick_prompt = (
            "Current Goal is: pick the green cup.I need to pick up the cups from top to the green cup."
            "What is the picking plan steps to finish the goal?"
        )
        place_prompt = "the smaller number at the center of a white round coaster"

        plan_steps: List[str] = []
        confirm_win = "Planning Step"
        cv2.namedWindow(confirm_win, cv2.WINDOW_NORMAL)

        while rclpy.ok():
            rclpy.spin_once(window_node, timeout_sec=0.1)
            color, _ = window_node.get_frames()
            if color is None:
                continue
            # 调用 VLM 生成步骤
            current_plan = do_replan(color, pick_prompt)

            # --- 可视化：在图像上打印出规划结果供确认 ---
            vis_img = color.copy()
            print(f"生成的规划结果 ({len(current_plan)} 步):")

            # 在画面左上角绘制提示背景
            if not current_plan:
                print("未生成有效步骤！")
            else:
                for i, step in enumerate(current_plan):
                    print(f"  {i+1}. {step}")
                    # 将文字绘制在图片上
            cv2.imshow(confirm_win, vis_img)
            print("按'y' 确认, 'r' 重试, 'q' 退出")
            key = cv2.waitKey(0)
            if key == ord('y') and current_plan:
                plan_steps = current_plan
                print("规划已确认，进入执行模式。")
                break
            elif key == ord('r'):
                print("重新尝试规划...")
                continue
            elif key == ord('q'):
                print("退出程序。")
                planned = False
                break
        if planned:
            cv2.destroyWindow(confirm_win)
            win = "place_planning_predict2"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        while rclpy.ok() and planned:
            rclpy.spin_once(window_node, timeout_sec=0.1)
            color, depth = window_node.get_frames()
            if color is None or depth is None:
                continue
            color = color.copy()
            if step_idx != 0:
                arx.step_lift(13.0)
            if step_idx % 2 == 0:
                u, v = predict_point_from_rgb(
                    color,
                    text_prompt=plan_steps[step_idx // 2],
                    assume_bgr=False,
                )
                predicted_px = (int(round(u)), int(round(v)))
                cv2.circle(
                    color,
                    center=(predicted_px[0], predicted_px[1]),
                    radius=5,
                    color=(0, 0, 255),
                    thickness=-1,
                )
                cv2.putText(
                    color,
                    f"Step {step_idx + 1}: {plan_steps[step_idx // 2]}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                pt_ref = pixel_to_ref_point(
                    predicted_px, depth, K, T_cam2ref
                )
            else:
                u, v = predict_point_from_rgb(
                    color,
                    text_prompt=place_prompt,
                    assume_bgr=False,
                )
                predicted_px = (int(round(u)), int(round(v)))
                cv2.circle(
                    color,
                    center=(predicted_px[0], predicted_px[1]),
                    radius=5,
                    color=(0, 0, 255),
                    thickness=-1,
                )
                cv2.putText(
                    color,
                    f"{place_prompt}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                pt_ref = pixel_to_ref_point(
                    predicted_px, depth, K, T_cam2ref
                )
            cv2.imshow(win, color)

            key = cv2.waitKey(0)

            if key == ord("r"):
                continue

            if key == ord("e"):
                if step_idx % 2 == 0:
                    action_seq = build_pick_cup_sequence(pt_ref)
                    for act in action_seq:
                        arx.step(act)
                else:
                    action_seq = build_place_cup_sequence(pt_ref)
                    for act in action_seq:
                        arx.step(act)
                    arx._go_to_initial_pose()
                step_idx += 1
            if key == ord("q"):
                break
    finally:
        window_node.destroy_node()
        cv2.destroyAllWindows()
        arx.close()

if __name__ == "__main__":
    main()
