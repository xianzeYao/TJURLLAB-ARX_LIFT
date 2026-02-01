"""
基于 point2pos_live4demo.py --predict 的多点抓取版本。

保持偶数轮为 pick、奇数轮为 place 的交替逻辑；
唯一差异：pick prompt 调用返回多个像素点，按顺序依次执行每个点的抓取。
"""
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
        super().__init__("place_planning")
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
    """过滤深度无效的像素点并转换为 ref 坐标系下的 3D 点。"""
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


def _extract_targets(raw: Optional[str]) -> List[str]:
    """从大模型原始回复中提取 target 文本（不含坐标）。"""
    if not raw:
        return []
    # 去掉代码块包裹
    raw_clean = re.sub(r"```(?:json|python)?\n?(.*?)\n?```",
                       r"\1", raw, flags=re.DOTALL)
    # 简单 JSON/列表解析
    try:
        data = ast.literal_eval(raw_clean)
        targets: List[str] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "target" in item:
                    targets.append(str(item["target"]))
        elif isinstance(data, dict) and "target" in data:
            targets.append(str(data["target"]))
        if targets:
            return targets
    except Exception:
        pass
    # 退化：regex 抓取 "target": "xxx"
    matches = re.findall(r'"target"\s*:\s*"([^"]+)"', raw_clean)
    return matches


def main():
    parser = argparse.ArgumentParser(
        description="多点抓取 + 放置演示 (偶数轮 pick、奇数轮 place)"
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="使用 ER1.5 预测像素点（默认模式）",
    )
    args = parser.parse_args()

    if not args.predict:
        print("当前脚本仅支持 --predict 模式")
        return

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
    time.sleep(3.0)
    arx.reset()
    arx.step_lift(17.0)

    window_node = FrameBuffer()
    K = load_intrinsics()
    T_cam2ref = load_cam2ref()

    # pick 队列：预测一次后缓存多个点，依次 pick -> place -> pick -> place ...
    pick_uv_queue: List[Tuple[int, int]] = []
    pick_ref_queue: List[np.ndarray] = []
    current_pick_uv: Optional[Tuple[int, int]] = None
    current_pick_ref: Optional[np.ndarray] = None
    pick_answer_text: Optional[str] = None
    pick_targets: List[str] = []
    predicted_place_uv: Optional[Tuple[int, int]] = None
    place_pt_ref: Optional[np.ndarray] = None
    attachment_uvs: Optional[List[Tuple[int, int]]] = None

    # pick_prompt = """Given an RGB image, output the minimal sequence of cup pick actions required to finally pick the red cup.

    #                     Rules:
    #                     - A cup can only be picked if no other cup is placed on top of it.
    #                     - A cup that is partially or fully occluded by another cup is NOT pickable.
    #                     - If the goal cup is not immediately pickable, you must first pick the cups that block it.
    #                     - Cups that are on the same level and not stacked on top of each other do not affect each other’s pickability.
    #                     - The order of pick actions is the actual execution order.

    #                     For each pick action, provide one valid 2D pick point on the visible surface of the cup.

    #                     Output ONLY a JSON array in execution order:
    #                     [
    #                     {
    #                         "target": "cup description",
    #                         "point_2d": [x, y]
    #                     }
    #                     ]"""
    pick_prompt = """Given an RGB image, output the minimal sequence of cup pick actions required to finally pick the red cup.

                        Rules:
                        - A cup can only be picked if no other cup is placed on top of it.
                        - A cup that is partially or fully occluded by another cup is NOT pickable.
                        - If the goal cup is not immediately pickable, you must first pick the cups that block it.
                        - The order of pick actions is the actual execution order.

                        For each pick action, provide one valid 2D pick point on the middle center of the cup.

                        Output ONLY a JSON array in execution order:
                        [
                        {
                            "point_2d": [x, y]
                        }
                        ]"""
    place_prompt = "the smaller number at the center of a white round coaster"

    try:
        win = "place_planning_predict"
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

            if (need_pick and current_pick_ref is None) or (need_place and predicted_place_uv is None):
                if i != 0:
                    arx.step_lift(13.0)

                if need_pick:
                    if not pick_ref_queue:
                        raw_result = predict_multi_points_from_rgb(
                            color,
                            text_prompt="",
                            all_prompt=pick_prompt,
                            return_raw=True,
                        )
                        if isinstance(raw_result, tuple):
                            raw_uvs, pick_answer_text = raw_result
                        else:
                            raw_uvs = raw_result
                            pick_answer_text = None
                        pick_targets = _extract_targets(pick_answer_text)
                        print(f"pick 模型回答: {pick_answer_text}")
                        uv_ints = [(int(round(u)), int(round(v)))
                                   for (u, v) in raw_uvs]
                        valid_uvs, valid_refs = _filter_valid_points(
                            uv_ints, depth, K, T_cam2ref
                        )
                        # 只保留前 4 个点，便于一层1个、二层2个、三层1个的用法
                        valid_uvs = valid_uvs[:5]
                        valid_refs = valid_refs[:5]
                        if not valid_uvs:
                            print("未得到有效 pick 点，按 r 重新预测")
                            continue
                        pick_uv_queue.extend(valid_uvs)
                        pick_ref_queue.extend(valid_refs)
                        print(
                            f"缓存 {len(valid_uvs)} 个 pick 点(最多4个)，将按一抓一放执行；q/ESC 退出")
                    # 取队首作为当前 pick
                    current_pick_uv = pick_uv_queue[0]
                    current_pick_ref = pick_ref_queue[0]
                else:
                    # place 与原逻辑一致
                    color_latest, depth_latest = window_node.get_frames()
                    if color_latest is None or depth_latest is None:
                        cv2.waitKey(1)
                        continue
                    if "attachment" in place_prompt.lower():
                        prompts = [
                            "the right and top attachment of the left cup",
                            "the left and top attachemnt of the right cup",
                        ]
                        uv_list = []
                        pt_refs = []
                        invalid_depth = False
                        for sub_prompt in prompts:
                            sub_u, sub_v = predict_point_from_rgb(
                                color_latest,
                                text_prompt=sub_prompt,
                                assume_bgr=False
                            )
                            uv = (int(round(sub_u)), int(round(sub_v)))
                            raw_depth = depth_latest[uv[1], uv[0]]
                            if np.isnan(raw_depth) or raw_depth == 0:
                                print(
                                    f"预测像素 {uv} 深度无效({raw_depth})，按 r 重新预测"
                                )
                                invalid_depth = True
                                break
                            uv_list.append(uv)
                            pt_refs.append(
                                pixel_to_ref_point(
                                    uv, depth_latest, K, T_cam2ref)
                            )
                        if invalid_depth:
                            predicted_place_uv = None
                            place_pt_ref = None
                            attachment_uvs = None
                            continue
                        attachment_uvs = uv_list
                        predicted_place_uv = tuple(
                            np.mean(np.array(uv_list),
                                    axis=0).round().astype(int)
                        )
                        place_pt_ref = np.mean(np.array(pt_refs), axis=0)
                    else:
                        u, v = predict_point_from_rgb(
                            color_latest,
                            text_prompt=place_prompt,
                            assume_bgr=False
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
                    print(
                        f"place 预测像素 {predicted_place_uv} -> ref {place_pt_ref.tolist()}，按 e 执行放置"
                    )

            # 可视化
            disp = color.copy()
            if pick_uv_queue:
                for idx_uv, uv in enumerate(pick_uv_queue):
                    color_draw = (0, 0, 255)  # 统一颜色，靠序号区分
                    cv2.circle(disp, uv, 5, color_draw, -1)
                    # 标出队列序号，便于识别目标
                    cv2.putText(
                        disp,
                        f"{idx_uv+1}",
                        (uv[0] + 6, uv[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color_draw,
                        2,
                        cv2.LINE_AA,
                    )
            if predicted_place_uv and i % 2 == 1:
                cv2.circle(disp, predicted_place_uv, 5, (0, 0, 255), -1)
            if attachment_uvs:
                for uv in attachment_uvs:
                    cv2.circle(disp, uv, 5, (255, 0, 0), -1)
            curr_prompt = (
                "\n".join(pick_targets) if pick_targets else (
                    pick_answer_text or pick_prompt)
            ) if i % 2 == 0 else place_prompt
            prompt_lines = textwrap.wrap(f"target: {curr_prompt}", width=32)
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
                pick_uv_queue = []
                pick_ref_queue = []
                current_pick_uv = None
                current_pick_ref = None
                pick_answer_text = None
                pick_targets = []
                predicted_place_uv = None
                place_pt_ref = None
                attachment_uvs = None
                continue

            if key == ord("p"):
                new_p = input("输入新的 prompt (留空保持当前): ").strip()
                if new_p:
                    if i % 2 == 0:
                        pick_prompt = new_p
                    else:
                        place_prompt = new_p
                pick_uv_queue = []
                pick_ref_queue = []
                current_pick_uv = None
                current_pick_ref = None
                pick_answer_text = None
                pick_targets = []
                predicted_place_uv = None
                place_pt_ref = None
                attachment_uvs = None
                continue

            if key == ord("e"):
                if i % 2 == 0 and current_pick_ref is not None:
                    print(
                        f"执行 pick 点 {current_pick_uv} -> {current_pick_ref.tolist()}"
                    )
                    seq = build_pick_cup_sequence(current_pick_ref)
                    for act in seq:
                        arx.step(act)
                    # 弹出已执行的点，切到 place 轮
                    pick_uv_queue.pop(0)
                    pick_ref_queue.pop(0)
                    if not pick_uv_queue:
                        pick_answer_text = None
                        pick_targets = []
                    current_pick_uv = None
                    current_pick_ref = None
                    i += 1
                elif i % 2 == 1 and place_pt_ref is not None:
                    seq = build_place_cup_sequence(place_pt_ref)
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
