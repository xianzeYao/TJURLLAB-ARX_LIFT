"""
实时订阅彩色 + 对齐深度（aligned_depth_to_color），鼠标点击像素输出基坐标系下的 3D 点。

默认 topic:
- 彩色: /camera_h_namespace/camera_h/color/image_rect_raw
- 对齐深度: /camera_h_namespace/camera_h/aligned_depth_to_color/image_raw

标定文件:
- 内参: instrinsics_right4camerah.json
- 外参: final_extrinsics_cam_h_right.json (T_cam2ref)

运行:
python3 -m arx_realenv.point2pos_live
左键点击弹窗画面即可在终端打印 3D 点 (基坐标系)。
"""
from __future__ import annotations
import time
from sensor_msgs.msg import Image
from rclpy.node import Node
from cv_bridge import CvBridge
import rclpy
import numpy as np
import cv2
from typing import Optional, Tuple
import threading
import argparse
import textwrap
from motion_pick_place_straw import *
from motion_pick_place_cup import *
from point2pos_utils import (
    load_intrinsics,
    load_cam2ref,
    pixel_to_ref_point,
)
from arx_pointing import predict_point_from_rgb

import sys
sys.path.append("../ARX_Realenv/ROS2")  # noqa

from arx_ros2_env import ARXRobotEnv  # noqa

COLOR_TOPIC = "/camera_h_namespace/camera_h/color/image_rect_raw"
DEPTH_TOPIC = "/camera_h_namespace/camera_h/aligned_depth_to_color/image_raw"


class FrameBuffer(Node):
    def __init__(self):
        super().__init__("point2pos_live")
        self.bridge = CvBridge()
        self.latest_color: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.create_subscription(
            Image, COLOR_TOPIC, self._on_color, 5)
        self.create_subscription(
            Image, DEPTH_TOPIC, self._on_depth, 5)
        self.get_logger().info(
            f"订阅彩色: {COLOR_TOPIC}, 对齐深度: {DEPTH_TOPIC}")

    def _on_color(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.lock:
            self.latest_color = img

    def _on_depth(self, msg: Image):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        # Realsense aligned depth 一般为 uint16(mm) 或 float32(m)
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32)
        with self.lock:
            self.latest_depth = depth

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self.lock:
            return self.latest_color, self.latest_depth


def main():
    parser = argparse.ArgumentParser(description="打点实现基坐标系下的 3D 点获取以及机械臂控制")
    parser.add_argument("--debug", action="store_true",
                        help="启用调试模式,手动打点不执行机械臂动作，输出转换后的xyz坐标")
    parser.add_argument("--predict", action="store_true",
                        help="使用ER1.5预测像素点")
    args = parser.parse_args()

    if not (args.debug or args.predict):
        print("未指定模式，请添加参数：--debug / --predict")
        return

    arx = ARXRobotEnv(duration_per_step=1.0/20.0,  # 就是插值里一步的时间，20Hz也就是0.05s
                      min_steps=20,
                      max_v_xyz=0.15, max_a_xyz=0.15,
                      max_v_rpy=0.3, max_a_rpy=1.00,
                      camera_type="all",
                      camera_view=("camera_h",),
                      img_size=(640, 480))
    time.sleep(2.0)  # 等待环境初始化完成
    arx.reset()
    arx.step_lift(13.0)
    if args.debug:
        window_node = FrameBuffer()
        K = load_intrinsics()
        T_cam2ref = load_cam2ref(side="left")

        clicked: Optional[Tuple[int, int]] = None

        def on_mouse(event, x, y, flags, param):
            nonlocal clicked
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and clicked is not None:
                # 允许长按拖动时跟随显示
                clicked = (x, y)

        win = "point2pos_live"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, on_mouse)

        while rclpy.ok():
            rclpy.spin_once(window_node, timeout_sec=0.01)
            color, depth = window_node.get_frames()
            if color is None or depth is None:
                cv2.waitKey(1)
                continue
            disp = color.copy()
            if clicked is not None:
                cv2.circle(disp, clicked, 3,  (0, 0, 255), -1)
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):  # ESC or q to quit
                break

            if clicked is not None:
                pt_ref: Optional[np.ndarray] = pixel_to_ref_point(
                    clicked, depth, K, T_cam2ref)
                print(f"点击 {clicked} -> 基坐标系 3D 点: {pt_ref.tolist()}")
                clicked = None
        arx._go_to_initial_pose()
        window_node.destroy_node()
        cv2.destroyAllWindows()

        arx.close()
    elif args.predict:
        window_node = FrameBuffer()
        K = load_intrinsics()
        T_cam2ref = load_cam2ref(side="left")
        pt_ref = None
        predicted_px = None
        executed = False
        attachment_uvs = None
        pick_prompt = "a middle and center position of the yellow cup"
        place_prompt = "the center of the highest cups' bottom"
        try:
            win = "point2pos_predict"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            i = 0
            while rclpy.ok():
                # 交替pick and place 偶数pick 奇数place

                rclpy.spin_once(window_node, timeout_sec=0.05)

                color, depth = window_node.get_frames()
                if color is None or depth is None:
                    cv2.waitKey(1)
                    continue

                if predicted_px is None:
                    if i != 0:
                        arx.step_lift(17.0)
                    if i % 2 == 0:
                        prompt = pick_prompt
                        u, v = predict_point_from_rgb(
                            color,
                            text_prompt=prompt,
                            assume_bgr=False,
                        )
                        attachment_uvs = None
                        predicted_px = (int(round(u)), int(round(v)))
                        raw_depth = depth[predicted_px[1], predicted_px[0]]
                        if np.isnan(raw_depth) or raw_depth == 0:
                            print(
                                f"预测像素 {predicted_px} 深度无效({raw_depth})，按 r 重新预测")
                            predicted_px = None
                            pt_ref = None
                            executed = False
                            continue
                        pt_ref = pixel_to_ref_point(
                            predicted_px, depth, K, T_cam2ref)
                    else:
                        # 取最新帧再做place预测，避免显示停滞
                        color, depth = window_node.get_frames()
                        if color is None or depth is None:
                            cv2.waitKey(1)
                            continue
                        prompt = place_prompt
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
                                    color,
                                    text_prompt=sub_prompt,
                                    assume_bgr=False,
                                )
                                uv = (int(round(sub_u)), int(round(sub_v)))
                                raw_depth = depth[uv[1], uv[0]]
                                if np.isnan(raw_depth) or raw_depth == 0:
                                    print(
                                        f"预测像素 {uv} 深度无效({raw_depth})，按 r 重新预测")
                                    invalid_depth = True
                                    break
                                uv_list.append(uv)
                                pt_refs.append(pixel_to_ref_point(
                                    uv, depth, K, T_cam2ref))
                            if invalid_depth:
                                predicted_px = None
                                pt_ref = None
                                executed = False
                                attachment_uvs = None
                                continue
                            attachment_uvs = uv_list
                            predicted_px = tuple(
                                np.mean(np.array(uv_list), axis=0).round().astype(int))
                            pt_ref = np.mean(np.array(pt_refs), axis=0)
                        else:
                            u, v = predict_point_from_rgb(
                                color,
                                text_prompt=prompt,
                                assume_bgr=False,
                            )
                            attachment_uvs = None
                            predicted_px = (int(round(u)), int(round(v)))
                            raw_depth = depth[predicted_px[1], predicted_px[0]]
                            if np.isnan(raw_depth) or raw_depth == 0:
                                print(
                                    f"预测像素 {predicted_px} 深度无效({raw_depth})，按 r 重新预测")
                                predicted_px = None
                                pt_ref = None
                                executed = False
                                attachment_uvs = None
                                continue
                            pt_ref = pixel_to_ref_point(
                                predicted_px, depth, K, T_cam2ref)
                    if pt_ref is None:
                        predicted_px = None
                        executed = False
                        attachment_uvs = None
                        continue
                    executed = False
                    print(
                        f"预测像素 {predicted_px} -> 基坐标系 3D 点: {pt_ref.tolist()}，按 e 执行抓取，q/ESC 退出")

                disp = color.copy()
                if predicted_px is not None:
                    cv2.circle(disp, predicted_px, 3,  (0, 0, 255), -1)
                if attachment_uvs:
                    for uv in attachment_uvs:
                        cv2.circle(disp, uv, 3,  (255, 0, 0), -1)
                curr_prompt = pick_prompt if i % 2 == 0 else place_prompt
                prompt_lines = textwrap.wrap(
                    f"prompt: {curr_prompt}", width=32)
                for idx, line in enumerate(prompt_lines):
                    cv2.putText(disp, line, (10, 25 + idx * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow(win, disp)
                key = cv2.waitKey(1) & 0xFF
                # r 键刷新预测
                if key == ord("r"):
                    predicted_px = None
                    attachment_uvs = None
                    continue
                if key == ord("p"):
                    new_p = input("输入新的 prompt (留空保持当前): ").strip()
                    if new_p:
                        if i % 2 == 0:
                            pick_prompt = new_p
                        else:
                            place_prompt = new_p
                    predicted_px = None
                    pt_ref = None
                    executed = False
                    attachment_uvs = None
                    continue
                if key == ord("e") and pt_ref is not None and not executed:
                    seq = build_pick_cup_sequence(
                        pt_ref, arm="left") if i % 2 == 0 else build_place_cup_sequence(pt_ref, arm="left")
                    for act in seq:
                        arx.step(act)
                    if i % 2 == 1:
                        arx._go_to_initial_pose()
                    # 清空预测以便下一次循环重新打点，并切换模式
                    predicted_px = None
                    pt_ref = None
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
