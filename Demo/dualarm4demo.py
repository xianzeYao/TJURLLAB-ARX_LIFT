from __future__ import annotations

"""
双臂打点 demo：复用 ARXRobotEnv 自带相机订阅，左右两个窗口并列显示。
- 左窗点击 -> 按左臂外参计算 ref 3D 点并打印
- 右窗点击 -> 按右臂外参计算 ref 3D 点并打印

参数与旧版一致：支持 --debug / --predict（预测模式仅计算并显示像素/坐标，不执行动作）。
"""

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

import sys
sys.path.append("../ARX_Realenv/ROS2")  # noqa

from arx_ros2_env import ARXRobotEnv  # noqa
from arx_pointing import predict_point_from_rgb  # noqa
from point2pos_utils import load_intrinsics, load_cam2ref, pixel_to_ref_point  # noqa
from pick_place_cup_motion import build_pick_cup_sequence, build_place_cup_sequence  # noqa
from pick_place_straw_motion import build_pick_straw_sequence, build_place_straw_sequence  # noqa

WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_INTR = WORKSPACE / "ARX_Realenv/Tools/instrinsics_camerah.json"
DEFAULT_EXT_LEFT = WORKSPACE / "ARX_Realenv/Tools/final_extrinsics_cam_h_left.json"
DEFAULT_EXT_RIGHT = WORKSPACE / "ARX_Realenv/Tools/final_extrinsics_cam_h_right.json"


def main():
    parser = argparse.ArgumentParser(
        description="双臂打点（左右窗口分开点击），使用 ARX 内部相机订阅。")
    parser.add_argument("--debug", action="store_true",
                        help="手动点击，输出/执行：左臂杯子、右臂吸管")
    parser.add_argument("--predict", action="store_true",
                        help="ER1.5 预测像素，输出/执行：左臂杯子、右臂吸管")
    args = parser.parse_args()

    if not (args.debug or args.predict):
        print("未指定模式，请添加参数：--debug / --predict")
        return

    K = load_intrinsics(DEFAULT_INTR)
    T_left = load_cam2ref(DEFAULT_EXT_LEFT)
    T_right = load_cam2ref(DEFAULT_EXT_RIGHT)

    arx = ARXRobotEnv(duration_per_step=1.0/20.0,
                      min_steps_per_action=60,
                      min_steps_gripper=20,
                      max_v_xyz=0.1,
                      max_v_rpy=0.1,
                      camera_type="all",
                      camera_view=("camera_h",),
                      img_size=(640, 480))
    arx.reset()
    arx.step_lift(15.0)

    if args.debug:
        win_left = "dualarm_left"
        win_right = "dualarm_right"
        cv2.namedWindow(win_left, cv2.WINDOW_NORMAL)
        cv2.namedWindow(win_right, cv2.WINDOW_NORMAL)

        state = {
            "left_pick": None,    # type: Optional[Tuple[int, int]]
            "left_place": None,
            "right_pick": None,
            "right_place": None,
            "left_pick_ref": None,   # type: Optional[np.ndarray]
            "left_place_ref": None,
            "right_pick_ref": None,
            "right_place_ref": None,
        }

        def on_mouse_left(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if state["left_pick"] is None:
                    state["left_pick"] = (x, y)
                else:
                    state["left_place"] = (x, y)

        def on_mouse_right(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if state["right_pick"] is None:
                    state["right_pick"] = (x, y)
                else:
                    state["right_place"] = (x, y)

        cv2.setMouseCallback(win_left, on_mouse_left)
        cv2.setMouseCallback(win_right, on_mouse_right)

        try:
            while True:
                frames = arx.node.get_camera(
                    target_size=(640, 480), return_status=False)
                color = frames.get("camera_h_color")
                depth = frames.get("camera_h_aligned_depth_to_color")
                if color is None or depth is None:
                    cv2.waitKey(1)
                    continue

                disp_l = color.copy()
                disp_r = color.copy()
                if state["left_pick"] is not None:
                    cv2.circle(disp_l, state["left_pick"], 5, (0, 0, 255), -1)
                if state["left_place"] is not None:
                    cv2.circle(disp_l, state["left_place"], 5, (255, 0, 0), -1)
                if state["right_pick"] is not None:
                    cv2.circle(disp_r, state["right_pick"], 5, (0, 0, 255), -1)
                if state["right_place"] is not None:
                    cv2.circle(
                        disp_r, state["right_place"], 5, (255, 0, 0), -1)

                cv2.putText(disp_l, "Left: cup (pick red, place blue)",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(disp_r, "Right: straw (pick red, place blue)",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow(win_left, disp_l)
                cv2.imshow(win_right, disp_r)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    state["left_pick"] = state["left_place"] = None
                    state["right_pick"] = state["right_place"] = None
                    state["left_pick_ref"] = state["left_place_ref"] = None
                    state["right_pick_ref"] = state["right_place_ref"] = None
                    continue
                if key == ord("e"):
                    frames = arx.node.get_camera(
                        target_size=(640, 480), return_status=False)
                    color = frames.get("camera_h_color")
                    depth = frames.get("camera_h_aligned_depth_to_color")
                    if color is None or depth is None:
                        print("无有效图像，无法执行。")
                        continue

                    def _maybe_cache(px_key: str, ref_key: str, T):
                        px = state[px_key]
                        if px is None or state[ref_key] is not None:
                            return
                        u, v = px
                        raw = depth[v, u]
                        if not np.isfinite(raw) or raw <= 0:
                            raise ValueError(f"深度无效: {raw} @ {px}")
                        state[ref_key] = pixel_to_ref_point(px, depth, K, T)

                    _maybe_cache("left_pick", "left_pick_ref", T_left)
                    _maybe_cache("left_place", "left_place_ref", T_left)
                    _maybe_cache("right_pick", "right_pick_ref", T_right)
                    _maybe_cache("right_place", "right_place_ref", T_right)

                    if state["left_pick_ref"] is not None and state["left_place_ref"] is not None:
                        lp_pick = state["left_pick_ref"]
                        lp_place = state["left_place_ref"]
                        for act in build_pick_cup_sequence(lp_pick):
                            arx.step(act)
                        for act in build_place_cup_sequence(lp_place):
                            arx.step(act)
                        arx._go_to_initial_pose()
                        print(
                            f"左臂执行完毕 pick={lp_pick.tolist()} place={lp_place.tolist()}")
                    else:
                        print("左臂缺少 pick/place 点，跳过。")

                    if state["right_pick_ref"] is not None and state["right_place_ref"] is not None:
                        rp_pick = state["right_pick_ref"]
                        rp_place = state["right_place_ref"]
                        for act in build_pick_straw_sequence(rp_pick):
                            arx.step(act)
                        for act in build_place_straw_sequence(rp_place):
                            arx.step(act)
                        arx._go_to_initial_pose()
                        print(
                            f"右臂执行完毕 pick={rp_pick.tolist()} place={rp_place.tolist()}")
                    else:
                        print("右臂缺少 pick/place 点，跳过。")
        finally:
            arx._go_to_initial_pose()
            cv2.destroyAllWindows()
            arx.close()

    elif args.predict:
        win_left = "dualarm_left_predict"
        win_right = "dualarm_right_predict"
        cv2.namedWindow(win_left, cv2.WINDOW_NORMAL)
        cv2.namedWindow(win_right, cv2.WINDOW_NORMAL)

        predicted_px: Optional[Tuple[int, int]] = None

        try:
            while True:
                frames = arx.node.get_camera(
                    target_size=(640, 480), return_status=False)
                color = frames.get("camera_h_color")
                depth = frames.get("camera_h_aligned_depth_to_color")
                if color is None or depth is None:
                    cv2.waitKey(1)
                    continue

                if predicted_px is None:
                    u, v = predict_point_from_rgb(
                        color,
                        text_prompt="choose a grasp point",
                        assume_bgr=False,
                    )
                    predicted_px = (int(round(u)), int(round(v)))
                    raw_depth = depth[predicted_px[1], predicted_px[0]]
                    if np.isnan(raw_depth) or raw_depth == 0:
                        print(
                            f"预测像素 {predicted_px} 深度无效({raw_depth})，按 r 重新预测")
                        predicted_px = None
                        continue
                    left_ref = pixel_to_ref_point(
                        predicted_px, depth, K, T_left)
                    right_ref = pixel_to_ref_point(
                        predicted_px, depth, K, T_right)
                    print(
                        f"预测像素 {predicted_px} -> 左臂 ref 3D: {left_ref.tolist()} | 右臂 ref 3D: {right_ref.tolist()}，按 r 重新预测，q/ESC 退出")

                disp_left = color.copy()
                disp_right = color.copy()
                if predicted_px is not None:
                    cv2.circle(disp_left, predicted_px, 5, (0, 0, 255), -1)
                    cv2.circle(disp_right, predicted_px, 5, (255, 0, 0), -1)
                cv2.imshow(win_left, disp_left)
                cv2.imshow(win_right, disp_right)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("r"):
                    predicted_px = None
                    continue
                if key in (27, ord("q")):
                    break
        finally:
            arx._go_to_initial_pose()
            cv2.destroyAllWindows()
            arx.close()


if __name__ == "__main__":
    main()
