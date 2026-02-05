from __future__ import annotations

"""
双臂打点 demo：复用 ARXRobotEnv 自带相机订阅，左右两个窗口并列显示。
- 左窗点击 -> 按左臂外参计算 ref 3D 点并打印
- 右窗点击 -> 按右臂外参计算 ref 3D 点并打印

参数与旧版一致：支持 --debug / --predict（预测模式仅计算并显示像素/坐标，不执行动作）。
"""

import argparse
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
    T_left, T_right = load_cam2ref()

    arx = ARXRobotEnv(duration_per_step=1.0/20.0,
                      min_steps_per_action=60,
                      min_steps_gripper=20,
                      max_v_xyz=0.1,
                      max_v_rpy=0.1,
                      camera_type="all",
                      camera_view=("camera_h",),
                      img_size=(640, 480))
    arx.reset()
    arx.step_lift(14.0)

    if args.debug:
        windows = {
            "left": "dualarm_left",
            "right": "dualarm_right",
        }
        views = {
            "left": {"title": "Left ", "T": T_left},
            "right": {"title": "Right", "T": T_right},
        }
        state = {
            "left": {"pick_px": None, "place_px": None, "pick_ref": None, "place_ref": None},
            "right": {"pick_px": None, "place_px": None, "pick_ref": None, "place_ref": None},
        }

        for win in windows.values():
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        def clear_state():
            for arm_state in state.values():
                arm_state["pick_px"] = None
                arm_state["place_px"] = None
                arm_state["pick_ref"] = None
                arm_state["place_ref"] = None

        def on_mouse(arm: str, event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            arm_state = state[arm]
            if arm_state["pick_px"] is None:
                arm_state["pick_px"] = (x, y)
            else:
                arm_state["place_px"] = (x, y)

        cv2.setMouseCallback(
            windows["left"],
            lambda event, x, y, flags, param: on_mouse(
                "left", event, x, y, flags, param),
        )
        cv2.setMouseCallback(
            windows["right"],
            lambda event, x, y, flags, param: on_mouse(
                "right", event, x, y, flags, param),
        )

        def draw_view(color: np.ndarray, arm: str) -> np.ndarray:
            disp = color.copy()
            arm_state = state[arm]
            if arm_state["pick_px"] is not None:
                cv2.circle(disp, arm_state["pick_px"], 5, (0, 0, 255), -1)
            if arm_state["place_px"] is not None:
                cv2.circle(disp, arm_state["place_px"], 5, (255, 0, 0), -1)
            cv2.putText(
                disp,
                views[arm]["title"],
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            return disp

        def cache_ref_points(depth: np.ndarray):
            for arm in ("left", "right"):
                T = views[arm]["T"]
                arm_state = state[arm]
                for px_key, ref_key in (("pick_px", "pick_ref"), ("place_px", "place_ref")):
                    px = arm_state[px_key]
                    if px is None or arm_state[ref_key] is not None:
                        continue
                    u, v = px
                    raw = depth[v, u]
                    if not np.isfinite(raw) or raw <= 0:
                        raise ValueError(f"{arm} {px_key} 深度无效: {raw} @ {px}")
                    arm_state[ref_key] = pixel_to_ref_point(px, depth, K, T)

        def execute_arm(arm: str):
            arm_state = state[arm]
            pick_ref = arm_state["pick_ref"]
            place_ref = arm_state["place_ref"]
            if pick_ref is None or place_ref is None:
                print(f"{arm} 臂缺少 pick/place 点，跳过。")
                return

            if arm == "left":
                pick_seq = build_pick_cup_sequence(pick_ref, arm="left")
                place_seq = build_place_cup_sequence(place_ref, arm="left")
                arm_desc = "左臂(杯子)"
            else:
                pick_seq = build_pick_cup_sequence(pick_ref, arm="right")
                place_seq = build_place_cup_sequence(place_ref, arm="right")
                arm_desc = "右臂(杯子)"

            for act in pick_seq:
                arx.step(act)
            for act in place_seq:
                arx.step(act)
            arx._go_to_initial_pose()
            print(
                f"{arm_desc}执行完毕 pick={pick_ref.tolist()} place={place_ref.tolist()}")

        try:
            while True:
                frames = arx.node.get_camera(
                    target_size=(640, 480), return_status=False)
                color = frames.get("camera_h_color")
                depth = frames.get("camera_h_aligned_depth_to_color")
                if color is None or depth is None:
                    cv2.waitKey(1)
                    continue

                cv2.imshow(windows["left"], draw_view(color, "left"))
                cv2.imshow(windows["right"], draw_view(color, "right"))
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    clear_state()
                    continue
                if key == ord("e"):
                    try:
                        cache_ref_points(depth)
                        execute_arm("left")
                        execute_arm("right")
                    except ValueError as exc:
                        print(f"执行失败：{exc}")
                    finally:
                        clear_state()
                        print("已自动清点。")
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
