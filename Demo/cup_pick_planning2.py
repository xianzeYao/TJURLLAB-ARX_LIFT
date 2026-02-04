from __future__ import annotations

import re
from typing import List, Optional

import cv2
import numpy as np

from pick_place_cup_motion import build_pick_cup_sequence, build_place_cup_sequence
from point2pos_utils import load_cam2ref, load_intrinsics, pixel_to_ref_point
from arx_pointing import predict_multi_points_from_rgb, predict_point_from_rgb

import sys

sys.path.append("../ARX_Realenv/ROS2")  # noqa
from arx_ros2_env import ARXRobotEnv  # noqa


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
        temperature=0.0
    )
    if isinstance(raw_result, tuple):
        _, pick_answer_text = raw_result
    else:
        pick_answer_text = None

    pick_plan = _extract_numbered_sentences(pick_answer_text)
    if not pick_plan:
        return do_replan(color_img=color_img, pick_prompt=pick_prompt)  # 递归重试
    return pick_plan


def pick_planning(arx: ARXRobotEnv, reset_robot: bool = True, close_robot: bool = True):
    try:
        if reset_robot:
            arx.reset()
        arx.step_lift(17.0)

        K = load_intrinsics()
        T_cam2ref = load_cam2ref()

        planned = True
        step_idx = 0
        plan_steps: List[str] = []

        goal_cup = "red cup"
        pick_prompt = f"Current Goal is: pick the {goal_cup}. I need to pick up the cups from top to the goal cup.What is the picking plan steps to finish the goal?"
        place_prompt = "the smaller number at the center of a white round coaster"

        plan_steps: List[str] = []
        confirm_win = "Planning Step"
        cv2.namedWindow(confirm_win, cv2.WINDOW_NORMAL)

        while True:
            frames = arx.node.get_camera(
                target_size=(640, 480), return_status=False)
            color = frames.get("camera_h_color")
            if color is None:
                cv2.waitKey(1)
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
            print("按'y' 确认, 'r' 重试, 'p' 更新目标, 'n' 退出")
            key = cv2.waitKey(0)
            if key == ord('y') and current_plan:
                plan_steps = current_plan
                print("规划已确认，进入执行模式。")
                break
            elif key == ord('r'):
                print("重新尝试规划...")
                continue
            elif key == ord('p'):
                new_goal = input("输入新的需求 (留空保持当前): ").strip()
                if new_goal:
                    goal_cup = new_goal
                    pick_prompt = (
                        f"Current Goal is: pick the {goal_cup}. "
                        "I need to pick up the cups from top to the goal cup."
                        "What is the picking plan steps to finish the goal?"
                    )
                    print(f"新的 pick prompt 已设置为: {pick_prompt!r}")
                continue
            elif key == ord('n'):
                print("退出程序。")
                planned = False
                break
        if planned:
            cv2.destroyWindow(confirm_win)
            win = "pick_planning_predict2"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        while planned:
            frames = arx.node.get_camera(
                target_size=(640, 480), return_status=False)
            color = frames.get("camera_h_color")
            depth = frames.get("camera_h_aligned_depth_to_color")
            if color is None or depth is None:
                cv2.waitKey(1)
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
                    action_seq = build_pick_cup_sequence(pt_ref, arm="left")
                    for act in action_seq:
                        arx.step(act)
                else:
                    action_seq = build_place_cup_sequence(pt_ref, arm="left")
                    for act in action_seq:
                        arx.step(act)
                    arx._go_to_initial_pose()
                step_idx += 1
            if key == ord("n"):
                break
    finally:
        cv2.destroyAllWindows()
        if close_robot:
            arx.close()


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
    pick_planning(arx)


if __name__ == "__main__":
    main()
