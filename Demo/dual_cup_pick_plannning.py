from __future__ import annotations

import re
import textwrap
from typing import List, Optional

import cv2

from pick_place_cup_motion import build_pick_cup_sequence, build_place_cup_sequence
from point2pos_utils import load_cam2ref, load_intrinsics, pixel_to_ref_point
from arx_pointing import predict_multi_points_from_rgb

import time
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


def do_replan(color_img: np.ndarray, planning_prompt: str) -> List[str]:
    raw_result = predict_multi_points_from_rgb(
        color_img,
        text_prompt="",
        all_prompt=planning_prompt,
        assume_bgr=False,
        return_raw=True,
        temperature=0.0,
    )
    if isinstance(raw_result, tuple):
        _, pick_answer_text = raw_result
    else:
        pick_answer_text = None

    pick_plan = _extract_numbered_sentences(pick_answer_text)
    if not pick_plan:
        # 递归重试
        return do_replan(color_img=color_img, planning_prompt=planning_prompt)
    return pick_plan


def _predict_pick_place_once(
    color: np.ndarray, base_prompt: str
) -> tuple[tuple[int, int], tuple[int, int]]:
    full_prompt = (
        "Provide exactly two points coordinate of the pick object and the place coaster this sentence describes: "
        f"{base_prompt} "
        "First point is the object, second point is the coaster."
    )
    points = predict_multi_points_from_rgb(
        color,
        text_prompt="",
        all_prompt=full_prompt,
        assume_bgr=False,
        temperature=0.5,
    )
    if len(points) < 2:
        raise RuntimeError(f"未获取到足够点({len(points)}): {points}")
    pick = (int(round(points[0][0])), int(round(points[0][1])))
    place = (int(round(points[1][0])), int(round(points[1][1])))
    return pick, place


def _predict_pick_only(
    color: np.ndarray, base_prompt: str
) -> tuple[int, int]:
    full_prompt = (
        "Provide exactly one point coordinate of objects region this sentence describes: "
        f"{base_prompt} "
        'The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}]. '
        "Return only JSON."
    )
    points = predict_multi_points_from_rgb(
        color,
        text_prompt="",
        all_prompt=full_prompt,
        assume_bgr=False,
        temperature=0.0,
    )
    if not points:
        raise RuntimeError(f"未获取到点: {points}")
    pick = (int(round(points[0][0])), int(round(points[0][1])))
    return pick


def _draw_text_lines(
    img: np.ndarray,
    lines: List[str],
    origin: tuple[int, int] = (10, 30),
    line_height: int = 22,
    color: tuple[int, int, int] = (0, 0, 255),
    scale: float = 0.5,
    thickness: int = 2,
) -> None:
    x, y = origin
    for i, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (x, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
        )


def dual_pick_planning(
    arx: ARXRobotEnv,
    reset_robot: bool = True,
    close_robot: bool = True,
    no_last_place: bool = False,
):
    try:
        if reset_robot:
            arx.reset()
        arx.step_lift(16.0)
        K = load_intrinsics()
        T_left, T_right = load_cam2ref()

        planned = True
        step_idx = 0
        plan_steps: List[str] = []

        goal_cup = "red cup"
        planning_prompt = (
            f"Current Goal is: pick the {goal_cup}. "
            "I need to pick up the cups from top to the goal cup."
            "What is the picking plan steps to finish the goal?"
        )
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
            current_plan = do_replan(color, planning_prompt)

            # --- 可视化：在图像上打印出规划 prompt 供确认 ---
            vis_img = color.copy()
            prompt_lines = textwrap.wrap(planning_prompt, width=60)
            _draw_text_lines(
                vis_img,
                ["Planning Prompt:"] + prompt_lines,
                origin=(10, 30),
                line_height=22,
                color=(0, 0, 255),
                scale=0.5,
                thickness=2,
            )
            print(f"生成的规划结果 ({len(current_plan)} 步):")

            if not current_plan:
                print("未生成有效步骤！")
            else:
                for i, step in enumerate(current_plan):
                    print(f"  {i+1}. {step}")
            cv2.imshow(confirm_win, vis_img)
            print("按'y' 确认, 'r' 重试, 'p' 更新目标, 'n' 退出")
            key = cv2.waitKey(0)
            if key == ord("y") and current_plan:
                plan_steps = current_plan
                print("规划已确认，进入执行模式。")
                break
            if key == ord("r"):
                print("重新尝试规划...")
                continue
            if key == ord("p"):
                new_goal = input("输入新的需求 (留空保持当前): ").strip()
                if new_goal:
                    goal_cup = new_goal
                    planning_prompt = (
                        f"Current Goal is: pick the {goal_cup}. "
                        "I need to pick up the cups from top to the goal cup."
                        "What is the picking plan steps to finish the goal?"
                    )
                    print(f"新的 pick prompt 已设置为: {planning_prompt!r}")
                continue
            if key == ord("n"):
                print("退出程序。")
                planned = False
                break

        if planned:
            cv2.destroyWindow(confirm_win)
            win = "dual_cup_pick_planning"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        while planned and step_idx < len(plan_steps):
            if step_idx != 0:
                arx.step_lift(13.0)
                time.sleep(1.5)
            frames = arx.node.get_camera(
                target_size=(640, 480), return_status=False)
            color = frames.get("camera_h_color")
            depth = frames.get("camera_h_aligned_depth_to_color")
            if color is None or depth is None:
                cv2.waitKey(1)
                continue

            color = color.copy()
            pick_text = plan_steps[step_idx]
            arm = "left" if step_idx % 2 == 0 else "right"
            is_last = step_idx == len(plan_steps) - 1
            skip_place = no_last_place and is_last
            if skip_place:
                dual_prompt = f"Point out the {pick_text}."
                try:
                    pick_px = _predict_pick_only(color, dual_prompt)
                except RuntimeError as exc:
                    print(f"预测失败，按 r 重试：{exc}")
                    pick_px = None
                place_px = None
            else:
                coaster_side = "left coaster" if arm == "left" else "right coaster"
                dual_prompt = f"{pick_text} and place on the {coaster_side} of it that has no cup on it."
                try:
                    pick_px, place_px = _predict_pick_place_once(
                        color, dual_prompt)
                except RuntimeError as exc:
                    print(f"预测失败，按 r 重试：{exc}")
                    pick_px, place_px = None, None

            if pick_px is not None:
                cv2.circle(color, pick_px, 5, (0, 0, 255), -1)
            if place_px is not None:
                cv2.circle(color, place_px, 5, (255, 0, 0), -1)
            prompt_lines = textwrap.wrap(dual_prompt, width=60)
            if skip_place:
                prompt_lines = ["(no_last_place)"] + prompt_lines
            _draw_text_lines(
                color,
                [f"Step {step_idx + 1}/{len(plan_steps)} ({arm}):"] +
                prompt_lines,
                origin=(10, 30),
                line_height=20,
                color=(0, 0, 255),
                scale=0.55,
                thickness=2,
            )
            cv2.imshow(win, color)

            key = cv2.waitKey(0)
            if key == ord("r"):
                continue
            if key == ord("n"):
                break
            if key == ord("e"):
                if pick_px is None:
                    print("当前未预测到足够点，按 r 重新预测。")
                    continue
                try:
                    T_cam2ref = T_left if arm == "left" else T_right
                    pick_ref = pixel_to_ref_point(pick_px, depth, K, T_cam2ref)
                    place_ref = None
                    if not skip_place:
                        if place_px is None:
                            print("当前未预测到 place 点，按 r 重新预测。")
                            continue
                        place_ref = pixel_to_ref_point(
                            place_px, depth, K, T_cam2ref)
                except ValueError as exc:
                    print(f"像素/深度异常，重新预测：{exc}")
                    continue

                pick_seq = build_pick_cup_sequence(pick_ref, arm=arm)
                for act in pick_seq:
                    arx.step(act)
                if not skip_place and place_ref is not None:
                    place_seq = build_place_cup_sequence(place_ref, arm=arm)
                    for act in place_seq:
                        arx.step(act)
                arx._go_to_initial_pose()

                step_idx += 1

        if planned and step_idx >= len(plan_steps):
            print("全部步骤已完成。")

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
    dual_pick_planning(arx)


if __name__ == "__main__":
    main()
