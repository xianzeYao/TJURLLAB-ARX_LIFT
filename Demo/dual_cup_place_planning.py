"""
Dual cup place planning demo.

逻辑要求：
- 左右交替：左 pick+place，右 pick+place，共 6 次循环。
- pick：先描述左右各自区域内“离相机最近”的杯子，再用描述预测顶中心点。
- place：前三次使用黑色垫子固定提示词；之后使用全局队列预测两点取均值。
- 队列出队：进入双点 place 后，第 1 次 place 出 1 个，之后每次出 2 个。
- 高度：所有 pick 在 13.5；place 第 1-3 次 13.5，第 4-5 次 17，最后一次 20。
"""
from __future__ import annotations

import sys
import textwrap
import time
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np

from arx_pointing import predict_multi_points_from_rgb, predict_point_from_rgb
from demo_utils import draw_text_lines, execute_pick_place_cup_sequence
from point2pos_utils import (
    filter_valid_points,
    load_cam2ref,
    load_intrinsics,
    pixel_to_ref_point,
)

sys.path.append("../ARX_Realenv/ROS2")  # noqa
from arx_ros2_env import ARXRobotEnv  # noqa

PLACE_PROMPT_BLACK1 = "The center of the leftmost black coaster has no cup on it."
PLACE_PROMPT_BLACK2 = "The center of the rightmost black coaster has no cup on it."
PLACE_PROMPT_BLACK3 = "The center of the middle black coaster has no cup on it."
PLACE_PROMPTS_BLACK = (
    PLACE_PROMPT_BLACK1,
    PLACE_PROMPT_BLACK2,
    PLACE_PROMPT_BLACK3,
)


def _arm_for_cycle(cycle_idx: int) -> str:
    return "left" if cycle_idx % 2 == 0 else "right"


def _place_height(place_idx: int) -> float:
    if place_idx < 3:
        return 13.5
    if place_idx < 5:
        return 17.0
    return 20.0


def _get_frames(arx: ARXRobotEnv) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    frames = arx.node.get_camera(target_size=(640, 480), return_status=False)
    color = frames.get("camera_h_color")
    depth = frames.get("camera_h_aligned_depth_to_color")
    return color, depth


def _is_combined_stage(cycle_idx: int) -> bool:
    return cycle_idx < 3


def _desc_prompt(arm: str) -> str:
    if arm == "left":
        return (
            "Describe the cup on the left side which is not on the black coaster and not on other cups. "
            "Output format like: red cup."
        )
    return (
        "Describe the cup on the right side which is not on the black coaster and not on other cups. "
        "Output format like: red cup."
    )


def _clean_desc(raw_text: Optional[str]) -> str:
    if not raw_text:
        return "cup"
    desc = raw_text.strip().splitlines()[0].strip()
    desc = desc.strip(" \t\r\n\"'.,;:")
    if not desc:
        return "cup"
    if "cup" not in desc.lower():
        desc = f"{desc} cup"
    return desc


def describe_cup(color: np.ndarray, arm: str) -> str:
    prompt = _desc_prompt(arm)
    _, raw_text = predict_point_from_rgb(
        color,
        text_prompt="",
        all_prompt=prompt,
        assume_bgr=False,
        return_raw=True,
    )
    return _clean_desc(raw_text)


def predict_pick_point(color: np.ndarray, desc: str) -> Tuple[int, int]:
    pick_prompt = f"{desc}"
    u, v = predict_point_from_rgb(
        color,
        text_prompt=pick_prompt,
        assume_bgr=False,
    )
    return int(round(u)), int(round(v))


def predict_place_black(color: np.ndarray, idx: int) -> Tuple[int, int]:
    prompt = PLACE_PROMPTS_BLACK[idx]
    u, v = predict_point_from_rgb(
        color,
        text_prompt=prompt,
        assume_bgr=False,
    )
    return int(round(u)), int(round(v))


def predict_place_from_queue(
    color: np.ndarray,
    cup1: str,
    cup2: str,
    depth: np.ndarray,
    K: np.ndarray,
    T_cam2ref: np.ndarray,
) -> Tuple[Optional[Tuple[int, int]], Optional[np.ndarray], Optional[List[Tuple[int, int]]], str]:
    prompt = (
        f"Point to exactly top center of the {cup1}.\n"
        f"Point to exactly top center of the {cup2}.\n"
        "Output the pixel coordinates of the two points."
    )
    raw_uvs, raw_text = predict_multi_points_from_rgb(
        color,
        text_prompt="",
        all_prompt=prompt,
        assume_bgr=False,
        return_raw=True,
    )
    raw_uvs = raw_uvs[:2]
    uv_ints = [(int(round(u)), int(round(v+2.5))) for (u, v) in raw_uvs]
    valid_uvs, valid_refs = filter_valid_points(uv_ints, depth, K, T_cam2ref)
    if len(valid_uvs) < 2:
        print("需要至少 2 个有效放置参考点，按 r 重新预测")
        return None, None, None, prompt
    attachment_uvs = valid_uvs[:2]
    place_uv = tuple(
        np.mean(np.array(attachment_uvs), axis=0).round().astype(int)
    )
    place_ref = np.mean(np.stack(valid_refs[:2]), axis=0)
    return place_uv, place_ref, attachment_uvs, prompt


def dual_cup_place_planning(
    arx: ARXRobotEnv,
    reset_robot: bool = True,
    close_robot: bool = True,
):
    total_cycles = 6
    if reset_robot:
        arx.reset()
    arx.step_lift(13.5)

    K = load_intrinsics()
    T_left, T_right = load_cam2ref()

    picked_queue = deque()
    queue_place_count = 0

    win = "dual_cup_place_planning"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    cycle_idx = 0
    stage = "pick"

    current_desc: Optional[str] = None
    pick_prompt: Optional[str] = None
    current_pick_uv: Optional[Tuple[int, int]] = None
    current_pick_ref: Optional[np.ndarray] = None
    pick_color: Optional[np.ndarray] = None

    predicted_place_uv: Optional[Tuple[int, int]] = None
    place_ref: Optional[np.ndarray] = None
    attachment_uvs: Optional[List[Tuple[int, int]]] = None
    place_prompt: Optional[str] = None
    place_color: Optional[np.ndarray] = None
    use_queue_place = False

    try:
        while cycle_idx < total_cycles:
            arm = _arm_for_cycle(cycle_idx)
            T_cam2ref = T_left if arm == "left" else T_right
            combined_stage = _is_combined_stage(cycle_idx)
            display_color = None

            # 预测 pick
            if stage == "pick" or combined_stage:
                if current_pick_ref is None:
                    arx.step_lift(13.5)
                    pick_color, pick_depth = _get_frames(arx)
                    if pick_color is None or pick_depth is None:
                        cv2.waitKey(1)
                        continue
                    current_desc = describe_cup(pick_color, arm)
                    pick_prompt = f"{current_desc}"
                    current_pick_uv = predict_pick_point(
                        pick_color, current_desc)
                    raw_depth = pick_depth[current_pick_uv[1],
                                           current_pick_uv[0]]
                    if np.isnan(raw_depth) or raw_depth == 0:
                        print(
                            f"pick 像素 {current_pick_uv} 深度无效({raw_depth})，按 r 重新预测"
                        )
                        current_pick_uv = None
                        current_pick_ref = None
                        continue
                    current_pick_ref = pixel_to_ref_point(
                        current_pick_uv, pick_depth, K, T_cam2ref
                    )
                    print(
                        f"pick 预测像素 {current_pick_uv} -> ref {current_pick_ref.tolist()}，按 e 执行"
                    )
                display_color = pick_color

            # 预测 place
            if stage == "place" or combined_stage:
                if place_ref is None:
                    place_idx = cycle_idx
                    place_height = _place_height(place_idx)
                    arx.step_lift(place_height)
                    if place_height > 13.5:
                        time.sleep(1.0)
                    place_color, place_depth = _get_frames(arx)
                    if place_color is None or place_depth is None:
                        cv2.waitKey(1)
                        continue

                    if place_idx < 3:
                        predicted_place_uv = predict_place_black(
                            place_color, place_idx)
                        place_prompt = PLACE_PROMPTS_BLACK[place_idx]
                        attachment_uvs = None
                        raw_depth = place_depth[
                            predicted_place_uv[1], predicted_place_uv[0]
                        ]
                        if np.isnan(raw_depth) or raw_depth == 0:
                            print(
                                f"place 像素 {predicted_place_uv} 深度无效({raw_depth})，按 r 重新预测"
                            )
                            predicted_place_uv = None
                            place_ref = None
                            continue
                        place_ref = pixel_to_ref_point(
                            predicted_place_uv, place_depth, K, T_cam2ref
                        )
                        use_queue_place = False
                    else:
                        use_queue_place = True
                        if len(picked_queue) < 2:
                            print("队列不足 2 个描述，按 r 重新预测或继续完成 pick")
                            predicted_place_uv = None
                            place_ref = None
                            attachment_uvs = None
                            continue
                        if place_idx == 3:
                            cup1, cup2 = picked_queue[1], picked_queue[2]
                            print(
                                f"place_idx: {place_idx},cup1: {cup1}, cup2: {cup2}")
                        else:
                            cup1, cup2 = picked_queue[0], picked_queue[1]
                            print(
                                f"place_idx: {place_idx},cup1: {cup1}, cup2: {cup2}")
                        (
                            predicted_place_uv,
                            place_ref,
                            attachment_uvs,
                            place_prompt,
                        ) = predict_place_from_queue(
                            place_color,
                            cup1,
                            cup2,
                            place_depth,
                            K,
                            T_cam2ref,
                        )
                        if place_ref is None:
                            continue

                    print(
                        f"place 预测像素 {predicted_place_uv} -> ref {place_ref.tolist()}，按 e 执行"
                    )
                display_color = place_color

            if display_color is None:
                display_color, _ = _get_frames(arx)
                if display_color is None:
                    cv2.waitKey(1)
                    continue

            # 可视化
            disp = display_color.copy()
            if current_pick_uv:
                cv2.circle(disp, current_pick_uv, 5, (0, 0, 255), -1)
            if predicted_place_uv:
                cv2.circle(disp, predicted_place_uv, 5, (0, 0, 255), -1)
            if attachment_uvs:
                for uv in attachment_uvs:
                    cv2.circle(disp, uv, 5, (255, 0, 0), -1)

            stage_label = "pick+place" if combined_stage else stage
            header = f"Cycle {cycle_idx + 1}/{total_cycles} ({arm}) [{stage_label}]"
            pick_line = f"Pick: {pick_prompt}" if pick_prompt else "Pick: -"
            place_line = f"Place: {place_prompt}" if place_prompt else "Place: -"
            lines = [header] + textwrap.wrap(pick_line, width=60)
            lines += textwrap.wrap(place_line, width=60)
            draw_text_lines(
                disp,
                lines,
                origin=(10, 30),
                line_height=22,
                color=(0, 0, 255),
                scale=0.55,
                thickness=2,
            )
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                if combined_stage or stage == "pick":
                    current_desc = None
                    pick_prompt = None
                    current_pick_uv = None
                    current_pick_ref = None
                    pick_color = None
                if combined_stage or stage == "place":
                    predicted_place_uv = None
                    place_ref = None
                    attachment_uvs = None
                    place_prompt = None
                    place_color = None
                    use_queue_place = False
                continue

            if key == ord("e"):
                # 前三次合并执行；之后拆成 pick->place 两段
                if combined_stage:
                    if current_pick_ref is None or place_ref is None:
                        print("当前未预测到足够点，按 r 重新预测。")
                        continue
                    arx.step_lift(13.5)
                    execute_pick_place_cup_sequence(
                        arx=arx,
                        pick_ref=current_pick_ref,
                        place_ref=None,
                        arm=arm,
                        do_pick=True,
                        do_place=False,
                        go_home=False,
                    )
                    if current_desc:
                        picked_queue.append(current_desc)
                    arx.step_lift(_place_height(cycle_idx))
                    if _place_height(cycle_idx) > 13.5:
                        time.sleep(1.0)
                    execute_pick_place_cup_sequence(
                        arx=arx,
                        pick_ref=None,
                        place_ref=place_ref,
                        arm=arm,
                        do_pick=False,
                        do_place=True,
                        go_home=True,
                    )
                    cycle_idx += 1
                    current_desc = None
                    pick_prompt = None
                    current_pick_uv = None
                    current_pick_ref = None
                    predicted_place_uv = None
                    place_ref = None
                    attachment_uvs = None
                    place_prompt = None
                    pick_color = None
                    place_color = None
                    use_queue_place = False
                    stage = "pick"
                    continue

                if stage == "pick":
                    if current_pick_ref is None:
                        print("当前未预测到 pick 点，按 r 重新预测。")
                        continue
                    arx.step_lift(13.5)
                    execute_pick_place_cup_sequence(
                        arx=arx,
                        pick_ref=current_pick_ref,
                        place_ref=None,
                        arm=arm,
                        do_pick=True,
                        do_place=False,
                        go_home=False,
                    )
                    if current_desc:
                        picked_queue.append(current_desc)
                    current_desc = None
                    current_pick_uv = None
                    current_pick_ref = None
                    pick_color = None
                    stage = "place"
                    continue

                if stage == "place":
                    if place_ref is None:
                        print("当前未预测到 place 点，按 r 重新预测。")
                        continue
                    arx.step_lift(_place_height(cycle_idx))
                    if _place_height(cycle_idx) > 13.5:
                        time.sleep(1.0)
                    execute_pick_place_cup_sequence(
                        arx=arx,
                        pick_ref=None,
                        place_ref=place_ref,
                        arm=arm,
                        do_pick=False,
                        do_place=True,
                        go_home=True,
                    )
                    if use_queue_place:
                        if queue_place_count == 0:
                            if len(picked_queue) > 1:
                                temp_list = list(picked_queue)
                                del temp_list[1]
                                picked_queue = deque(temp_list)
                        else:
                            if picked_queue:
                                picked_queue.popleft()
                            if picked_queue:
                                picked_queue.popleft()
                        queue_place_count += 1

                    cycle_idx += 1
                    predicted_place_uv = None
                    place_ref = None
                    attachment_uvs = None
                    place_prompt = None
                    place_color = None
                    use_queue_place = False
                    pick_prompt = None
                    stage = "pick"
                    continue

            if key in (27, ord("q")):
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
    dual_cup_place_planning(arx)


if __name__ == "__main__":
    main()
