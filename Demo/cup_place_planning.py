"""
Cup place planning demo.

逻辑要求：
- pick prompt 固定为 "left side cups not on coaster"。
- 前 3 次 place：使用单点 text prompt "the coaster with the smallest number"。
- 第 4 次及之后的 place：使用双点 prompt
    1. Point to top of the cup that has no cup on it.
    2. Point to top of the another cup that has no cup on it.
    - Output the pixel coordinates of the center points.
  取两个点的深度，UV 取平均用于可视化，3D 点取两个 ref 点的均值作为放杯位。

"""
from __future__ import annotations

import textwrap
from typing import List, Optional, Tuple
from collections import deque

import cv2
import numpy as np

from point2pos_utils import (
    load_cam2ref,
    load_intrinsics,
    pixel_to_ref_point,
    filter_valid_points,
)
from arx_pointing import predict_multi_points_from_rgb, predict_point_from_rgb
from demo_utils import draw_text_lines, execute_pick_place_cup_sequence

import sys

sys.path.append("../ARX_Realenv/ROS2")  # noqa
from arx_ros2_env import ARXRobotEnv  # noqa


def place_planning(arx: ARXRobotEnv, reset_robot: bool = True, close_robot: bool = True):
    if reset_robot:
        arx.reset()
    arx.step_lift(13.0)

    K = load_intrinsics()
    T_cam2ref = load_cam2ref(side="left")

    current_pick_uv: Optional[Tuple[int, int]] = None
    current_pick_ref: Optional[np.ndarray] = None

    predicted_place_uv: Optional[Tuple[int, int]] = None
    place_pt_ref: Optional[np.ndarray] = None
    attachment_uvs: Optional[List[Tuple[int, int]]] = None

    picked_cup = deque()
    cup1 = None
    cup2 = None
    # pick_prompt = "left side cup's top"
    place_prompt_single = "the coaster with the smallest number"
    # place_prompt_multi = f""" Point to top of the {cup1}.
    #                          Point to top of the {cup2}.
    #                         - Output the pixel coordinates of the two points."""
    current_place_prompt = place_prompt_single

    try:
        win = "cup_place_planning"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        i = 0  # 偶数轮 pick，奇数轮 place
        while True:
            frames = arx.node.get_camera(
                target_size=(640, 480), return_status=False)
            color = frames.get("camera_h_color")
            depth = frames.get("camera_h_aligned_depth_to_color")
            if color is None or depth is None:
                cv2.waitKey(1)
                continue

            need_pick = i % 2 == 0
            need_place = i % 2 == 1
            if (need_pick and current_pick_ref is None) or (need_place and place_pt_ref is None):
                if need_pick:
                    # 保持所有 pick 高度为 13.0
                    if i != 0:
                        arx.step_lift(13.0)
                    _, target = predict_point_from_rgb(
                        color,
                        text_prompt="",
                        all_prompt=f"Describe the left most side cup's color.Output format like : green cup ",
                        assume_bgr=False,
                        return_raw=True,
                    )
                    pick_prompt = f"the up-center of {target}"
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
                    frames = arx.node.get_camera(
                        target_size=(640, 480), return_status=False)
                    color_latest = frames.get("camera_h_color")
                    depth_latest = frames.get(
                        "camera_h_aligned_depth_to_color")
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
                        if i == 7 or i == 9:
                            arx.step_lift(17.0)
                        if i == 11:
                            arx.step_lift(20.0)
                        if len(picked_cup) < 2:
                            print("picked_cup 不足 2 个，按 r 重新预测或先完成更多 pick")
                            predicted_place_uv = None
                            place_pt_ref = None
                            attachment_uvs = None
                            continue
                        cup1, cup2 = picked_cup[0], picked_cup[1]
                        place_prompt_multi = f""" Point to top center of the {cup1} cup.
                                                  Point to top center of the {cup2} cup.
                                                Output the pixel coordinates of the two points."""
                        print(f"place 使用双点提示词，杯子分别为: {cup1}, {cup2}")
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
                        valid_uvs, valid_refs = filter_valid_points(
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
                cv2.circle(disp, current_pick_uv, 3,  (0, 0, 255), -1)
            if predicted_place_uv and i % 2 == 1:
                cv2.circle(disp, predicted_place_uv, 3,  (0, 0, 255), -1)
            if attachment_uvs:
                for uv in attachment_uvs:
                    cv2.circle(disp, uv, 3,  (255, 0, 0), -1)

            curr_prompt = pick_prompt if i % 2 == 0 else current_place_prompt
            prompt_lines = textwrap.wrap(f"prompt: {curr_prompt}", width=32)
            draw_text_lines(
                disp,
                prompt_lines,
                origin=(10, 25),
                line_height=25,
                color=(0, 0, 255),
                scale=0.7,
                thickness=2,
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
                    execute_pick_place_cup_sequence(
                        arx=arx,
                        pick_ref=current_pick_ref,
                        place_ref=None,
                        arm="left",
                        do_pick=True,
                        do_place=False,
                        go_home=False,
                    )
                    picked_cup.append(target)
                    current_pick_uv = None
                    current_pick_ref = None
                    i += 1
                elif i % 2 == 1 and place_pt_ref is not None:
                    execute_pick_place_cup_sequence(
                        arx=arx,
                        pick_ref=None,
                        place_ref=place_pt_ref,
                        arm="left",
                        do_pick=False,
                        do_place=True,
                        go_home=True,
                    )
                    if len(picked_cup) >= 2 and i >= 7:
                        if i <= 8:
                            picked_cup.popleft()
                        else:
                            picked_cup.popleft()
                            picked_cup.popleft()
                    predicted_place_uv = None
                    place_pt_ref = None
                    attachment_uvs = None
                    i += 1

            if key in (27, ord("q")):
                break
    finally:
        cv2.destroyAllWindows()
        if close_robot:
            arx.close()


def main():
    arx = ARXRobotEnv(
        duration_per_step=1.0 / 20.0,
        min_steps=20,
        max_v_xyz=0.25, max_a_xyz=0.20,
        max_v_rpy=0.3, max_a_rpy=1.00,
        camera_type="all",
        camera_view=("camera_h",),
        img_size=(640, 480),
    )
    place_planning(arx)


if __name__ == "__main__":
    main()
