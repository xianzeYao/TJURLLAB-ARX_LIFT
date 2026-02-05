from __future__ import annotations

from typing import Optional, Tuple

import cv2

from arx_pointing import predict_multi_points_from_rgb
from demo_utils import draw_text_lines, execute_pick_place_cup_sequence
from point2pos_utils import load_cam2ref, load_intrinsics, pixel_to_ref_point

import sys

sys.path.append("../ARX_Realenv/ROS2")  # noqa
from arx_ros2_env import ARXRobotEnv  # noqa


def _predict_one_point(color: np.ndarray, base_prompt: str) -> Tuple[int, int]:
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
        raise RuntimeError("未解析到坐标")
    u, v = points[0]
    return int(round(u)), int(round(v))


def _predict_two_points(
    color: np.ndarray, pick_prompt: str, place_prompt: str
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    full_prompt = (
        "Provide exactly two points coordinate of objects region this sentence describes: "
        f"{pick_prompt} and {place_prompt}. "
        'The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}]. '
        "Return only JSON. First point is pick, second point is place."
    )
    points = predict_multi_points_from_rgb(
        color,
        text_prompt="",
        all_prompt=full_prompt,
        assume_bgr=False,
        temperature=0.0,
    )
    if len(points) < 2:
        raise RuntimeError("未解析到足够坐标")
    pick = (int(round(points[0][0])), int(round(points[0][1])))
    place = (int(round(points[1][0])), int(round(points[1][1])))
    return pick, place


def _get_frame(arx: ARXRobotEnv) -> Tuple[np.ndarray, np.ndarray]:
    while True:
        frames = arx.node.get_camera(
            target_size=(640, 480), return_status=False)
        color = frames.get("camera_h_color")
        depth = frames.get("camera_h_aligned_depth_to_color")
        if color is None or depth is None:
            cv2.waitKey(1)
            continue
        return color, depth


def single_arm_pick_place(
    arx: ARXRobotEnv,
    pick_prompt: str,
    place_prompt: str,
    arm: str = "left",
    reset_robot: bool = True,
    close_robot: bool = True,
    confirm: bool = True,
    debug: bool = True,
    go_home: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        if reset_robot:
            arx.reset()

        K = load_intrinsics()
        T_cam2ref = load_cam2ref(side=arm)

        while True:
            confirm_now = confirm and debug
            do_pick = bool(pick_prompt)
            do_place = bool(place_prompt)
            if not do_pick and not do_place:
                raise ValueError("pick_prompt 和 place_prompt 不能同时为空")

            color, depth = _get_frame(arx)
            pick_px = None
            place_px = None
            if do_pick and do_place:
                pick_px, place_px = _predict_two_points(
                    color, pick_prompt, place_prompt
                )
            elif do_pick:
                pick_px = _predict_one_point(color, pick_prompt)
            else:
                place_px = _predict_one_point(color, place_prompt)

            pick_ref = None
            place_ref = None
            if pick_px is not None:
                pick_ref = pixel_to_ref_point(pick_px, depth, K, T_cam2ref)
            if place_px is not None:
                place_ref = pixel_to_ref_point(place_px, depth, K, T_cam2ref)

            vis = color.copy()
            lines = []
            if pick_px is not None:
                cv2.circle(vis, pick_px, 5, (0, 0, 255), -1)
                lines.append(f"Pick: {pick_prompt}")
            if place_px is not None:
                cv2.circle(vis, place_px, 5, (255, 0, 0), -1)
                lines.append(f"Place: {place_prompt}")
            if lines:
                draw_text_lines(
                    vis,
                    lines,
                    origin=(10, 25),
                    line_height=22,
                    color=(0, 0, 255),
                    scale=0.6,
                    thickness=2,
                )

            if confirm_now:
                win = "single_arm_pick_place"
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                cv2.imshow(win, vis)
                key = cv2.waitKey(0)
                if key == ord("r"):
                    continue
                if key == ord("n"):
                    return None, None
                # 默认确认
            else:
                cv2.destroyAllWindows()

            execute_pick_place_cup_sequence(
                arx=arx,
                pick_ref=pick_ref,
                place_ref=place_ref,
                arm=arm,
                do_pick=do_pick,
                do_place=do_place,
                go_home=go_home,
            )
            return pick_ref, place_ref
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
    pick_prompt = "the target cup"
    place_prompt = "the target coaster"
    single_arm_pick_place(arx, pick_prompt, place_prompt)


if __name__ == "__main__":
    main()
