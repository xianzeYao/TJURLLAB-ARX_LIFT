
import time
import sys
sys.path.append("../ARX_Realenv/ROS2")  # noqa
from arx_ros2_env import ARXRobotEnv  # noqa
import numpy as np
from arx5_arm_msg.msg._robot_cmd import RobotCmd
from point2pos_utils import load_cam2ref, load_intrinsics, pixel_to_ref_point_safe
from arx_pointing import predict_point_from_rgb
from demo_utils import execute_pick_place_straw_sequence


def dual_cup_straw(arx: ARXRobotEnv, cup_side="left", close_robot=True):
    try:
        arx.reset()
        K = load_intrinsics()
        T = load_cam2ref(side=straw_side)
        pick_straw_prompt = f"the center of the black straw on the {straw_side} side"
        place_straw_prompt = f"the opening of the center of the cup's opening on the {cup_side} hand"
        pick_execute = False
        place_execute = False
        while not pick_execute:
            straw_side = "right" if cup_side == "left" else "left"
            frames = arx.node.get_camera(
                target_size=(640, 480), return_status=False)
            color = frames.get("camera_h_color")
            depth = frames.get("camera_h_aligned_depth_to_color")
            u, v = predict_point_from_rgb(
                color,
                text_prompt=pick_straw_prompt,
                assume_bgr=False)
            predicted_px = (int(round(u)), int(round(v)))
            raw_depth = depth[predicted_px[1], predicted_px[0]]
            if np.isnan(raw_depth) or raw_depth == 0:
                print(
                    f"预测像素 {predicted_px} 深度无效({raw_depth})，按 r 重新预测")
                predicted_px = None
                pt_ref = None
                continue
            pt_ref = pixel_to_ref_point_safe(
                predicted_px, depth, K, T)
            execute_pick_place_straw_sequence(
                arx, pick_ref=pt_ref, arm=straw_side, do_pick=True, do_place=False, go_home=False)
            pick_execute = True
        # 右转90度
        arx.step_base(chx=0.0, chy=0.0, chz=-0.5, duration=10.0)
        # 杯子放到摄像头中央
        if cup_side == "left":
            suit_action = {cup_side: np.array(
                [0.25, -0.125, -0.05, 0, 0, -1.571, -2.0], dtype=np.float32)}
        else:
            suit_action = {cup_side: np.array(
                [0.25, 0.125, -0.05, 0, 0, 1.571, -2.0], dtype=np.float32)}
        arx.step(suit_action)
        while not place_execute:
            frames = arx.node.get_camera(
                target_size=(640, 480), return_status=False)
            color = frames.get("camera_h_color")
            depth = frames.get("camera_h_aligned_depth_to_color")
            u, v = predict_point_from_rgb(
                color,
                text_prompt=place_straw_prompt,
                assume_bgr=False)
            predicted_px = (int(round(u)), int(round(v)))
            raw_depth = depth[predicted_px[1], predicted_px[0]]
            if np.isnan(raw_depth) or raw_depth == 0:
                print(
                    f"预测像素 {predicted_px} 深度无效({raw_depth})，按 r 重新预测")
                predicted_px = None
                pt_ref = None
                continue
            pt_ref = pixel_to_ref_point_safe(
                predicted_px, depth, K, T)
            execute_pick_place_straw_sequence(
                arx, place_ref=pt_ref, arm=straw_side, do_pick=True, do_place=False, go_home=False)
            place_execute = True
        arx.step_base(chx=0.5, chy=0.0, chz=-0.0, duration=1.0)
        # 杯子放到摄像头中央
        if cup_side == "left":
            give_action = {cup_side: np.array(
                [0.3, 0, 0.1, 0, 0, 0, -2.0], dtype=np.float32)}
        else:
            give_action = {cup_side: np.array(
                [0.3, 0, 0.1, 0, 0, 0, -2.0], dtype=np.float32)}
        arx.step(give_action)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if close_robot:
            arx.close()


def main():
    arx = ARXRobotEnv(duration_per_step=1.0/20.0,  # 就是插值里一步的时间，20Hz也就是0.05s
                            min_steps_per_action=60,  # 每个动作至少插值60步，理论上来说越大越好
                            min_steps_gripper=20,  # 夹爪插值步数最少20步
                            max_v_xyz=0.15,
                            max_v_rpy=0.3,
                            camera_type="all",
                            camera_view=("camera_h",),
                            img_size=(640, 480))
    dual_cup_straw(arx, cup_side="left", close_robot=True)


if __name__ == "__main__":
    main()
