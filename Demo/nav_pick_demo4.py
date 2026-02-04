from autonav_robot import AutoNav_Robot
from nav_utils import path_to_actions, merge_forward_actions, index_resample
from pick_place_cup_motion import *
from arx_pointing import *
from point2pos_utils import *
from cup_pick_planning2 import pick_planning
import time
import cv2
import math


def main():
    arx_nav_robot = AutoNav_Robot()
    try:
        # go
        # -- get cup start --
        pick_planning(arx_nav_robot.arx, reset_robot=False, close_robot=False)
        # -- get cup end --

        # -- turn left pi/2 --
        arx_nav_robot.run_for_1s(chz=0.5, duration=20.6 / 2.0)

        # -- start point --
        arx_nav_robot.initialize_pose()
        color, depth = arx_nav_robot.get_color_depth()
        points = arx_nav_robot.turn_right_corner(color)

        # -- visualize --
        order_num = 0.0

        revised_points = []

        order_num = 0.0

        revised_points = []

        for (u, v) in points:
            u += 80
            v += 30
            cv2.circle(
                color,
                center=(int(u), int(v)),
                radius=5,
                color=(order_num, order_num, 255 - order_num),
                thickness=-1  # -1 表示实心圆
            )
            order_num += 30
            revised_points.append((u, v))

        cv2.imwrite("../Testdata4Nav/test_1.png", color)

        # time.sleep(10.0)

        path_xy = []

        # -- pixel to wolrd point --
        for point in revised_points:
            Pw = arx_nav_robot.pixel_to_pw(point, depth)
            path_xy.append((Pw[0], Pw[1]))

        print(path_xy[:7])

        arx_nav_robot.follow_path(
            path_xy[:7], lookahead=0.12, v_max=0.15, v_min=0.13, reach_dis=0.09, show_index=True)

        # time.sleep(10.0)

        # -- turn right --
        print("Turn right......")
        color, depth = arx_nav_robot.get_color_depth()
        points = arx_nav_robot.detect_goal(
            color, "red circular landmark on the ground")

        cv2.circle(
            color,
            center=(int(points[0][0]), int(points[0][1])),
            radius=5,
            color=(0, 0, 255),
            thickness=-1  # -1 表示实心圆
        )

        cv2.imwrite("../Testdata4Nav/test_2.png", color)

        # time.sleep(10.0)

        goal_pw = arx_nav_robot.pixel_to_pw(points[0], depth)
        start = (0, 0)
        goal = (goal_pw[0], -goal_pw[1])

        path = [start, goal]
        actions = path_to_actions(path)
        actions = merge_forward_actions(actions)

        # -- move to goal --
        for action, action_content in actions:
            if action == "forward":
                arx_nav_robot.run_for_1s(
                    chx=1.0, duration=(action_content)/0.23)
            elif action == "rotate":
                if action_content <= 0:
                    arx_nav_robot.run_for_1s(
                        chz=-0.5, duration=max(float((-action_content/(0.5 * 2*math.pi / 20.6))) - 0.5, 0.0))
                else:
                    arx_nav_robot.run_for_1s(
                        chz=0.5, duration=action_content/(0.5 * 2*math.pi / 20.6))

        # -- turn right start--
        # arx_nav_robot.turn_right_until_see_goal(goal="black round coaster on table", max_angle=(math.pi*2.0/3.0))
        # arx_nav_robot.run_for_1s(chz=-0.5, duration=20.6/12.0)
        # points, detect_flag, color = arx_nav_robot.turn_right_until_see_goal(goal="black round coaster", max_angle=(math.pi*2.0/3.0))
        # if not detect_flag:
        #     raise RuntimeError(f"未找到目标")
        # cv2.circle(
        #     color,
        #     center=(int(points[0][0]), int(points[0][1])),
        #     radius=5,
        #     color=(0, 0, 255),
        #     thickness=-1  # -1 表示实心圆
        # )

        # cv2.imwrite("../Testdata4Nav/test_3.png", color)

        arx_nav_robot.run_for_1s(chz=-0.5, duration=20.6 / 2.5)
        # time.sleep(10.0)
        # -- turn right end--

        color, depth = arx_nav_robot.get_color_depth()
        points = arx_nav_robot.detect_goal(
            color, "the black round coaster on the table")
        goal_pw = arx_nav_robot.pixel_to_pw(points[0], depth)
        start = (0, 0)
        goal = (goal_pw[0], -goal_pw[1])

        path = [start, goal]
        actions = path_to_actions(path)
        actions = merge_forward_actions(actions)

        cv2.circle(
            color,
            center=(int(points[0][0]), int(points[0][1])),
            radius=5,
            color=(0, 0, 255),
            thickness=-1  # -1 表示实心圆
        )

        cv2.imwrite("../Testdata4Nav/test_3.png", color)

        for action, action_content in actions:
            if action == "rotate":
                if action_content <= 0:
                    arx_nav_robot.run_for_1s(
                        chz=-0.5, duration=max(float((-action_content/(0.5 * 2*math.pi / 20.6))), 0.0))
                else:
                    arx_nav_robot.run_for_1s(
                        chz=0.5, duration=action_content/(0.5 * 2*math.pi / 20.6))

        # foward a little
        arx_nav_robot.run_for_1s(chx=0.5, duration=2.2)

        # -- place cup start --
        arx_nav_robot.arx.step_lift(18.0)
        K = load_intrinsics()
        T_cam2ref = load_cam2ref(side="left")
        pt_ref = None
        pick_prompt = "the black round coaster"
        color, depth = arx_nav_robot.get_color_depth()
        # if color is None or depth is None:
        #     print("Failed to get color or depth image.")
        u, v = predict_point_from_rgb(
            color,
            text_prompt=pick_prompt,
        )
        predicted_px = (int(round(u)), int(round(v)))
        # TODO 深度check
        # raw_depth= depth[predicted_px[1], predicted_px[0]]
        # 得到XYZ
        pt_ref = pixel_to_ref_point(
            predicted_px, depth, K, T_cam2ref)
        action_seq = build_place_cup_sequence(pt_ref, arm="left")
        for act in action_seq:
            # print(act)
            arx_nav_robot.arx.step(act)
        arx_nav_robot.arx._go_to_initial_pose()
        # -- place cup end --

        time.sleep(15.0)

        # -- pick cup start --
        arx_nav_robot.arx.step_lift(18.0)
        K = load_intrinsics()
        T_cam2ref = load_cam2ref(side="left")
        pt_ref = None
        pick_prompt = "the cup on the grey round coaster"
        color, depth = arx_nav_robot.get_color_depth()
        # if color is None or depth is None:
        #     print("Failed to get color or depth image.")
        u, v = predict_point_from_rgb(
            color,
            text_prompt=pick_prompt,
        )
        predicted_px = (int(round(u)), int(round(v)))
        # TODO 深度check
        # raw_depth= depth[predicted_px[1], predicted_px[0]]
        # 得到XYZ
        pt_ref = pixel_to_ref_point(
            predicted_px, depth, K, T_cam2ref)
        action_seq = build_pick_cup_sequence(pt_ref, arm="left")
        for act in action_seq:
            # print(act)
            arx_nav_robot.arx.step(act)
        arx_nav_robot.arx.step_lift(15.0)
        # -- pick cup end --

        # step back a little
        arx_nav_robot.run_for_1s(chx=-0.5, duration=2.2)

        arx_nav_robot.run_for_1s(chz=-0.5, duration=20.6 - 20.6 / 2.5)

        arx_nav_robot.motion_inversion()

        arx_nav_robot.run_for_1s(chx=0.5, duration=2.5)

        arx_nav_robot.run_for_1s(chz=0.5, duration=10.3)

        arx_nav_robot.run_for_1s(chx=0.5, duration=1.5)

        # -- put cup start --
        arx_nav_robot.arx.step_lift(14.0)
        K = load_intrinsics()
        T_cam2ref = load_cam2ref(side="left")
        pt_ref = None
        place_prompt = "the grey round coaster"
        color, depth = arx_nav_robot.get_color_depth()
        u, v = predict_point_from_rgb(
            color,
            text_prompt=place_prompt,
        )
        predicted_px = (int(round(u)), int(round(v)))
        # TODO 深度check
        # raw_depth= depth[predicted_px[1], predicted_px[0]]
        # 得到XYZ
        pt_ref = pixel_to_ref_point(
            predicted_px, depth, K, T_cam2ref)
        action_seq = build_place_cup_sequence(pt_ref, arm="left")
        for act in action_seq:
            arx_nav_robot.arx.step(act)
        arx_nav_robot.arx._go_to_initial_pose()
        # -- put cup end --

    finally:
        arx_nav_robot.arx.close()


if __name__ == "__main__":
    main()
