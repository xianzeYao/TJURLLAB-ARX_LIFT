from autonav_robot import AutoNav_Robot
from nav_utils import path_to_actions, merge_forward_actions
from pick_place_cup_motion import *
from arx_pointing import *
from point2pos_utils import *
import time
import cv2
import math


def main():
    arx_nav_robot = AutoNav_Robot()
    try:
        # go
        # -- get cup start --
        arx_nav_robot.arx.step_lift(18.0)
        K = load_intrinsics()
        T_cam2ref = load_cam2ref()
        pt_ref = None
        pick_prompt = "the highest cup"
        color, depth = arx_nav_robot.get_color_depth()
        if color is None or depth is None:
            print("Failed to get color or depth image.")
        u, v = predict_point_from_rgb(
            color,
            text_prompt=pick_prompt,
            assume_bgr=False,
        )
        predicted_px = (int(round(u)), int(round(v)))
        # TODO 深度check
        # raw_depth= depth[predicted_px[1], predicted_px[0]]
        # 得到XYZ
        pt_ref = pixel_to_ref_point(
            predicted_px, depth, K, T_cam2ref)
        action_seq = build_pick_cup_sequence(pt_ref)
        for act in action_seq:
            arx_nav_robot.arx.step(act)
        arx_nav_robot.arx.step_lift(15.0)
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

        for (u, v) in points:
            v += 95
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

        print(path_xy[:3])

        arx_nav_robot.follow_path(path_xy[:3], lookahead=0.12, v_max=0.12)

        # -- turn right --
        print("Turn right......")
        arx_nav_robot.run_for_1s(chz=-0.5, duration=20.6 / 4.0, record=False)

        color, depth = arx_nav_robot.get_color_depth()
        points = arx_nav_robot.detect_goal(color)

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
                    chx=0.5, duration=(action_content - 0.45)/0.064)
            elif action == "rotate":
                if action_content <= 0:
                    arx_nav_robot.run_for_1s(
                        chz=-0.5, duration=-(action_content + math.pi / 12.0)/(0.5 * 2*math.pi / 20.6))
                else:
                    arx_nav_robot.run_for_1s(
                        chz=0.5, duration=action_content/(0.5 * 2*math.pi / 20.6))

        # -- turn right start--
        arx_nav_robot.run_for_1s(chz=-0.5, duration=20.6 / 4.0, record=False)
        time.sleep(10.0)
        # -- turn right end--

        # -- forward a little start --

        # -- forward a little end --

        # -- put cup start --

        # -- put cup end --

        # # return
        # print("Final turn right......")
        # arx_nav_robot.run_for_1s(chz=-0.5, duration=10.8)

        # # -- turn left --
        # color, depth = arx_nav_robot.get_color_depth()

        # points = arx_nav_robot.turn_left_corner(color)

        # # visualize
        # order_num = 0.0

        # revised_points = []

        # for (u, v) in points:
        #     v += 75
        #     v = min(v, 470)
        #     cv2.circle(
        #         color,
        #         center=(int(u), int(v)),
        #         radius=5,
        #         color=(order_num, order_num, 255 - order_num),
        #         thickness=-1  # -1 表示实心圆
        #     )
        #     order_num += 30
        #     revised_points.append((u, v))

        # cv2.imwrite("../Testdata4Nav/test_3.png", color)

        # # time.sleep(15.0)

        # time.sleep(10.0)

        # path_xy = []
        # pw_all = []

        # # -- pixel to wolrd point --
        # for point in revised_points:
        #     Pw = arx_nav_robot.pixel_to_pw(point, depth)
        #     path_xy.append((Pw[0] - 0.40, Pw[1])) ## bias to the front claw
        #     pw_all.append(Pw)
        #     # time.sleep(1.0)

        # print(path_xy[:3])

        # arx_nav_robot.follow_path(path_xy[:3], lookahead=0.12, v_max=0.06)

        # # -- turn left pi/4 --
        # print("Turn left a little......")
        # arx_nav_robot.run_for_1s(chz=0.5, duration=20.6 / 4.0)

        # # point on the ground
        # color, depth = arx_nav_robot.get_color_depth()
        # points = arx_nav_robot.detect_goal(color)

        # cv2.circle(
        #     color,
        #     center=(int(points[0][0]), int(points[0][1])),
        #     radius=5,
        #     color=(0, 0, 255),
        #     thickness=-1  # -1 表示实心圆
        # )

        # cv2.imwrite("../Testdata4Nav/test_4.png", color)

        # time.sleep(10.0)

        # goal_pw = arx_nav_robot.pixel_to_pw(points[0], depth)
        # start = (0, 0)
        # goal = (goal_pw[0], -goal_pw[1])

        # path = [start, goal]
        # actions = path_to_actions(path)
        # actions = merge_forward_actions(actions)

        # # 移动到目标点
        # for action, action_content in actions:
        #     if action == "forward":
        #         arx_nav_robot.run_for_1s(chx=0.5, duration=(action_content - 0.45)/0.064)
        #         # time.sleep(duration_time)
        #     elif action == "rotate":
        #         if action_content <= 0:
        #             arx_nav_robot.run_for_1s(chz=-0.5, duration=-action_content/(0.5 * 2*math.pi / 20.6))
        #             # time.sleep(duration_time)
        #         else:
        #             arx_nav_robot.run_for_1s(chz=0.5, duration=action_content/(0.5 * 2*math.pi / 20.6))

        # return

        # -- turn right --
        arx_nav_robot.run_for_1s(chz=-0.5, duration=(20.6 * 3.0) / 4.0)

        # -- return origin path --
        return_path = [(x, y)
                       for (x, y, _) in reversed(arx_nav_robot.pose_log)]
        print(return_path)
        time.sleep(10.0)
        arx_nav_robot.follow_path(
            return_path,
            lookahead=0.12,
            v_max=0.12
        )

        # -- final turn left --
        arx_nav_robot.run_for_1s(chz=0.5, duration=20.6 / 2.0)

        # -- put cup start --
        arx_nav_robot.arx.step_lift(20.0)
        K = load_intrinsics()
        T_cam2ref = load_cam2ref()
        pt_ref = None
        place_prompt = "the black round coaster"
        color, depth = arx_nav_robot.get_color_depth()
        u, v = predict_point_from_rgb(
            color,
            text_prompt=place_prompt,
            assume_bgr=False,
        )
        predicted_px = (int(round(u)), int(round(v)))
        # TODO 深度check
        # raw_depth= depth[predicted_px[1], predicted_px[0]]
        # 得到XYZ
        pt_ref = pixel_to_ref_point(
            predicted_px, depth, K, T_cam2ref)
        action_seq = build_place_cup_sequence(pt_ref)
        for act in action_seq:
            arx_nav_robot.arx.step(act)
        arx_nav_robot.arx.step_lift(15.0)
        # -- put cup end --

    finally:
        arx_nav_robot.arx.close()


if __name__ == "__main__":
    main()
