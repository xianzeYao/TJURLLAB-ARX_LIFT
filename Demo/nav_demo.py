from autonav_robot import AutoNav_Robot
from nav_utils import path_to_actions, merge_forward_actions, index_resample
import time
import cv2
import math

def main():
    arx_nav_robot = AutoNav_Robot()
    try:
        # go
        # -- get cup start --

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
            u += 90
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

        arx_nav_robot.follow_path(path_xy[:3], lookahead=0.12, v_max=0.12, reach_dis=0.09, show_index=True)

        # time.sleep(10.0)

        # -- turn right --
        print("Turn right......")
        # arx_nav_robot.run_for_1s(chz=-1.0, duration=20.6 / 6.0)
        points, detect_flag, color = arx_nav_robot.turn_right_until_see_goal(goal="red circular landmark on the ground", max_angle=math.pi/2.0)
        if not detect_flag:
            raise RuntimeError(f"未找到目标")
        # arx_nav_robot.run_for_1s(chz=-0.5, duration=20.6 / 3.0)

        # color, depth = arx_nav_robot.get_color_depth()
        # points = arx_nav_robot.detect_goal(color)

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
                arx_nav_robot.run_for_1s(chx=0.5, duration=(action_content - 0.5)/0.064)
            elif action == "rotate":
                if action_content <= 0:
                    arx_nav_robot.run_for_1s(chz=-0.5, duration=float(-(action_content)/(0.5 * 2*math.pi / 20.6)))
                else:
                    arx_nav_robot.run_for_1s(chz=0.5, duration=action_content/(0.5 * 2*math.pi / 20.6))

        # -- turn right start--
        # arx_nav_robot.turn_right_until_see_goal(goal="black round coaster on table", max_angle=(math.pi*2.0/3.0))
        arx_nav_robot.run_for_1s(chx=-0.5, duration=20.6/12.0)
        points, detect_flag, color = arx_nav_robot.turn_right_until_see_goal(goal="black round coaster", max_angle=(math.pi*2.0/3.0))
        if not detect_flag:
            raise RuntimeError(f"未找到目标")
        cv2.circle(
            color,
            center=(int(points[0][0]), int(points[0][1])),
            radius=5,
            color=(0, 0, 255),
            thickness=-1  # -1 表示实心圆
        )

        cv2.imwrite("../Testdata4Nav/test_3.png", color)
        

        # arx_nav_robot.run_for_1s(chz=-0.5, duration=20.6 / 2.5)
        # time.sleep(10.0)
        # -- turn right end--

        # foward a little
        arx_nav_robot.run_for_1s(chx=0.5, duration=1.5)

        # -- turn right --
        arx_nav_robot.run_for_1s(chz=-0.5, duration=(20.6 * 2.0) / 3.0)

        # -- return origin path -- 
        raw_path = [(x, y) for (x, y, _) in reversed(arx_nav_robot.pose_log)]
        # return_path = index_resample(
        #     raw_path,
        #     num_points=25,
        #     gamma=1.8
        # )
        return_path = raw_path
        print(return_path)
        # time.sleep(10.0)
        arx_nav_robot.follow_path(
            return_path,
            lookahead=0.12,
            v_max=0.12,
            show_index=True,
            reach_dis=0.07
        )

        # -- final turn left --
        arx_nav_robot.run_for_1s(chz=0.5, duration=20.6 / 2.0)

        # -- put cup start --

        # -- put cup end --

    finally:
        arx_nav_robot.arx.close()

if __name__ == "__main__":
    main()