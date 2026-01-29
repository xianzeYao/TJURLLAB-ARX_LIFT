from autonav_robot import AutoNav_Robot
from nav_utils import path_to_actions, merge_forward_actions
import time
import cv2
import math

def main():
    arx_nav_robot = AutoNav_Robot()
    try:
        # go
        time.sleep(1.0)
        color, depth = arx_nav_robot.get_color_depth()
        points = arx_nav_robot.turn_right_corner(color)

        # -- visualize --
        order_num = 0.0

        revised_points = []
    
        for (u, v) in points:
            v += 25
            cv2.circle(
                color,
                center=(int(u), int(v)),
                radius=5,
                color=(order_num, order_num, 255 - order_num),
                thickness=-1  # -1 表示实心圆
            )
            order_num += 30
            revised_points.append((u, v))

        cv2.imwrite("test_1.png", color)

        time.sleep(10.0)

        path_xy = []
        pw_all = []

        # -- pixel to wolrd point --
        for point in revised_points:
            Pw = arx_nav_robot.pixel_to_pw(point, depth)
            path_xy.append((Pw[0], Pw[1]))
            pw_all.append(Pw)
            # time.sleep(1.0)

        print(path_xy[:3])

        arx_nav_robot.follow_path(path_xy[:3], lookahead=0.12, v_max=0.12)

        # -- turn right --
        print("Turn right......")
        arx_nav_robot.run_for_1s(chz=-0.5, duration=20.6 / 2.0)

        color, depth = arx_nav_robot.get_color_depth()
        points = arx_nav_robot.detect_goal(color)

        cv2.circle(
            color,
            center=(int(points[0][0]), int(points[0][1])),
            radius=5,
            color=(0, 0, 255),
            thickness=-1  # -1 表示实心圆
        )

        cv2.imwrite("test_2.png", color)

        time.sleep(10.0)

        goal_pw = arx_nav_robot.pixel_to_pw(points[0], depth)
        start = (0, 0)
        goal = (goal_pw[0], -goal_pw[1])

        path = [start, goal]
        actions = path_to_actions(path)
        actions = merge_forward_actions(actions)

        # -- move to goal --
        for action, action_content in actions:
            if action == "forward":
                arx_nav_robot.run_for_1s(chx=0.5, duration=(action_content - 0.45)/0.064)
            elif action == "rotate":
                if action_content <= 0:
                    arx_nav_robot.run_for_1s(chz=-0.5, duration=-action_content/(0.5 * 2*math.pi / 20.6))
                else:
                    arx_nav_robot.run_for_1s(chz=0.5, duration=action_content/(0.5 * 2*math.pi / 20.6))


        # -- turn left --
        print("Revised turn right......")
        arx_nav_robot.run_for_1s(chz=-0.5, duration=20.6 / 6.0)

        time.sleep(3.0)

        
        # return 
        print("Final turn right......")
        arx_nav_robot.run_for_1s(chz=-0.5, duration=10.3)

        # -- turn left --
        color, depth = arx_nav_robot.get_color_depth()

        points = arx_nav_robot.turn_left_corner(color)

        # visualize
        order_num = 0.0

        revised_points = []
    
        for (u, v) in points:
            v += 100
            v = min(v, 470)
            cv2.circle(
                color,
                center=(int(u), int(v)),
                radius=5,
                color=(order_num, order_num, 255 - order_num),
                thickness=-1  # -1 表示实心圆
            )
            order_num += 30
            revised_points.append((u, v))

        cv2.imwrite("test_3.png", color)

        # time.sleep(15.0)

        time.sleep(10.0)

        path_xy = []
        pw_all = []

        # -- pixel to wolrd point --
        for point in revised_points:
            Pw = arx_nav_robot.pixel_to_pw(point, depth)
            path_xy.append((Pw[0] - 0.45, Pw[1])) ## bias to the front claw
            pw_all.append(Pw)
            # time.sleep(1.0)

        print(path_xy[:3])

        arx_nav_robot.follow_path(path_xy[:3], lookahead=0.12, v_max=0.12)
        
        # -- turn left pi/4 --
        print("Turn left a little......")
        arx_nav_robot.run_for_1s(chz=0.5, duration=20.6 / 4.0)

        # point on the ground
        color, depth = arx_nav_robot.get_color_depth()
        points = arx_nav_robot.detect_goal(color)

        cv2.circle(
            color,
            center=(int(points[0][0]), int(points[0][1])),
            radius=5,
            color=(0, 0, 255),
            thickness=-1  # -1 表示实心圆
        )

        cv2.imwrite("test_4.png", color)

        time.sleep(10.0)

        goal_pw = arx_nav_robot.pixel_to_pw(points[0], depth)
        start = (0, 0)
        goal = (goal_pw[0], -goal_pw[1])

        path = [start, goal]
        actions = path_to_actions(path)
        actions = merge_forward_actions(actions)

        # 移动到目标点
        for action, action_content in actions:
            if action == "forward":
                arx_nav_robot.run_for_1s(chx=0.5, duration=(action_content - 0.45)/0.064)
                # time.sleep(duration_time)
            elif action == "rotate":
                if action_content <= 0:
                    arx_nav_robot.run_for_1s(chz=-0.5, duration=-action_content/(0.5 * 2*math.pi / 20.6))
                    # time.sleep(duration_time)
                else:
                    arx_nav_robot.run_for_1s(chz=0.5, duration=action_content/(0.5 * 2*math.pi / 20.6))
    finally:
        arx_nav_robot.arx.close()

if __name__ == "__main__":
    main()