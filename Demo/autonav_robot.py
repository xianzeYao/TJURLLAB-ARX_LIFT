from nav_utils import depth_to_meters, get_key, extract_actions, merge_forward_actions, path_to_actions
# from qwen3_vl_8b_tool import predict_point_from_rgb
from arx_pointing import predict_multi_points_from_rgb

import numpy as np
import threading
import time
import math
from pathlib import Path
import cv2

from arm_control.msg._pos_cmd import PosCmd

import sys
import select
import termios
sys.path.append("../ARX_Realenv/ROS2")  # noqa

from arx_ros2_env import ARXRobotEnv

# ===============================
# 相机内参
# ===============================
K = np.array([
    [391.9335632324219, 0.0, 320.5389099121094],
    [0.0, 391.6839294433594, 239.18849182128906],
    [0.0, 0.0, 1.0]
])

FX, FY = K[0, 0], K[1, 1]
CX, CY = K[0, 2], K[1, 2]

# ===============================
# 相机 → base_link 外参
# ===============================

T_CAM2REF = np.array([
    [-0.01022451527760726, -0.5071681372702741, 0.861786481574838, 0.019333535519116728],
    [-0.9997376669412061, -0.012479673613300601, -0.019205599325708644, -0.23751223916353767],
    [0.020495282049587504, -0.8617567744348205, -0.5069074916879823, 0.13595597780350663],
    [0.0, 0.0, 0.0, 1.0]
])

BIAS_REF2CAM = np.array([0.0, 0.48, 0.0, 0.0])

class AutoNav_Robot():
    def __init__(self, camera_type="all", camera_view=("camera_h",), img_size=(640, 480)):
        # -- arx robot env --
        self.arx = ARXRobotEnv(
            duration_per_step=1.0 / 20.0,
            min_steps_per_action=60,
            min_steps_gripper=20,

            max_v_xyz=0.15,
            max_v_rpy=0.3,

            camera_type=camera_type,
            camera_view=camera_view,
            img_size=img_size,
        )

        obs = self.arx.reset()

        # -- emergency stop --
        self.running = True
        
        # -- initial pose information --
        self.x_r = 0.0
        self.y_r = 0.0
        self.theta_r = 0.0
        self.pose_log = []

        # -- update frequency --
        self.dt = 0.05

        # -------- emergency stop ------
        # self.running = True
        self.kb_thread = threading.Thread(
            target=self._safe_key_listener,
            daemon=True
        )
        self.kb_thread.start()

        # ---------- save path ----------
        self.save_root = Path(
            "/home/arx/Robotbase_base/data/camera_record"
        )
        self.save_root.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.save_dir = self.save_root / ts
        self.save_dir.mkdir()

        self.rgb_dir = self.save_dir / "rgb"
        self.depth_dir = self.save_dir / "depth"
        self.rgb_dir.mkdir()
        self.depth_dir.mkdir()

        # -- save content --
        self.save_frequency = 100
        self.action_log = []   # [(chx, chy, chz, duration)]
        self.rgb_frames = []
        self.frame_id = 0

        # -- default height --
        self.default_height = 15.0

        self.lift_to_default_height()

    def get_color_depth(self):
        frames = self.arx.node.get_camera(target_size=(640, 480), return_status=False)
        color = frames.get("camera_h_color")
        depth = frames.get("camera_h_aligned_depth_to_color")
        return color, depth

    # slowly raise up
    def lift_to_default_height(self):
        self.arx.step_lift(self.default_height)
    
    # change pixel to world point
    def pixel_to_pw(self, pixel, depth):
        u, v = pixel
        z = depth_to_meters(float(depth[int(v), int(u)]))
        if z <= 0:
            return None, None
        
        # 像素 → 相机坐标
        x = (u - CX) * z / FX
        y = (v - CY) * z / FY
        Pc = np.array([x, y, z, 1.0], dtype=np.float64)

        # 相机 → ref → base_link
        Pw_right = T_CAM2REF @ Pc
        Pw = Pw_right + BIAS_REF2CAM

        return Pw
    
    # motion
    def stop(self):
        msg = PosCmd()
        msg.chx = msg.chy = msg.chz = 0.0
        base_status = self.arx.node.get_robot_status().get("base")
        msg.height = float(
            base_status.height) if base_status is not None else 0.0
        msg.mode1 = 2
        self.arx.node.send_base_msg(msg)

    def run_for_1s(self, chx=0.0, chy=0.0, chz=0.0, duration=1.0, record=True):
        v = 0.24 * chx**2
        omega = chz * (2 * math.pi / 20.6)

        msg = PosCmd()
        msg.chx = chx
        msg.chy = chy
        msg.chz = chz
        base_status = self.arx.node.get_robot_status().get("base")
        msg.height = float(
            base_status.height) if base_status is not None else 0.0
        msg.mode1 = 1
        self.arx.node.send_base_msg(msg)


        if record and self.running:
            self.integrate_motion(v, omega, duration)

        self.stop()

        if record and self.running:
            self.action_log.append((chx, chy, chz, duration))

    def run_for_1s_return(self, chx=0.0, chy=0.0, chz=0.0, duration=1.0, record=True):
        msg = PosCmd()
        msg.chx = chx
        msg.chy = chy
        msg.chz = chz
        msg.mode1 = 1
        base_status = self.arx.node.get_robot_status().get("base")
        msg.height = float(
            base_status.height) if base_status is not None else 0.0
        self.arx.node.send_base_msg(msg)
        start_time = time.time()

        if duration < 0.1:
            time.sleep(duration)
            return
        
        else:
            while time.time() - start_time < duration and self.running:
                a = 1
            return

    def nav_plan(self, user_instruction):
        correct_flag = False
        prompt = f"""
        You are a robot motion planner. Current goal: {user_instruction}
        ### Instruction Rules:
        1. **Task Categories:** Every step must strictly start with the word **Turn** or **Move**.
        2. **Decomposition Logic:** - To "Go around the left side," you must first Turn Left, then Move Forward-Right to clear the obstacle and see the target.
        ### Output Format:
        1. [Action]: [Brief Description]
        2. ...
        Please give me the remaining 5 step.
        """
        while not correct_flag:
            color, depth = self.get_color_depth()

            _, generated_content = predict_multi_points_from_rgb(
                color,
                text_prompt="",
                all_prompt=prompt,
                base_url="http://172.28.102.11:22002/v1",
                model_name="Embodied-R1.5-SFT-0128",
                api_key="EMPTY",
                assume_bgr=False,
                return_raw=True
            )

            # print(generated_content)

            actions = extract_actions(generated_content)

            # print(actions)
            
            actions = ['Turn Left', 'Move Forward-Right', 'Adjust Position to Face Red Dot', 'Move Directly Toward Red Dot', 'Turn Right to Face Bubble Tea Preparation Area']

            if actions == ['Turn Left', 'Move Forward-Right', 'Adjust Position to Face Red Dot', 'Move Directly Toward Red Dot', 'Turn Right to Face Bubble Tea Preparation Area']:
                correct_flag = True

        for action in actions:
            if action == 'Turn Left':
                self.turn_left(math.pi / 2.0)
            elif action == 'Move Forward-Right':
                self.turn_right_corner()
            elif action == 'Adjust Position to Face Red Dot':
                self.run_for_1s(chz=-0.5, duration=20.6/6.0)
                self.go_to_goal("center of red circular landmark on the ground")
            elif action.startswith("Turn Right"):
                action_return = self.go_to_table()
        
        return action_return

    def turn_left(self, angle):
        print(f"Turn left 90°......")
        duration_time = 20.6 * float(angle / math.pi)
        self.run_for_1s(chz=0.5, duration=duration_time)

    def go_to_table(self):
        self.run_for_1s(chz=-0.5, duration=20.6 / 2.5)

        self.arx.step_lift(17.0)

        self.run_for_1s(chx=0.5, duration=2.2)

        color, depth = self.get_color_depth()
        points = self.detect_goal(color, "the brown round coaster on the table on the left")
        goal_pw = self.pixel_to_pw(points[0], depth)
        goal_pw[0] += 0.25
        goal_pw[1] -= 0.25
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
                    duration = max(float((-(action_content)/(0.5 * 2*math.pi / 20.6))), 0.0)
                    self.run_for_1s(chz=-0.5, duration=duration)
                    action_return = (0.5, duration)
                else:
                    duration = action_content/(0.5 * 2*math.pi / 20.6)
                    self.run_for_1s(chz=0.5, duration=action_content/(0.5 * 2*math.pi / 20.6))
                    action_return = (-0.5, duration)

        return action_return

        # time.sleep(10.0)
        # -- turn right end--

        # foward a little
        # self.run_for_1s(chx=0.5, duration=2.2)
    
    # intelligent turn right
    def turn_right_until_see_goal(self, goal, max_angle):
        # start_turn_right
        msg = PosCmd()
        msg.chx = 0.0
        msg.chy = 0.0
        msg.chz = -0.3
        msg.mode1 = 1
        self.arx.node.send_base_msg(msg)
        start_time = time.time()

        max_turn_time = max_angle / (0.6 * math.pi / 20.6)

#         detect_prompt = """Is there {goal}? If you think there is, ouput the point coordinates on the center of it; if you think there is not, the output point coordinates should be (1000, 1000).
#         Output format:
# Return the result in JSON format as:
# [
#   {"point_2d": [x, y]}
# ]""".replace("{goal}", goal)

        judge_prompt = f"""
Is there a {goal} in the picture? If you think there is no {goal}, output 'False'; if you think there is {goal}, output 'True'.
"""     
        detect_prompt = prompt_format = (
        "Provide one or more points coordinate of objects region this sentence describes: "
        f"{goal}. "
        'The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}].'
    )
        # detect_prompt = detect_prompt.replace("{goal}", goal)

        print(judge_prompt)

        print(detect_prompt)

        detect_flag = False

        while not detect_flag and time.time() - start_time < max_turn_time:
            color, depth = self.get_color_depth() 
            # h, w = color.shape[:2]
            points, generated_content = predict_multi_points_from_rgb(
                color,
                text_prompt="",
                all_prompt=judge_prompt,
                base_url="http://172.28.102.11:22002/v1",
                model_name="Embodied-R1.5-SFT-0128",
                api_key="EMPTY",
                assume_bgr=False,
                return_raw=True
            )

            print(generated_content)

            if generated_content == "True":
                break
        
        while not detect_flag and time.time() - start_time < max_turn_time:
            color, depth = self.get_color_depth() 
            h, w = color.shape[:2]
            points = predict_multi_points_from_rgb(
                color,
                text_prompt="",
                all_prompt=detect_prompt,
                base_url="http://172.28.102.11:22002/v1",
                model_name="Embodied-R1.5-SFT-0128",
                api_key="EMPTY",
                assume_bgr=False
            )

            print(points[0][0])

            if points[0][0] > w / 4.0 and points[0][0] < (w * 3.0) / 4.0:
                print(points[0])
                self.action_log.append((0.0, 0.0, -0.5, time.time() - start_time))
                detect_flag = True

        self.stop()
        if not detect_flag:
            self.action_log.append((0.0, 0.0, -0.5, time.time() - start_time))

        cv2.circle(
            color,
            center=(int(points[0][0]), int(points[0][1])),
            radius=5,
            color=(0, 0, 255),
            thickness=-1  # -1 表示实心圆
        )

        cv2.imwrite("../Testdata4Nav/test_2.png", color)
        
        return points, detect_flag, color


    def turn_right_corner(self):
        print("Turn right corner......")
        self.initialize_pose()
        color, depth = self.get_color_depth()
        prompt = """**Task**

Given an image captured from a top-mounted robot camera,Use 2D points to trace the movement trajectory as it moves.

**Trajectory requirements**

- Output **exactly 8 points** on the **ground (floor)** that form a single continuous trajectory.
- The **first point** must be at the **bottom center of the image**, representing the robot’s current position.
- The last point must be located on the right image boundary, below the vertical midpoint.
- The trajectory must represent **a clear forward motion first, followed by a right turn**.
- The **first 2–3 points** should lie approximately on a **straight forward path** before any noticeable rightward deviation.
- The right turn should **start later**, not immediately near the starting point.
Output format:
Return the result in JSON format as:
[
  {"point_2d": [x, y]}
]"""

        points = predict_multi_points_from_rgb(
            color,
            text_prompt="",
            all_prompt=prompt,
            base_url="http://172.28.102.11:22002/v1",
            model_name="Embodied-R1.5-SFT-0128",
            api_key="EMPTY",
            temperature=0.2
            # assume_bgr=False
        )

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

        path_xy = []

        # -- pixel to wolrd point --
        for point in revised_points:
            Pw = self.pixel_to_pw(point, depth)
            path_xy.append((Pw[0], Pw[1]))

        print(path_xy[:7])

        self.follow_path(path_xy[:7], lookahead=0.12, v_max=0.15, v_min=0.13, reach_dis=0.09, show_index=True)

    
    def go_to_goal(self, goal, left_side=False):
        print(f"Go to goal {goal}......")
        color, depth = self.get_color_depth()
        prompt = """Provide one or more points coordinate of objects region this sentence describes: {goal}.
        Output format: Return the result in JSON format as:
        [ 
            {"point_2d": [x, y]}
        ]
        """.replace("{goal}", goal)

        points = predict_multi_points_from_rgb(
                color,
                text_prompt="",
                all_prompt=prompt,
                base_url="http://172.28.102.11:22002/v1",
                model_name="Embodied-R1.5-SFT-0128",
                api_key="EMPTY",
                assume_bgr=False
            )

        cv2.circle(
            color,
            center=(int(points[0][0]), int(points[0][1])),
            radius=5,
            color=(0, 0, 255),
            thickness=-1  # -1 表示实心圆
        )

        # cv2.imwrite("../Testdata4Nav/test_2.png", color)

        goal_pw = self.pixel_to_pw(points[0], depth)
        if left_side:
            goal_pw[1] -= 0.24
            cv2.imwrite("../Testdata4Nav/test_3.png", color)
        else:
            cv2.imwrite("../Testdata4Nav/test_2.png", color)
        start = (0, 0)
        goal = (goal_pw[0], -goal_pw[1])

        path = [start, goal]
        actions = path_to_actions(path)
        actions = merge_forward_actions(actions)

        # -- move to goal --
        for action, action_content in actions:
            if action == "forward":
                self.run_for_1s(chx=1.0, duration=(action_content)/0.245)
            elif action == "rotate":
                if action_content <= 0:
                    self.run_for_1s(chz=-0.5, duration=max(float((-action_content/(0.5 * 2*math.pi / 20.6))) - 0.5, 0.0))
                else:
                    self.run_for_1s(chz=0.5, duration=action_content/(0.5 * 2*math.pi / 20.6))

    def detect_goal(self, color, goal):
        prompt = """Provide one point coordinate of object region this sentence describes: {goal}.
        Output format: Return the result in JSON format as:
        [ 
            {"point_2d": [x, y]}
        ]
        """.replace("{goal}", goal)

        points = predict_multi_points_from_rgb(
                color,
                text_prompt="",
                all_prompt=prompt,
                base_url="http://172.28.102.11:22002/v1",
                model_name="Embodied-R1.5-SFT-0128",
                api_key="EMPTY",
                assume_bgr=False
            )

        return points
    
    def turn_left_corner(self):
        print("Turn left corner......")
        self.initialize_pose()
        color, depth = self.get_color_depth()
        prompt = """Task
Given an image captured from a top-mounted robot camera, use 2D points to trace the movement trajectory as it moves.
Trajectory requirements
Output exactly 8 points on the ground (floor) that form a single continuous trajectory.
The first point must be at the bottom center of the image, representing the robot’s current position.
The last point must be located on the left image boundary, below the vertical midpoint (to complete the bypass).
The trajectory must represent a clear forward motion first, followed by a left turn to navigate around the table on the left.
The first 2–3 points should lie approximately on a straight forward path to establish clearance before initiating the turn.
The left turn should start mid-trajectory, angling toward the left boundary to successfully bypass the obstacle.
Surface Constraint: All points, especially the final destination point, must be located strictly within the blue floor area. Avoid any points overlapping with the table or non-floor surfaces.
Output format:
Return the result in JSON format"""

        points = predict_multi_points_from_rgb(
            color,
            text_prompt="",
            all_prompt=prompt,
            base_url="http://172.28.102.11:22002/v1",
            model_name="Embodied-R1.5-SFT-0128",
            api_key="EMPTY",
            temperature=0.2,
            assume_bgr=False
        )

        order_num = 0.0

        revised_points = []
    
        for (u, v) in points:
            v += 10
            v = min(478, v)
            cv2.circle(
                color,
                center=(int(u), int(v)),
                radius=5,
                color=(order_num, order_num, 255 - order_num),
                thickness=-1  # -1 表示实心圆
            )
            order_num += 30
            revised_points.append((u, v))

        cv2.imwrite("../Testdata4Nav/test_4.png", color)

        path_xy = []

        # -- pixel to wolrd point --
        for point in revised_points:
            Pw = self.pixel_to_pw(point, depth)
            path_xy.append((Pw[0], Pw[1]-0.24))

        print(path_xy[:7])

        self.follow_path(path_xy[:7], lookahead=0.12, v_max=0.15, v_min=0.13, reach_dis=0.09, show_index=True, return_=True)
    
    # emergency read keyboard
    def keyboard_listener(self):
        while self.running:
            try:
                ch = get_key()
            except Exception:
                continue

            if ch == 'q':
                # print("Key 'q' pressed! Emergency stop!")
                # raise RuntimeError("Key 'q' pressed! Emergency stop!")
                self.running = False
                self.arx.close() 
                # self.running = False
                self.stop()
                break
    
    # compute pose
    def initialize_pose(self):
        self.x_r = 0.0
        self.y_r = 0.0
        self.theta_r = 0.0
        self.pose_log = []

    def update_pose(self, v, omega):

        # 更新机器人位姿
        self.x_r += v * math.cos(self.theta_r) * self.dt
        self.y_r += v * math.sin(self.theta_r) * self.dt
        self.theta_r += omega * self.dt

        self.pose_log.append((self.x_r, self.y_r, self.theta_r))

    def integrate_motion(self, v, omega, duration):
        t = 0.0
        while t < duration and self.running:
            self.update_pose(v, omega)
            time.sleep(self.dt)
            t += self.dt

    # get robot pose
    def get_robot_pose(self):
        return self.x_r, self.y_r, self.theta_r

    def get_lookahead_point(self, path_xy, lookahead, _index):
        x_r, y_r, theta_r = self.get_robot_pose()
        for index, (x_g, y_g) in enumerate(path_xy):
            # change coordinate
            x_t = math.cos(-theta_r)*(x_g - x_r) - math.sin(-theta_r)*(y_g - y_r)
            y_t = math.sin(-theta_r)*(x_g - x_r) + math.cos(-theta_r)*(y_g - y_r)
            dist = math.hypot(x_t, y_t)
            if dist >= lookahead and x_t > 0 and index >= _index:
                return x_t, y_t, dist, index
        # return the final point
        x_t = math.cos(-theta_r)*(path_xy[-1][0] - x_r) - math.sin(-theta_r)*(path_xy[-1][1] - y_r)
        y_t = math.sin(-theta_r)*(path_xy[-1][0] - x_r) + math.cos(-theta_r)*(path_xy[-1][1] - y_r)
        dist = math.hypot(x_t, y_t)
        return x_t, y_t, dist, len(path_xy) - 1
    
    # pure pursuite follow path
    def follow_path(self, path_xy, lookahead=0.6, v_max=0.12, v_min=0.10, omega_max=0.2, reach_dis=0.08, return_=False, show_index=False):
        # reset pose
        # self.x_r = 0.0
        # self.y_r = 0.0
        # self.theta_r = 0.0

        index = 0
        rate = self.dt
        final_count = 0
        if return_:
            max_final_count = (math.hypot(abs(path_xy[-2][0] - path_xy[-1][0]), abs(path_xy[-2][1] - path_xy[-1][1])) / 0.03)
        else:
            max_final_count = (math.hypot(abs(path_xy[-2][0] - path_xy[-1][0]), abs(path_xy[-2][1] - path_xy[-1][1])) / 0.06)

        while self.running:
            # 获取目标点
            x_t, y_t, dist, index = self.get_lookahead_point(path_xy, lookahead, index)
            if index == len(path_xy) - 1:
                final_count += 1
            if show_index:
                print(index)

            # 非常接近终点，允许真正停下
            if dist < reach_dis or final_count>max_final_count:
                break

            # Pure Pursuit 曲率
            curvature = 2 * y_t / (dist**2 + 1e-6)
            omega = 1.2 * curvature

            # 原始速度衰减
            v = v_max * math.exp(-abs(omega))

            # -------- 独立限制线速度和角速度 --------
            v = max(min(v, v_max), v_min)             # v ∈ [v_min, v_max]
            omega = max(min(omega, omega_max), -omega_max)  # omega ∈ [-omega_max, omega_max]
            # -----------------------------------------

            # 遥控信号映射
            msg = PosCmd()
            msg.chx = math.sqrt(v / 0.24)          # 前后速度
            msg.chz = omega / (2 * math.pi / 20.6) # 正数向左
            base_status = self.arx.node.get_robot_status().get("base")
            msg.height = float(
                base_status.height) if base_status is not None else 0.0
            msg.mode1 = 1
            self.arx.node.send_base_msg(msg)
            self.action_log.append((msg.chx, msg.chy, msg.chz, rate))
            # print(math.sqrt(v / 0.24))
            # print(omega / (2 * math.pi / 20.6))

            # 更新位姿
            self.update_pose(v, omega)
            time.sleep(rate)

        self.stop()

    # Motion Inversion with Forward-only Constraint
    def motion_inversion(self):
        # turn back
        # self.run_for_1s(chz=-0.5, duration=20.6)
        # print(self.action_log)
        action_log = self.action_log[1:-5].copy()
        action_log.append(self.action_log[-4])
        for chx, chy, chz, duration in reversed(action_log):
            # ignore still motion
            if abs(chx) < 1e-3 and abs(chz) < 1e-3:
                continue

            # print((chx, chy, -chz, duration))
            
            self.run_for_1s_return(chx, chy, -chz, duration)

            if not self.running:
                break

    def _safe_key_listener(self):
        """
        safe key listener
        """
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            while self.running:
                # 0.05s超时轮询，不阻塞主线程
                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch == 'k':
                        print("\n[Emergency Stop] 'k' pressed.")
                        self.arx.close()
                        self.running = False
                        self.stop()
                        break
        finally:
            # 确保退出时终端状态恢复
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def nav_back(self, action_return):
        """
        return back
        """
        # -- adjust face --
        self.run_for_1s(chz=action_return[0], duration=action_return[1])

        # -- step back a little --
        self.run_for_1s(chx=-0.5, duration=5.5)
        
        # -- turn right --
        self.run_for_1s(chz=-0.5, duration=20.6/2.0)

        # -- go to table corner --
        self.run_for_1s(chx=0.5, duration=11.5)

        # -- turn left corner --
        self.turn_left_corner()

        # -- turn left to see landmark --
        self.run_for_1s(chz=0.5, duration=20.6/6.0)

        # -- go to landmark
        self.go_to_goal("center of red circular landmark on the ground", left_side=True)

        # -- turn left to face the table --
        self.run_for_1s(chz=0.5, duration=((20.6*5.0)/12.0-0.5))
