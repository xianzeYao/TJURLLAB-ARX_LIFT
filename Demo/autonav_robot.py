from nav_utils import depth_to_meters, get_key
# from qwen3_vl_8b_tool import predict_point_from_rgb
from arx_pointing import predict_multi_points_from_rgb

import numpy as np
import threading
import time
import math
from pathlib import Path

from arm_control.msg._pos_cmd import PosCmd

import sys
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

BIAS_REF2CAM = np.array([0.25, 0.24, 0.0, 0.0])

class AutoNav_Robot():
    def __init__(self, camera_type="all", camera_view=("camera_h",), img_size=(640, 480)):
        # -- arx robot env --
        self.arx = ARXRobotEnv(
            duration_per_step=1.0 / 20.0,
            min_steps_per_action=60,
            min_steps_gripper=20,

            max_v_xyz=0.1,
            max_v_rpy=0.1,

            camera_type=camera_type,
            camera_view=camera_view,
            img_size=img_size,
        )

        time.sleep(1.0)
        obs = self.arx.reset()

        # -- emergency stop --
        self.running = True
        
        # -- initial pose information --
        self.x_r = 0.0
        self.y_r = 0.0
        self.theta_r = 0.0

        # -- update frequency --
        self.dt = 0.05

        # -------- emergency stop ------
        self.running = True
        self.kb_thread = threading.Thread(
            target=self.keyboard_listener,
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
        self.default_height = 19.0

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
        msg.mode1 = 2
        msg.height = self.default_height
        self.arx.node.send_base_msg(msg)

    def run_for_1s(self, chx=0.0, chy=0.0, chz=0.0, duration=1.0, record=True):
        start = time.time()
        # while time.time() - start < duration:
        msg = PosCmd()
        msg.chx = chx
        msg.chy = chy
        msg.chz = chz
        # 每次发送控制指令时，都附带期望的高度，这样底盘就会一直维持该高度
        msg.height = self.default_height
        msg.mode1 = 1

        self.arx.node.send_base_msg(msg)

        while self.running and time.time() - start < duration:
            time.sleep(0.03)   # 小步 sleep，方便中断

        self.stop()

        if record and self.running:
            self.action_log.append((chx, chy, chz, duration))

    def turn_right_corner(self, color):
        prompt = """The task is: go forward and then make a decisive right turn at the nearest table corner.

Requirements for the trajectory:
- Give exactly 5 points on the ground forming a movement trajectory.
- The trajectory starts at the bottom-center area of the image (robot position).
- The robot first moves mostly straight forward.
- The right turn should happen mainly in the second half of the trajectory.
- The turning curvature should be relatively high: the last 3 points should shift clearly to the right with smaller forward progress.
- Avoid a long, gradual curve. The turn should be compact and decisive.

Direction information:
- The robot is located at the bottom center of the image.
- The robot is facing upward in the image.
- Forward motion corresponds to decreasing y values in the image.
- Rightward motion corresponds to increasing x values in the image.

The points must on ground, not on table.

Output format:
Return the result in JSON format as:
[{"point_2d": [x, y]}]"""

        points = predict_multi_points_from_rgb(
            color,
            all_prompt=prompt,
            base_url="http://172.28.102.11:22014/v1",
            model_name="Qwen3-VL-8B-Instruct",
            api_key="EMPTY",
            assume_bgr=False
        )

        return points
    
    def detect_goal(self, color):
        prompt = """Provide one or more points coordinate of objects region this sentence describes: tennis ball on the ground.
        Output format: Return the result in JSON format as:
        [ 
            {"point_2d": [x, y]}
        ]
        """

        points = predict_multi_points_from_rgb(
                color,
                all_prompt=prompt,
                base_url="http://172.28.102.11:22014/v1",
                model_name="Qwen3-VL-8B-Instruct",
                api_key="EMPTY",
            )

        return points
    
    def turn_left_corner(self, color):
        prompt = """The task is: move forward briefly and then make a decisive, compact left turn at the nearest left table corner without touching the table.

Requirements for the trajectory:
- Give exactly 5 points on the ground forming a movement trajectory.
- The trajectory starts at the bottom and slightly right area of the image (right of the robot position).

Trajectory structure (IMPORTANT):
- Points 1–2: move mostly forward with only small lateral change.
- Points 3–5: perform the left turn.

Strict turning constraints:
- The left turn must be sharp and compact, NOT gradual.
- For points 3–5, lateral left movement must dominate over forward movement.
- For points 3–5, the change in x (leftward) must be significantly larger than the change in y (forward).
- Ideally, points 3–5 should have very little forward progress compared to points 1–2.

Strong prohibitions:
- Do NOT make a long, smooth curve.
- Do NOT continue moving forward while turning.
- Do NOT spread the turn over the entire trajectory.

Direction information:
- The robot is located at the bottom center of the image.
- The robot is facing upward in the image.
- Forward motion corresponds to decreasing y values.
- Leftward motion corresponds to decreasing x values.

Ground constraint:
- All points must be on the ground (floor), not on the table or table edges.

Self-check before output:
- If the turning points still show clear forward motion, make the turn tighter.

Output format:
Return the result in JSON format as:
[
  {"point_2d": [x, y]}
]
"""

        points = predict_multi_points_from_rgb(
            color,
            all_prompt=prompt,
            base_url="http://172.28.102.11:22014/v1",
            model_name="Qwen3-VL-8B-Instruct",
            api_key="EMPTY",
            assume_bgr=False
        )

        return points
    
    # emergency read keyboard
    def keyboard_listener(self):
        while self.running:
            try:
                ch = get_key()
            except Exception:
                continue

            if ch == 'q':
                print("Key 'q' pressed! Emergency stop!")
                self.running = False
                self.stop()
                break

    
    # compute pose
    def update_pose(self, v, omega):

        # 更新机器人位姿
        self.x_r += v * math.cos(self.theta_r) * self.dt
        self.y_r += v * math.sin(self.theta_r) * self.dt
        self.theta_r += omega * self.dt

    # get robot pose
    def get_robot_pose(self):
        return self.x_r, self.y_r, self.theta_r

    def get_lookahead_point(self, path_xy, lookahead):
        x_r, y_r, theta_r = self.get_robot_pose()
        for index, (x_g, y_g) in enumerate(path_xy):
            # change coordinate
            x_t = math.cos(-theta_r)*(x_g - x_r) - math.sin(-theta_r)*(y_g - y_r)
            y_t = math.sin(-theta_r)*(x_g - x_r) + math.cos(-theta_r)*(y_g - y_r)
            dist = math.hypot(x_t, y_t)
            if dist >= lookahead and x_t > 0:
                # print(index)
                return x_t, y_t, dist
        # return the final point
        x_t = math.cos(-theta_r)*(path_xy[-1][0] - x_r) - math.sin(-theta_r)*(path_xy[-1][1] - y_r)
        y_t = math.sin(-theta_r)*(path_xy[-1][0] - x_r) + math.cos(-theta_r)*(path_xy[-1][1] - y_r)
        dist = math.hypot(x_t, y_t)
        return x_t, y_t, dist
    
    # pure pursuite follow path
    def follow_path(self, path_xy, lookahead=0.6, v_max=0.12, v_min=0.09, omega_max=0.2):
        # reset pose
        self.x_r = 0.0
        self.y_r = 0.0
        self.theta_r = 0.0


        rate = self.dt
        while self.running:
            # 获取目标点
            x_t, y_t, dist = self.get_lookahead_point(path_xy, lookahead)

            # 非常接近终点，允许真正停下
            if dist < 0.05:
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
            msg.height = self.default_height
            msg.mode1 = 1
            self.arx.node.send_base_msg(msg)
            print(math.sqrt(v / 0.24))

            # 更新位姿
            self.update_pose(v, omega)
            time.sleep(rate)

        self.stop()
