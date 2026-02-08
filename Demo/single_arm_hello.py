
import time
import sys
sys.path.append("../ARX_Realenv/ROS2")  # noqa
from arx_ros2_env import ARXRobotEnv  # noqa
import numpy as np


def hello(arx: ARXRobotEnv, side="left", close_robot=True):
    try:
        arx.step_lift(16.0)
        hello_base = np.array([0.05, 0.0, 0.4], dtype=np.float32)
        if side == "left":
            lift_action = hello_base.
            lift_action = {side: np.array(
                [0.4, 0, 0.25, 0, 0, 0, -2.5], dtype=np.float32)}
        else:
            lift_action = {side: np.array(
                [0.4, 0, 0.25, 0, 0, 0, -2.5], dtype=np.float32)}
        arx.step(lift_action)
        time.sleep(5.0)
        arx.step(shake_action)
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
    arx.reset()
    hello(arx, side="left", close_robot=True)


if __name__ == "__main__":
    main()
