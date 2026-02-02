from arx5_arm_msg.msg._robot_cmd import RobotCmd
from arm_control.msg._pos_cmd import PosCmd
import sys
sys.path.append("../ARX_Realenv/ROS2")  # noqa
from arx_ros2_env import ARXRobotEnv  # noqa


def main():
    arx = ARXRobotEnv(
        control_mode="pos",
        camera_type="all",
        camera_view=("camera_h",),
        img_size=(640, 480))
    arx.reset()
    arx.step_lift(14.0)

    # 在此处添加杯子放置规划的代码逻辑


if __name__ == "__main__":
