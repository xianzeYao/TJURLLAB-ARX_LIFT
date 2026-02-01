import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from arx_pointing import predict_multi_points_from_rgb, predict_point_from_rgb
import sys
sys.path.append("../ARX_Realenv/ROS2")  # noqa
from arx_ros2_env import ARXRobotEnv  # noqa


def main():
    color = cv2.imread("../Testdata4Mani/multicup.png")
    points = predict_multi_points_from_rgb(
        image=color,
        all_prompt="Pick the top cup and place it to the coaster with the smallest number.",
    )

    for (u, v) in points:
        cv2.circle(
            color,
            center=(int(u), int(v)),
            radius=5,
            color=(0, 255, 0),
            thickness=-1  # -1 表示实心圆
        )
    cv2.imshow("Predicted Points", color)
    cv2.waitKey(0)
    cv2. destroyAllWindows()
    cv2.imwrite("../Testdata4Mani/multicup_out.png", color)


if __name__ == "__main__":
    main()
