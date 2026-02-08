from autonav_robot import AutoNav_Robot
from nav_utils import path_to_actions, merge_forward_actions, index_resample
import time
import cv2
import math

def main():
    arx_nav_robot = AutoNav_Robot()
    try:
        # time.sleep(3.0)
        # color, depth = arx_nav_robot.get_color_depth()
        
        # cv2.imwrite("../Testdata4Nav/test_return_corner.png", color)
        
        user_instruction = """
        Now I need you to navigate to the bubble tea preparation area.
        To bypass the current table, first turn left and then move forward-right; And Then you will find the red dot.
        Once the red dot is in sight, head directly toward it. Upon reaching the red dot, perform a turn so that the bubble tea preparation area (which will be on your right) is directly in front of you.
        """
        # go
        action_return = arx_nav_robot.nav_plan(user_instruction)

        # back
        arx_nav_robot.nav_back(action_return)

        # -- put cup start --

        # -- put cup end --

    finally:
        arx_nav_robot.arx.close()

if __name__ == "__main__":
    main()