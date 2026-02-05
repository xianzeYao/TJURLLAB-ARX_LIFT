from autonav_robot import AutoNav_Robot
from nav_utils import path_to_actions, merge_forward_actions, index_resample
from dual_cup_pick_plannning import dual_arm_pick_planning

import time
import cv2
import math


def main():
    arx_nav_robot = AutoNav_Robot()
    try:
        #
        user_instruction = """
        Now I need you to navigate to the bubble tea preparation area.
        To bypass the current table, first turn left and then move forward-right; And Then you will find the red dot.
        Once the red dot is in sight, head directly toward it. Upon reaching the red dot, perform a turn so that the bubble tea preparation area (which will be on your right) is directly in front of you.
        """
        # dual arm pick planning
        dual_arm_pick_planning(
            arx_nav_robot.arx, reset_robot=False, close_robot=False, no_last_place=True)
        # go
        arx_nav_robot.nav_plan(user_instruction)
        # place empty cup to making area and pick the bubble tea
        
        arx_nav_robot.run_for_1s(chx=-0.5, duration=2.2)

        arx_nav_robot.run_for_1s(chz=-0.5, duration=20.6 - 20.6 / 2.5)

        arx_nav_robot.motion_inversion()

        arx_nav_robot.run_for_1s(chx=0.5, duration=2.5)

        arx_nav_robot.run_for_1s(chz=0.5, duration=10.3)

        arx_nav_robot.run_for_1s(chx=0.5, duration=1.5)

        # place the bubble tea cup to the customer area and insert a straw

    finally:
        arx_nav_robot.arx.close()


if __name__ == "__main__":
    main()
