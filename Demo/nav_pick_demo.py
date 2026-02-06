from autonav_robot import AutoNav_Robot
from nav_utils import path_to_actions, merge_forward_actions, index_resample
from Demo.dual_cup_pick_planning import dual_arm_pick_planning
from single_arm_pick_place import single_arm_pick_place
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
        # dual arm pick planning and execute
        dual_arm_pick_planning(
            arx_nav_robot.arx, goal="red cup", reset_robot=False, close_robot=False, no_last_place=True)
        # nav go
        arx_nav_robot.arx.step_lift(15.0)
        arx_nav_robot.nav_plan(user_instruction)
        # place empty cup to making area and pick the bubble tea
        arx_nav_robot.arx.step_lift(17.0)
        single_arm_pick_place(
            arx_nav_robot.arx,
            arm="right",
            pick_prompt="",
            place_prompt="the center part of the brown coaster on the right side",
            reset_robot=False,
            close_robot=False,
            debug=False,
        )

        time.sleep(10.0)
        # pick bubble tea cup
        single_arm_pick_place(
            arx_nav_robot.arx,
            arm="left",
            pick_prompt="the cup on the left brown coaster",
            place_prompt="",
            reset_robot=False,
            close_robot=False,
            debug=False,
            go_home=False,
        )
        # nav back
        arx_nav_robot.run_for_1s(chx=-0.5, duration=2.2)
        arx_nav_robot.run_for_1s(chz=-0.5, duration=20.6 - 20.6 / 2.5)
        arx_nav_robot.motion_inversion()
        arx_nav_robot.run_for_1s(chx=0.5, duration=2.5)
        arx_nav_robot.run_for_1s(chz=0.5, duration=10.3)
        arx_nav_robot.run_for_1s(chx=0.5, duration=1.5)
        # place the bubble tea cup to the customer area and insert a straw
        arx_nav_robot.arx.step_lift(14.0)
        single_arm_pick_place(
            arx_nav_robot.arx,
            arm="left",
            pick_prompt="",
            place_prompt="the center of the brown coaster",
            reset_robot=False,
            close_robot=False,
            debug=False,
        )
    finally:
        arx_nav_robot.arx.close()


if __name__ == "__main__":
    main()
