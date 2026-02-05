import os
import time
import threading
import queue
from typing import Dict, Iterable, Optional, Literal

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image

from arm_control.msg._pos_cmd import PosCmd
from arx5_arm_msg.msg._robot_cmd import RobotCmd
from arx5_arm_msg.msg._robot_status import RobotStatus

import cv2
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer


class RobotIO(Node):
    def __init__(self, camera_type: Literal["color", "depth", "all"] = "all", camera_view: Iterable[str] = ("camera_l", "camera_h")):
        super().__init__('robot_io')
        self.bridge = CvBridge() if CvBridge is not None else None

        self.cmd_pub_l = self.create_publisher(RobotCmd, 'arm_cmd_l', 5)
        self.cmd_pub_r = self.create_publisher(RobotCmd, 'arm_cmd_r', 5)
        self.cmd_pub_base = self.create_publisher(PosCmd, 'ARX_VR_L', 5)

        self.latest_height = 0.0
        self.latest_status: Dict[str, Optional[RobotStatus]] = {
            "left": None, "right": None}
        self.latest_base: Optional[PosCmd] = None

        self.status_snapshot: Optional[Dict[str, Optional[RobotStatus]]] = None
        self.status_lock = threading.Lock()

        self.create_subscription(
            RobotStatus, 'arm_status_l', lambda msg: self._on_status('left', msg), 5)
        self.create_subscription(
            RobotStatus, 'arm_status_r', lambda msg: self._on_status('right', msg), 5)
        self.sta_cmd_sub = self.create_subscription(
            PosCmd, 'body_information', self._on_base_status, 1)

        self.camera_type = camera_type  # color/depth/all
        self.camera_view = list(camera_view) if camera_view else []
        self.cam_lock = threading.Lock()
        self.latest_images: Dict[str, Image] = {}
        self.subscribed_topics = []
        self.save_queue: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self.saver_thread = threading.Thread(
            target=self._save_worker, daemon=True)
        self.saver_thread.start()

        subs: list[Subscriber] = []
        labels: list[str] = []

        types = [
            "color", "aligned_depth_to_color"] if camera_type == "all" else [camera_type]
        for cam in self.camera_view:
            for typ in types:
                if "aligned" in typ:
                    topic = f"/{cam}_namespace/{cam}/aligned_depth_to_color/image_raw"
                else:
                    topic = f"/{cam}_namespace/{cam}/{typ}/image_rect_raw"
                subs.append(Subscriber(self, Image, topic, qos_profile=5))
                labels.append(f"{cam}_{typ}")
                self.subscribed_topics.append(topic)
        self.labels = labels
        if subs:
            self.sync = ApproximateTimeSynchronizer(
                subs, queue_size=5, slop=0.02)
            self.sync.registerCallback(self._on_images_status)
            self.get_logger().info(
                f"Subscribed camera topics: {self.subscribed_topics}")
        else:
            self.get_logger().warn("No camera subscriptions configured.")
        self.get_logger().info(
            f"Init camera_view={self.camera_view}, types={types}")

    def _on_status(self, side: str, msg: RobotStatus):
        with self.status_lock:
            self.latest_status[side] = msg

    def _on_base_status(self, msg: PosCmd):
        with self.status_lock:
            self.latest_base = msg
            self.latest_height = float(msg.height)

    def _on_images_status(self, *msgs):

        with self.status_lock:
            snap = dict(self.latest_status)
            snap["base"] = self.latest_base
            self.status_snapshot = snap
        with self.cam_lock:
            for label, msg in zip(self.labels, msgs):
                self.latest_images[label] = msg

    def send_base_msg(self, cmd: PosCmd):
        """Send base command."""
        if not rclpy.ok():
            try:
                self.get_logger().warn("ROS not ready, base command not sent")
            except Exception:
                pass
            return False
        sub_count = self.cmd_pub_base.get_subscription_count()
        if sub_count == 0:
            try:
                self.get_logger().warn(f"No base subscribers, base command not sent")
            except Exception:
                pass
            return False
        # track latest height even when publishing commands
        try:
            self.latest_height = float(cmd.height)
        except Exception:
            pass
        self.cmd_pub_base.publish(cmd)
        return True

    def send_control_msg(self, side: str, cmd: RobotCmd):
        pub = self.cmd_pub_l if side == "left" else self.cmd_pub_r
        if not rclpy.ok():
            try:
                self.get_logger().warn("ROS not ready, arm command not sent")
            except Exception:
                pass
            return False
        sub_count = pub.get_subscription_count()
        if sub_count == 0:
            try:
                self.get_logger().warn(
                    f"{side} no subscribers, arm command not sent")
            except Exception:
                pass
            return False
        pub.publish(cmd)
        return True

    def get_robot_status(self):
        # TODO:考虑删除此方法，直接使用 get_camera(return_status=True) 获取快照
        with self.status_lock:
            status = dict(self.latest_status)
            status["base"] = self.latest_base
            return status

    def get_camera(self, save_dir: Optional[str] = None, target_size: Optional[tuple[int, int]] = None,
                   return_status: bool = False):
        """Return latest approx-synced camera frames, optional status snapshot."""
        if self.bridge is None:
            print("CvBridge not initialized, cannot decode images.")
            return (dict(), self.status_snapshot) if return_status else dict()
        frames = dict()
        with self.cam_lock:
            items = list(self.latest_images.items())
        for key, msg in items:
            try:
                img = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding='passthrough')
                if "depth" not in key:
                    img = img[:, :, ::-1]
            except Exception as exc:  # pragma: no cover
                self.get_logger().warn(f"{key} decode failed: {exc}")
                continue
            if target_size:
                img = cv2.resize(img, target_size)
            frames[key] = img
            if save_dir:
                stamp = getattr(msg, "header", None)
                if stamp:
                    ts = stamp.stamp.sec + stamp.stamp.nanosec * 1e-9
                else:
                    ts = time.time()
                self.save_queue.put((save_dir, key, ts, img))
        if return_status:
            with self.status_lock:
                # prefer snapshot if available; otherwise use latest
                if self.status_snapshot is not None:
                    snap = dict(self.status_snapshot)
                else:
                    snap = dict(self.latest_status)
                # ensure base present
                if "base" not in snap:
                    snap["base"] = self.latest_base
            return frames, snap
        return frames

    def get_camera_keys(self):
        with self.cam_lock:
            return list(self.latest_images.keys())

    def _save_worker(self):
        while True:
            task = self.save_queue.get()
            if task is None:
                break
            save_dir, key, ts, img = task
            try:
                os.makedirs(save_dir, exist_ok=True)
                base = os.path.join(save_dir, f"{key}_{ts}")
                if "depth" in key:
                    np.save(base + ".npy", img)
                    depth_float = img.astype(np.float32)
                    finite = depth_float[np.isfinite(depth_float)]
                    # if finite.size == 0:
                    #     continue
                    vmax = np.percentile(finite, 99)
                    # if vmax <= 1e-6:
                    #     continue
                    vis = cv2.convertScaleAbs(depth_float, alpha=255.0 / vmax)
                    cv2.imwrite(base + "_vis.png", vis,
                                [cv2.IMWRITE_PNG_COMPRESSION, 0])
                else:
                    cv2.imwrite(base + ".png", img,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
            except Exception as exc:  # pragma: no cover
                try:
                    self.get_logger().warn(f"save {key} failed: {exc}")
                except Exception:
                    pass
            finally:
                self.save_queue.task_done()

    def stop_saver(self):
        self.save_queue.put(None)
        if getattr(self, "saver_thread", None) is not None:
            self.saver_thread.join(timeout=2.0)


def start_robot_io(camera_type: Literal["color", "depth", "all"] = "all", camera_view: Iterable[str] = ("camera_l", "camera_h")):
    node = RobotIO(camera_type=camera_type, camera_view=camera_view)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    t = threading.Thread(target=executor.spin, daemon=True)
    t.start()
    return node, executor, t


def build_observation(
    camera_all: Dict[str, Image] | Dict,
    status_all: Dict[str, object] | None,
    include_arm: bool = True,
    include_camera: bool = True,
    include_base: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Pack status和相机到扁平观测字典。

    include_arm / include_camera / include_base 控制是否写入对应部分，默认全开。
    """
    obs: Dict[str, np.ndarray] = {}

    if include_arm and isinstance(status_all, dict):
        lstatus = status_all.get("left")
        rstatus = status_all.get("right")
        if lstatus is not None:
            obs["left_end_pos"] = np.array(lstatus.end_pos, dtype=np.float32)
            obs["left_joint_pos"] = np.array(
                lstatus.joint_pos, dtype=np.float32)
            obs["left_joint_cur"] = np.array(
                lstatus.joint_cur, dtype=np.float32)
            obs["left_joint_vel"] = np.array(
                lstatus.joint_vel, dtype=np.float32)
        if rstatus is not None:
            obs["right_end_pos"] = np.array(rstatus.end_pos, dtype=np.float32)
            obs["right_joint_pos"] = np.array(
                rstatus.joint_pos, dtype=np.float32)
            obs["right_joint_cur"] = np.array(
                rstatus.joint_cur, dtype=np.float32)
            obs["right_joint_vel"] = np.array(
                rstatus.joint_vel, dtype=np.float32)

    if include_base and isinstance(status_all, dict):
        base_status = status_all.get("base")
        if base_status is not None:
            obs["base_height"] = np.array(
                [base_status.height], dtype=np.float32)
            obs["base_wheel1"] = np.array(
                [base_status.temp_float_data[1]], dtype=np.float32)  # 后轮
            obs["base_wheel2"] = np.array(
                [base_status.temp_float_data[2]], dtype=np.float32)  # 右前
            obs["base_wheel3"] = np.array(
                [base_status.temp_float_data[3]], dtype=np.float32)  # 左前
    if include_camera:
        # Attach camera frames as numpy arrays
        for key, msg in (camera_all or {}).items():
            if msg is None:
                continue
            try:
                img = msg
                if isinstance(msg, np.ndarray):
                    img_np = msg
                else:
                    img_np = np.asarray(img)
                if "color" in key:
                    img_np = np.asarray(img_np, dtype=np.uint8)
                obs[key] = img_np
            except Exception:
                # Skip broken frame
                continue
    return obs


def compute_interp_steps(curr_obs: Dict[str, np.ndarray],
                         action: Dict[str, np.ndarray],
                         max_v_xyz: float,
                         max_v_rpy: float,
                         duration_per_step: float,
                         min_steps_per_action: int,
                         min_steps_gripper: int) -> tuple[Dict[str, tuple[int, int]], Dict[str, bool]]:
    """
    Compute interpolation steps for each side.

    Returns:
        steps_by_side: dict[side] -> (pose_steps, gripper_steps)
        pose_changed: dict[side] -> bool indicating pose change (affects success check)
    """
    steps_by_side: Dict[str, tuple[int, int]] = {}
    pose_changed: Dict[str, bool] = {}
    # Small-move deadband and short-move single-step thresholds.
    # Units: xyz in meters, rpy in radians, gripper in raw units.
    eps_xyz = 5e-4
    eps_rpy = 5e-4
    eps_grip = 5e-4
    short_xyz = max_v_xyz * duration_per_step
    short_rpy = max_v_rpy * duration_per_step
    for side in ("left", "right"):
        target = action.get(side)
        curr_end = curr_obs.get(f"{side}_end_pos")
        curr_joint = curr_obs.get(f"{side}_joint_pos")
        if target is None or curr_end is None or curr_joint is None:
            continue
        start = np.concatenate([np.array(curr_end, dtype=np.float32),
                                [float(curr_joint[6])]])
        target_arr = target if isinstance(
            target, np.ndarray) else np.array(target, dtype=np.float32)
        diff = np.abs(target_arr - start)
        diff_xyz = float(diff[:3].max())
        diff_rpy = float(diff[3:6].max())
        if diff_xyz < eps_xyz and diff_rpy < eps_rpy:
            pose_steps = 0
            pose_changed[side] = False
        else:
            need_steps = [
                int(np.ceil(diff_xyz / (max_v_xyz * duration_per_step))),
                int(np.ceil(diff_rpy / (max_v_rpy * duration_per_step))),
            ]
            pose_steps = max(min_steps_per_action, max(need_steps))
            pose_changed[side] = True

        grip_steps = 0
        delta_g = diff[6]
        if delta_g > 0:
            if delta_g <= eps_grip:
                grip_steps = 0
            elif delta_g <= 1e-3:
                grip_steps = 1
            else:
                grip_steps = max(min_steps_gripper, max(1, pose_steps // 3))

        steps_by_side[side] = (pose_steps, grip_steps)
    return steps_by_side, pose_changed


def interpolate_action(curr_obs: Dict[str, np.ndarray], action: Dict[str, np.ndarray],
                       steps: Dict[str, tuple[int, int]]) -> Dict[str, list[np.ndarray]]:
    """Interpolate end-effector+gripper sequences for both arms.

    steps: per side (pose_steps, gripper_steps); length is per-side max; beyond length hold last value.
    """
    results: Dict[str, list[np.ndarray]] = {}

    def smoothstep5(n: int) -> np.ndarray:
        """Quintic smoothstep 6t^5-15t^4+10t^3 for smooth start/stop."""
        if n <= 1:
            return np.array([1.0], dtype=np.float32)
        t = np.linspace(0.0, 1.0, n)
        return t**3 * (10 - 15 * t + 6 * t * t)

    for side in ("left", "right"):
        target = action.get(side)
        if target is None:
            continue
        if not isinstance(target, np.ndarray):
            target = np.array(target, dtype=np.float32)
        curr_end = curr_obs.get(f"{side}_end_pos")
        curr_joint = curr_obs.get(f"{side}_joint_pos")
        if curr_end is None or curr_joint is None:
            continue

        curr_gripper = float(curr_joint[6])
        start_pose = np.array(curr_end, dtype=np.float32)
        start_grip = curr_gripper

        pose_steps, grip_steps = steps.get(side, (0, 0))
        max_steps = max(pose_steps, grip_steps)
        if max_steps <= 0:
            continue

        pose_alphas = smoothstep5(pose_steps)
        grip_alphas = smoothstep5(grip_steps)
        seq: list[np.ndarray] = []
        for i in range(max_steps):
            pose_alpha = pose_alphas[min(
                i, pose_steps - 1)] if pose_steps > 0 else 1.0
            grip_alpha = grip_alphas[min(
                i, grip_steps - 1)] if grip_steps > 0 else 1.0
            pose_interp = start_pose + (target[:6] - start_pose) * pose_alpha
            grip_interp = start_grip + \
                (float(target[6]) - start_grip) * grip_alpha
            seq.append(np.concatenate([pose_interp, [grip_interp]]))
        results[side] = seq
    return results
