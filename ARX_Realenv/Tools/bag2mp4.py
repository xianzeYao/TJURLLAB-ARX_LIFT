import argparse
import os
import re

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


def sanitize_topic(topic: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", topic).strip("_")


class MultiWriter(Node):
    def __init__(self, topics, out_dir, fps):
        super().__init__("bag_to_mp4_multi")
        self.bridge = CvBridge()
        self.out_dir = out_dir
        self.fps = fps
        self.writers = {}
        self.topics = topics
        os.makedirs(out_dir, exist_ok=True)

        for t in topics:
            self.create_subscription(Image, t, self._make_cb(t), 10)

    def _make_cb(self, topic):
        def cb(msg: Image):
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if topic not in self.writers:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_path = os.path.join(
                    self.out_dir, f"{sanitize_topic(topic)}.mp4"
                )
                self.writers[topic] = cv2.VideoWriter(out_path, fourcc, self.fps, (w, h))
                self.get_logger().info(f"Writing {out_path} ({w}x{h}@{self.fps})")
            self.writers[topic].write(frame)
        return cb

    def destroy_node(self):
        for w in self.writers.values():
            w.release()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", nargs="+", required=True)
    parser.add_argument("--out_dir", default="Testdata4mani/video")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    rclpy.init()
    node = MultiWriter(args.topics, args.out_dir, args.fps)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
