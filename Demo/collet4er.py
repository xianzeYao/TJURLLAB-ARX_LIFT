"""
交互式采集脚本：
- `w` / `s` 控制升降高度（上/下）
- 鼠标左键在图像上打点，实时显示红点和编号
- `y` 保存一次样本（无点图、有点图、json）
- `r` 清空当前点
- `q` 或 `ESC` 退出

每次保存会在输出目录下创建一个子文件夹，包含：
- image_raw.png
- image_marked.png
- meta.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import sys

THIS_DIR = Path(__file__).resolve().parent
ROS2_DIR = (THIS_DIR / "../ARX_Realenv/ROS2").resolve()
if str(ROS2_DIR) not in sys.path:
    sys.path.insert(0, str(ROS2_DIR))

from arx_ros2_env import ARXRobotEnv  # noqa: E402


def find_color_frame(obs: dict, prefer_cam: str = "camera_h") -> tuple[Optional[str], Optional[np.ndarray]]:
    candidates = [(k, v) for k, v in obs.items()
                  if isinstance(v, np.ndarray) and "color" in k]
    if not candidates:
        return None, None

    for key, img in candidates:
        if prefer_cam in key:
            return key, img
    return candidates[0]


def draw_points(image: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    canvas = image.copy()
    for idx, (x, y) in enumerate(points, start=1):
        cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)
        label = str(idx)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(
            canvas,
            label,
            (x - tw // 2, y + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def save_sample(
    out_dir: Path,
    raw_img: np.ndarray,
    marked_img: np.ndarray,
    points: list[tuple[int, int]],
    base_height: float,
    camera_key: str,
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    sample_dir = out_dir / f"sample_{ts}"
    sample_dir.mkdir(parents=True, exist_ok=False)

    raw_path = sample_dir / "image_raw.png"
    marked_path = sample_dir / "image_marked.png"
    cv2.imwrite(str(raw_path), raw_img)
    cv2.imwrite(str(marked_path), marked_img)

    h, w = raw_img.shape[:2]
    meta = {
        "timestamp": ts,
        "camera_key": camera_key,
        "image_size": {"width": int(w), "height": int(h)},
        "base_height": float(base_height),
        "points": [
            {"id": int(i), "x": int(x), "y": int(y)}
            for i, (x, y) in enumerate(points, start=1)
        ],
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return sample_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="ARX 交互式打点采集")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("../Testdata4Mani/collect4er"), help="保存根目录")
    parser.add_argument("--camera", type=str,
                        default="camera_h", help="优先使用的相机名")
    parser.add_argument("--img-w", type=int, default=640)
    parser.add_argument("--img-h", type=int, default=480)
    parser.add_argument("--height-step", type=float,
                        default=0.5, help="每次调高/调低的高度步长")
    parser.add_argument("--height-min", type=float, default=0.0)
    parser.add_argument("--height-max", type=float, default=20.0)
    parser.add_argument("--skip-home", action="store_true", help="不执行 reset()")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = ARXRobotEnv(
        camera_type="color",
        camera_view=(args.camera,),
        img_size=(args.img_w, args.img_h),
    )

    if not args.skip_home:
        env.reset()

    time.sleep(1.0)
    obs0 = env.get_observation(
        include_arm=False, include_camera=False, include_base=True)
    base_height = float(
        obs0.get("base_height", np.array([0.0], dtype=np.float32))[0])

    win = "collet4er"
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x), int(y)))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    print("操作说明: w上升, s下降, y保存, r清空点, q/ESC退出")

    try:
        while True:
            obs = env.get_observation(
                include_arm=False, include_base=True, include_camera=True)
            cam_key, frame = find_color_frame(obs, prefer_cam=args.camera)
            if frame is None or cam_key is None:
                cv2.waitKey(1)
                continue

            raw_img = frame.copy()
            marked = draw_points(raw_img, points)

            tip = f"h={base_height:.2f} | points={len(points)} | w/s height | y save"
            cv2.putText(marked, tip, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, marked)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("w"):
                base_height = min(
                    args.height_max, base_height + args.height_step)
                env.step_lift(base_height)
                print(f"lift -> {base_height:.2f}")
            elif key == ord("s"):
                base_height = max(
                    args.height_min, base_height - args.height_step)
                env.step_lift(base_height)
                print(f"lift -> {base_height:.2f}")
            elif key == ord("y"):
                folder = save_sample(
                    out_dir=args.out_dir,
                    raw_img=raw_img,
                    marked_img=marked,
                    points=points,
                    base_height=base_height,
                    camera_key=cam_key,
                )
                points.clear()
                print(f"saved -> {folder}")
            elif key == ord("r"):
                points.clear()
                print("points cleared")

    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
