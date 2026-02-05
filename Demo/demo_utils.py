from __future__ import annotations

import re
from typing import List, Optional

import cv2
import numpy as np

from arx_pointing import predict_multi_points_from_rgb
from pick_place_cup_motion import build_pick_cup_sequence, build_place_cup_sequence
from pick_place_straw_motion import (
    build_pick_straw_sequence,
    build_place_straw_sequence,
)


def extract_numbered_sentences(raw: Optional[str]) -> List[str]:
    """提取形如 '1. xxx' / '2) xxx' / '3- xxx' 的编号句子（行内行间都可）。"""
    if not raw:
        return []
    # 去掉代码块包裹
    raw_clean = re.sub(
        r"```(?:json|python)?\n?(.*?)\n?```", r"\1", raw, flags=re.DOTALL
    )
    steps: List[str] = []

    # 行内 / 行间匹配：1. xxx 2) yyy 3- zzz
    inline_matches = re.finditer(
        r"(\d+)[\.\)\-]\s*(.+?)(?=(?:\d+[\.\)\-])|$)",
        raw_clean,
        flags=re.DOTALL,
    )
    for m in inline_matches:
        steps.append(m.group(2).strip())

    # 去重保持顺序
    seen = set()
    uniq_steps = []
    for s in steps:
        if s not in seen:
            seen.add(s)
            uniq_steps.append(s)
    return uniq_steps


def do_replan(color_img: np.ndarray, planning_prompt: str) -> List[str]:
    raw_result = predict_multi_points_from_rgb(
        color_img,
        text_prompt="",
        all_prompt=planning_prompt,
        assume_bgr=False,
        return_raw=True,
        temperature=0.0,
    )
    if isinstance(raw_result, tuple):
        _, pick_answer_text = raw_result
    else:
        pick_answer_text = None

    pick_plan = extract_numbered_sentences(pick_answer_text)
    if not pick_plan:
        # 递归重试
        return do_replan(color_img=color_img, planning_prompt=planning_prompt)
    return pick_plan


def draw_text_lines(
    img: np.ndarray,
    lines: List[str],
    origin: tuple[int, int] = (10, 30),
    line_height: int = 22,
    color: tuple[int, int, int] = (0, 0, 255),
    scale: float = 0.5,
    thickness: int = 2,
) -> None:
    x, y = origin
    for i, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (x, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
        )


def draw_point_label(
    img: np.ndarray,
    label: str,
    pos: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    draw_text_lines(
        img,
        [label],
        origin=(pos[0] + 6, pos[1] - 6),
        line_height=18,
        color=color,
        scale=0.55,
        thickness=2,
    )


def execute_pick_place_cup_sequence(
    arx,
    pick_ref: Optional[np.ndarray],
    place_ref: Optional[np.ndarray],
    arm: str,
    do_pick: bool = True,
    do_place: bool = True,
    go_home: bool = True,
) -> None:
    if do_pick:
        if pick_ref is None:
            raise ValueError("pick_ref 为空")
        pick_seq = build_pick_cup_sequence(pick_ref, arm=arm)
        for act in pick_seq:
            arx.step(act)
    if do_place:
        if place_ref is None:
            raise ValueError("place_ref 为空")
        place_seq = build_place_cup_sequence(place_ref, arm=arm)
        for act in place_seq:
            arx.step(act)
    if go_home:
        arx._go_to_initial_pose()


def execute_pick_place_straw_sequence(
    arx,
    pick_ref: Optional[np.ndarray],
    place_ref: Optional[np.ndarray],
    arm: str,
    do_pick: bool = True,
    do_place: bool = True,
    go_home: bool = True,
) -> None:
    if do_pick:
        if pick_ref is None:
            raise ValueError("pick_ref 为空")
        pick_seq = build_pick_straw_sequence(pick_ref, arm=arm)
        for act in pick_seq:
            arx.step(act)
    if do_place:
        if place_ref is None:
            raise ValueError("place_ref 为空")
        place_seq = build_place_straw_sequence(place_ref, arm=arm)
        for act in place_seq:
            arx.step(act)
    if go_home:
        arx._go_to_initial_pose()
