###########------------pose_tracker.py
"""MediaPipe Pose 追踪引擎 — AI 人体姿态估计 + 反光点精化

流程:
  1. MediaPipe Pose Landmarker 检测人体 33 个关键点
  2. 判断左/右侧 (侧面拍摄, 选 visibility 高的一侧)
  3. 映射到骑行的 8 个关节
  4. 在 MediaPipe 位置附近用 top-hat 搜索反光点 (精化)
  5. 找到反光点 → 用反光点位置 (精确)
     找不到 → 用 MediaPipe 位置 (稳定后备)
"""
import os
import sys
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# MediaPipe 33 关键点 → 骑行 8 关节映射
SIDE_L = {11: 3, 13: 2, 15: 1, 23: 4, 25: 5, 27: 6, 29: 7, 31: 8}
SIDE_R = {12: 3, 14: 2, 16: 1, 24: 4, 26: 5, 28: 6, 30: 7, 32: 8}

# 模型文件路径 (兼容 PyInstaller frozen 模式)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from utils.frozen_path import resource_path
_MODEL_PATH = resource_path('pose_analysis', 'pose_landmarker_full.task')


class MediaPipeTracker:
    """MediaPipe Pose 追踪引擎"""

    def __init__(self, cfg):
        self.cfg = cfg
        self._calib_pos = {}
        self._side = None       # 'L' / 'R' (首帧确定后固定)
        self._frame_idx = 0
        self._detector = None
        self._init_detector()

    def _init_detector(self):
        if not os.path.exists(_MODEL_PATH):
            return
        base = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self._detector = mp_vision.PoseLandmarker.create_from_options(opts)

    def reset(self):
        self._calib_pos = {}
        self._side = None
        # 不重置 _frame_idx: MediaPipe detect_for_video 要求时间戳单调递增,
        # 重置会导致后续时间戳 < 上次时间戳 → ValueError

    def calibrate(self, calib_pos):
        """标定: 保存反光点位置 {jid_int: (x, y)}"""
        self._calib_pos = {int(j): (float(p[0]), float(p[1]))
                           for j, p in calib_pos.items()}
        self._side = None
        # 不重置 _frame_idx: MediaPipe detect_for_video 要求时间戳单调递增

    @property
    def calibrated(self):
        return bool(self._calib_pos)

    def track(self, frame_bgr):
        """追踪一帧, 返回 {jid_int: (x, y)} (像素坐标)

        frame_bgr: 缩放后的 BGR 帧 (NW x NH)
        """
        if self._detector is None:
            return {}
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = self._frame_idx * 33  # 毫秒时间戳 (30fps → 33ms/frame)
        self._frame_idx += 1
        result = self._detector.detect_for_video(mp_img, ts)
        if not result.pose_landmarks:
            return {}
        lm = result.pose_landmarks[0]

        # 判断左/右侧 (首帧确定, 后续固定)
        if self._side is None:
            vis_l = sum(lm[i].visibility for i in SIDE_L)
            vis_r = sum(lm[i].visibility for i in SIDE_R)
            self._side = 'L' if vis_l >= vis_r else 'R'
        side = SIDE_L if self._side == 'L' else SIDE_R

        # 映射到 8 关节
        mp_pos = {}
        for mpi, jid in side.items():
            x = lm[mpi].x * w
            y = lm[mpi].y * NH_factor(lm[mpi], w, h)
            mp_pos[int(jid)] = (x, y)
        return mp_pos

    def track_with_refine(self, frame_bgr, gray):
        """追踪 + 反光点精化 (全局贪心匹配 + 足部骨骼约束修正)

        返回 (pixels, source):
          pixels: {jid_int: (x, y)}
          source: {jid_int: 'marker' / 'mp'}
        """
        mp_pos = self.track(frame_bgr)
        if not mp_pos:
            return {}, {}

        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        # 1. 每个关节收集所有候选反光点
        joint_cands = {}  # {jid: [(x, y, d), ...]}
        all_cands = []    # [(d, jid, x, y), ...]
        for jid, (px, py) in mp_pos.items():
            # 4号点(髋) MediaPipe偏差大(~24px), 用更大搜索窗口
            r = 8.0 if jid == 4 else 6.0
            cands = self._find_markers_near(gray, px, py, r, se)
            joint_cands[jid] = cands
            for (mx, my, d) in cands:
                all_cands.append((d, int(jid), mx, my))

        # 2. 全局贪心: 按距离升序, 逐个分配
        all_cands.sort()
        used_markers = set()
        used_joints = set()
        pixels = {}
        source = {}
        for d, jid, mx, my in all_cands:
            if jid in used_joints:
                continue
            key = (int(mx) // 5, int(my) // 5)
            if key in used_markers:
                continue
            pixels[jid] = (int(mx), int(my))
            source[jid] = 'marker'
            used_joints.add(jid)
            used_markers.add(key)

        # 3a. 足部三关节(6,7,8)骨骼约束最优分配
        if all(jid in mp_pos for jid in (6, 7, 8)):
            pixels, source = self._optimize_foot(
                pixels, source, joint_cands, mp_pos)

        # 3b. 膝关节(5)骨骼约束: 在6确定后, 选5-6距离最接近标定的候选
        if 5 in mp_pos and 6 in pixels:
            pixels, source = self._optimize_knee(
                pixels, source, joint_cands, mp_pos)

        # 3c. 髋关节(4)骨骼约束: 在5确定后, 选4-5距离最接近标定的候选
        if 4 in mp_pos and 5 in pixels:
            pixels, source = self._optimize_hip(
                pixels, source, joint_cands, mp_pos)

        # 4. 没匹配到反光点的关节用 MediaPipe 位置
        for jid, (px, py) in mp_pos.items():
            if jid not in pixels:
                pixels[jid] = (int(px), int(py))
                source[jid] = 'mp'
        return pixels, source

    # 足部骨骼长度 (px)
    _FOOT_BONE = {(6, 7): 25.0, (6, 8): 53.0, (7, 8): 53.0}

    def _optimize_foot(self, pixels, source, joint_cands, mp_pos):
        """对 6/7/8 做骨骼约束最优分配

        枚举所有候选组合 (含 MediaPipe 位置), 代价 = 骨骼偏差*3 + MP距离*1。
        """
        # 收集候选: 只用反光点候选 (不含 MediaPipe 位置, 避免 mp_cost=0 干扰)
        cands = {}
        for jid in (6, 7, 8):
            c = []
            if jid in joint_cands:
                for (mx, my, _d) in joint_cands[jid][:5]:
                    pt = (int(mx), int(my))
                    if pt not in c:
                        c.append(pt)
            if not c:  # 没有反光点候选时, 用 MediaPipe 位置兜底
                c.append((int(mp_pos[jid][0]), int(mp_pos[jid][1])))
            cands[jid] = c

        # 枚举所有 3 元组
        mp_pts = {jid: (int(mp_pos[jid][0]), int(mp_pos[jid][1]))
                  for jid in (6, 7, 8)}
        best_cost = 1e9
        best = None
        for p6 in cands[6]:
            for p7 in cands[7]:
                if p7 == p6:
                    continue
                for p8 in cands[8]:
                    if p8 == p6 or p8 == p7:
                        continue
                    assign = {6: p6, 7: p7, 8: p8}
                    # 骨骼长度偏差
                    bone_cost = 0.0
                    for (a, b), L in self._FOOT_BONE.items():
                        pa, pb = assign[a], assign[b]
                        d = np.hypot(pa[0] - pb[0], pa[1] - pb[1])
                        bone_cost += abs(d - L) / L
                    # 到 MediaPipe 位置的距离 (防止关节间互换)
                    mp_cost = 0.0
                    for jid, pt in assign.items():
                        mp_cost += np.hypot(pt[0] - mp_pts[jid][0],
                                            pt[1] - mp_pts[jid][1]) / 30.0
                    cost = bone_cost * 3.0 + mp_cost
                    if cost < best_cost:
                        best_cost = cost
                        best = assign

        if best is not None:
            for jid in (6, 7, 8):
                pt = best[jid]
                mp_pt = (int(mp_pos[jid][0]), int(mp_pos[jid][1]))
                pixels[jid] = pt
                source[jid] = 'mp' if pt == mp_pt else 'marker'
        return pixels, source

    def _optimize_knee(self, pixels, source, joint_cands, mp_pos):
        """对 5 做骨骼约束: 5-6 距离 ≈ 116px

        在 6 确定后, 选 5 的候选使 5-6 距离最接近标定值。
        """
        p6 = pixels[6]
        L56 = 116.0
        mp_pt = (int(mp_pos[5][0]), int(mp_pos[5][1]))

        # 收集候选: 只用反光点候选 (不含 MediaPipe 位置)
        cands = []
        if 5 in joint_cands:
            for (mx, my, _d) in joint_cands[5][:5]:
                pt = (int(mx), int(my))
                if pt not in cands:
                    cands.append(pt)
        if not cands:  # 没有反光点候选时, 用 MediaPipe 位置兜底
            cands.append(mp_pt)

        best_cost = 1e9
        best = None
        for pt in cands:
            d = np.hypot(pt[0] - p6[0], pt[1] - p6[1])
            bone_cost = abs(d - L56) / L56
            mp_cost = np.hypot(pt[0] - mp_pt[0], pt[1] - mp_pt[1]) / 30.0
            cost = bone_cost * 3.0 + mp_cost
            if cost < best_cost:
                best_cost = cost
                best = pt

        if best is not None:
            pixels[5] = best
            source[5] = 'mp' if best == mp_pt else 'marker'
        return pixels, source

    def _optimize_hip(self, pixels, source, joint_cands, mp_pos):
        """对 4 做骨骼约束: 4-5 距离 ≈ 114px

        在 5 确定后, 选 4 的候选使 4-5 距离最接近标定值。
        """
        p5 = pixels[5]
        L45 = 114.0
        mp_pt = (int(mp_pos[4][0]), int(mp_pos[4][1]))

        # 收集候选: 只用反光点候选 (不含 MediaPipe 位置)
        cands = []
        if 4 in joint_cands:
            for (mx, my, _d) in joint_cands[4][:5]:
                pt = (int(mx), int(my))
                if pt not in cands:
                    cands.append(pt)
        if not cands:  # 没有反光点候选时, 用 MediaPipe 位置兜底
            cands.append(mp_pt)

        best_cost = 1e9
        best = None
        for pt in cands:
            d = np.hypot(pt[0] - p5[0], pt[1] - p5[1])
            bone_cost = abs(d - L45) / L45
            mp_cost = np.hypot(pt[0] - mp_pt[0], pt[1] - mp_pt[1]) / 30.0
            cost = bone_cost * 3.0 + mp_cost
            if cost < best_cost:
                best_cost = cost
                best = pt

        if best is not None:
            pixels[4] = best
            source[4] = 'mp' if best == mp_pt else 'marker'
        return pixels, source

    def _find_markers_near(self, gray, cx, cy, r_ref, se):
        """在 (cx, cy) 附近搜索反光点, 返回 [(x, y, 距离), ...] (按距离升序)"""
        H, W = gray.shape[:2]
        win = int(max(30, r_ref * 5))
        x1, y1 = max(0, int(cx - win)), max(0, int(cy - win))
        x2, y2 = min(W, int(cx + win) + 1), min(H, int(cy + win) + 1)
        if x2 <= x1 or y2 <= y1:
            return []
        sub = gray[y1:y2, x1:x2]
        tophat = cv2.morphologyEx(sub, cv2.MORPH_TOPHAT, se)
        peak = int(tophat.max())
        if peak < 20:
            return []
        _, bw = cv2.threshold(tophat, int(peak * 0.4), 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ox, oy = cx - x1, cy - y1
        cands = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 3:
                continue
            (lx, ly), lr = cv2.minEnclosingCircle(cnt)
            if not (2.0 <= lr <= 25):
                continue
            d = np.hypot(lx - ox, ly - oy)
            if d > win:  # 超出搜索半径
                continue
            cands.append((lx + x1, ly + y1, d))
        cands.sort(key=lambda c: c[2])
        return cands


def NH_factor(lm, w, h):
    """MediaPipe 的 y 坐标已经是归一化的, 直接乘 h"""
    return h
