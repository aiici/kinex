"""MediaPipe Pose 追踪引擎 — 修复版

改动:
  · 骨长不再硬编码, 标定时按真实标定点计算 (self._bones)
  · 骨长惩罚改为分级容差 (短骨蹬踏时 2D 投影变化大, 不再被过度惩罚)
  · 没有标定骨长时, 不做骨长优化, 直接用候选最近者, 不会用错误的常数
  · 推荐用法: app.py 里反光点追踪做主路径, 只调用 track() 取粗位置救场
"""
import os
import sys
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

SIDE_L = {11: 3, 13: 2, 15: 1, 23: 4, 25: 5, 27: 6, 29: 7, 31: 8}
SIDE_R = {12: 3, 14: 2, 16: 1, 24: 4, 26: 5, 28: 6, 30: 7, 32: 8}

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from utils.frozen_path import resource_path
_MODEL_PATH = resource_path('pose_analysis', 'pose_landmarker_full.task')

_BONE_PAIRS = ((4, 5), (5, 6), (6, 7), (6, 8), (7, 8))


class MediaPipeTracker:
    """MediaPipe Pose 追踪引擎"""

    def __init__(self, cfg):
        self.cfg = cfg
        self._calib_pos = {}
        self._bones = {}
        self._side = None
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
        self._bones = {}
        self._side = None
        # 不重置 _frame_idx: detect_for_video 要求时间戳单调递增

    def calibrate(self, calib_pos):
        """标定: 保存反光点位置并计算真实骨长"""
        self._calib_pos = {int(j): (float(p[0]), float(p[1]))
                           for j, p in calib_pos.items()}
        self._side = None
        self._bones = {}
        for a, b in _BONE_PAIRS:
            if a in self._calib_pos and b in self._calib_pos:
                pa, pb = self._calib_pos[a], self._calib_pos[b]
                self._bones[(a, b)] = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))

    @property
    def calibrated(self):
        return bool(self._calib_pos)

    def track(self, frame_bgr):
        """追踪一帧, 返回 {jid_int: (x, y)} (像素坐标, 仅 MediaPipe 粗位置)"""
        if self._detector is None:
            return {}
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = self._frame_idx * 33
        self._frame_idx += 1
        result = self._detector.detect_for_video(mp_img, ts)
        if not result.pose_landmarks:
            return {}
        lm = result.pose_landmarks[0]

        if self._side is None:
            vis_l = sum(lm[i].visibility for i in SIDE_L)
            vis_r = sum(lm[i].visibility for i in SIDE_R)
            self._side = 'L' if vis_l >= vis_r else 'R'
        side = SIDE_L if self._side == 'L' else SIDE_R

        mp_pos = {}
        for mpi, jid in side.items():
            mp_pos[int(jid)] = (lm[mpi].x * w, lm[mpi].y * h)
        return mp_pos

    def track_with_refine(self, frame_bgr, gray):
        """追踪 + 反光点精化. 返回 (pixels, source)"""
        mp_pos = self.track(frame_bgr)
        if not mp_pos:
            return {}, {}

        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        joint_cands = {}
        all_cands = []
        for jid, (px, py) in mp_pos.items():
            r = 8.0 if jid == 4 else 6.0
            cands = self._find_markers_near(gray, px, py, r, se)
            joint_cands[jid] = cands
            for (mx, my, d) in cands:
                all_cands.append((d, int(jid), mx, my))

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

        if self._bones:
            if all(jid in mp_pos for jid in (6, 7, 8)):
                pixels, source = self._optimize_foot(
                    pixels, source, joint_cands, mp_pos)
            if 5 in mp_pos and 6 in pixels:
                pixels, source = self._optimize_chain(
                    5, 6, pixels, source, joint_cands, mp_pos)
            if 4 in mp_pos and 5 in pixels:
                pixels, source = self._optimize_chain(
                    4, 5, pixels, source, joint_cands, mp_pos)

        for jid, (px, py) in mp_pos.items():
            if jid not in pixels:
                pixels[jid] = (int(px), int(py))
                source[jid] = 'mp'
        return pixels, source

    @staticmethod
    def _bone_cost(d, L):
        """分级容差的骨长惩罚 (与 MarkerAssociator 一致), 封顶 9"""
        tol = 0.35 * L + 12.0 if L <= 60 else 0.15 * L + 2.0
        return min((abs(d - L) / tol) ** 2, 9.0)

    @staticmethod
    def _cand_points(jid, joint_cands, mp_pos):
        c = []
        for (mx, my, _d) in joint_cands.get(jid, [])[:5]:
            pt = (int(mx), int(my))
            if pt not in c:
                c.append(pt)
        if not c:
            c.append((int(mp_pos[jid][0]), int(mp_pos[jid][1])))
        return c

    def _optimize_foot(self, pixels, source, joint_cands, mp_pos):
        """6/7/8 骨长约束最优分配; 代价 = 0.5*骨长 + MediaPipe 距离/30"""
        cands = {j: self._cand_points(j, joint_cands, mp_pos) for j in (6, 7, 8)}
        mp_pts = {j: (int(mp_pos[j][0]), int(mp_pos[j][1])) for j in (6, 7, 8)}
        best_cost, best = 1e9, None
        for p6 in cands[6]:
            for p7 in cands[7]:
                if p7 == p6:
                    continue
                for p8 in cands[8]:
                    if p8 == p6 or p8 == p7:
                        continue
                    assign = {6: p6, 7: p7, 8: p8}
                    bone_cost = 0.0
                    for (a, b), L in self._bones.items():
                        if a in assign and b in assign:
                            pa, pb = assign[a], assign[b]
                            bone_cost += self._bone_cost(
                                np.hypot(pa[0] - pb[0], pa[1] - pb[1]), L)
                    mp_cost = sum(
                        np.hypot(pt[0] - mp_pts[j][0], pt[1] - mp_pts[j][1]) / 30.0
                        for j, pt in assign.items())
                    cost = 0.5 * bone_cost + mp_cost
                    if cost < best_cost:
                        best_cost, best = cost, assign
        if best is not None:
            for j in (6, 7, 8):
                pixels[j] = best[j]
                source[j] = 'mp' if best[j] == mp_pts[j] else 'marker'
        return pixels, source

    def _optimize_chain(self, jid, anchor, pixels, source, joint_cands, mp_pos):
        """膝(5, 锚点6) / 髋(4, 锚点5): 选 jid-anchor 距离最接近标定骨长的候选"""
        L = self._bones.get((jid, anchor))
        if L is None:
            return pixels, source
        pa = pixels[anchor]
        mp_pt = (int(mp_pos[jid][0]), int(mp_pos[jid][1]))
        best_cost, best = 1e9, None
        for pt in self._cand_points(jid, joint_cands, mp_pos):
            d = np.hypot(pt[0] - pa[0], pt[1] - pa[1])
            cost = (0.5 * self._bone_cost(d, L)
                    + np.hypot(pt[0] - mp_pt[0], pt[1] - mp_pt[1]) / 30.0)
            if cost < best_cost:
                best_cost, best = cost, pt
        if best is not None:
            pixels[jid] = best
            source[jid] = 'mp' if best == mp_pt else 'marker'
        return pixels, source

    def _find_markers_near(self, gray, cx, cy, r_ref, se):
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
            if d > win:
                continue
            cands.append((lx + x1, ly + y1, d))
        cands.sort(key=lambda c: c[2])
        return cands