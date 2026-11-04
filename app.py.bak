# -*- coding: utf-8 -*-
"""
骑行姿态运动捕捉 GUI (2D)

使用真实视频 (pose_analysis/videos/demo.mp4) 捕捉 8 个分光点,
实时计算关节角度并可视化骨骼/编号/告警。

功能:
    - 参数配置面板: 亮度/半径/距离/圆度/角度阈值/平滑窗口, 全部可实时调整
    - 视频画面: 显示帧 + 骨骼连线 + 编号 + 角度数值 + 颜色告警
    - 首帧标定: 自动标定 (按位置启发式) 或 手动标定 (鼠标按 1~8 顺序点击)
    - 自适应亮度: 追踪时候选不足 8 个自动降低亮度阈值重试

运行:
    python pose_analysis/app.py

复用模块:
    vision.MarkerDetector / MarkerTracker  分光点检测 + 跨帧追踪
    core.models.JointID / core.geometry.angle_2d
"""

import sys
import os
import time
import threading
import itertools
from dataclasses import dataclass, field
from collections import deque

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QFormLayout, QGroupBox, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog, QSlider,
    QFrame, QSizePolicy, QDialog, QListWidget, QDialogButtonBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPoint
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent, QFont, QColor

from vision import MarkerDetector, MarkerConfig, MarkerTracker
from vision.circle_detection import find_circles_detailed, CircleCandidate
from core.models import JointID
from core.geometry import angle_2d
from pose_analysis.tracking_engine import MotionPredictor, MarkerAssociator
from pose_analysis.pose_tracker import MediaPipeTracker
from pose_analysis.metrics import MetricsCollector, METRIC_CN
from utils.frozen_path import resource_path
from capture.camera_device import CameraDevice
from pose_analysis.realtime_pipeline import (
    PipelineDriver, VideoLabel, to_display_bgra)


# ==================== 常量 ====================

POINT_LABEL = {
    JointID.WRIST: "1", JointID.ELBOW: "2", JointID.SHOULDER: "3",
    JointID.HIP: "4", JointID.KNEE: "5", JointID.ANKLE: "6",
    JointID.HEEL: "7", JointID.PHALANX: "8",
}
POINT_CN = {
    1: "腕关节", 2: "肘关节", 3: "肩关节", 4: "髋关节",
    5: "膝关节", 6: "踝关节", 7: "脚跟", 8: "趾骨",
}
# 画面标注用英文缩写 (OpenCV putText 不支持中文)
POINT_EN = {
    1: "1-WRIST", 2: "2-ELBOW", 3: "3-SHOULDER", 4: "4-HIP",
    5: "5-KNEE", 6: "6-ANKLE", 7: "7-HEEL", 8: "8-PHALANX",
}
CALIB_ORDER = [JointID.WRIST, JointID.ELBOW, JointID.SHOULDER, JointID.HIP,
               JointID.KNEE, JointID.ANKLE, JointID.HEEL, JointID.PHALANX]

SEGMENTS = [
    (JointID.WRIST, JointID.ELBOW, "elbow"),
    (JointID.ELBOW, JointID.SHOULDER, "shoulder"),
    (JointID.SHOULDER, JointID.HIP, "hip"),
    (JointID.HIP, JointID.KNEE, "knee"),
    (JointID.KNEE, JointID.ANKLE, "ankle"),
    (JointID.ANKLE, JointID.HEEL, "ankle"),
    (JointID.ANKLE, JointID.PHALANX, "ankle"),
]
ANGLE_LABELS = [
    ("elbow", JointID.ELBOW), ("shoulder", JointID.SHOULDER),
    ("hip", JointID.HIP), ("knee", JointID.KNEE),
    ("ankle", JointID.ANKLE),
]
JOINT_NAMES = ["elbow", "shoulder", "hip", "knee", "ankle"]
JOINT_CN = {"elbow": "肘", "shoulder": "肩", "hip": "髋",
            "knee": "膝", "ankle": "踝"}

GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)

# ─── 全息科技感配色 ───────────────────────────────────
# 主背景: 深空黑, 面板: 深蓝半透, 强调色: 电青/亮蓝
HOLO_BG       = "#070b14"   # 主背景 (深空黑)
HOLO_PANEL    = "#0d1520"   # 面板背景
HOLO_PANEL_2  = "#111b2e"   # 面板次级
HOLO_BORDER   = "#1a3a5c"   # 边框 (深蓝)
HOLO_BORDER_G = "#00d4ff"   # 发光边框 (电青)
HOLO_CYAN     = "#00f0ff"   # 主强调 (电青)
HOLO_BLUE     = "#0099ff"   # 次强调 (亮蓝)
HOLO_GREEN    = "#00ff88"   # 成功 (霓虹绿)
HOLO_ORANGE   = "#ff6b35"   # 警告 (橙红)
HOLO_RED      = "#ff3366"   # 危险 (粉红)
HOLO_TEXT     = "#c8d6e5"   # 主文字 (蓝灰白)
HOLO_DIM      = "#4a5a6a"   # 暗文字
HOLO_TITLE    = "#00d4ff"   # 标题 (电青)

# 骨骼可视化配色 (BGR for OpenCV)
HOLO_SKEL_OK  = (0, 240, 128)    # 正常骨骼 (霓虹绿)
HOLO_SKEL_BAD = (0, 51, 255)    # 异常骨骼 (粉红)
HOLO_NODE     = (255, 212, 0)   # 节点 (电青)
HOLO_GLOW     = (128, 224, 0)   # 光晕 (暗青)
HOLO_GREEN_BGR = (136, 255, 0)  # 霓虹绿 BGR (#00ff88)

# 全局 QSS 样式表 — 全息 HUD
HOLO_QSS = """
QMainWindow { background: #070b14; }

QFrame { background: #0d1520; }

QGroupBox {
    color: #00d4ff;
    font-weight: bold;
    font-size: 11px;
    border: 1px solid #1a3a5c;
    border-left: 2px solid #00d4ff;
    border-radius: 2px;
    margin-top: 10px;
    padding: 6px 4px 4px 6px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #0f1828, stop:1 #0a1018);
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #00f0ff;
}

QLabel { color: #c8d6e5; font-size: 12px; background: transparent; }

QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #142838, stop:1 #0a1828);
    color: #00f0ff;
    border: 1px solid #1a4a6c;
    border-radius: 2px;
    padding: 5px 14px;
    font-size: 11px;
    font-weight: bold;
    font-family: Consolas, "Courier New", monospace;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #1a3a5c, stop:1 #142838);
    border: 1px solid #00d4ff;
    color: #ffffff;
}
QPushButton:pressed {
    background: #00d4ff;
    color: #070b14;
}
QPushButton:disabled {
    background: #0a1018;
    color: #3a4a5a;
    border: 1px solid #1a2a3a;
}
QPushButton:checked {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #00a8cc, stop:1 #007090);
    color: #070b14;
    border: 1px solid #00f0ff;
}

QSpinBox, QDoubleSpinBox {
    background: #0a1320;
    color: #00f0ff;
    border: 1px solid #1a3a5c;
    border-radius: 2px;
    padding: 2px 4px;
    font-family: Consolas, "Courier New", monospace;
    font-size: 11px;
}
QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #00d4ff;
    background: #0d1828;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    background: #1a3a5c; border: none; width: 14px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background: #1a3a5c; border: none; width: 14px;
}

QCheckBox { color: #c8d6e5; font-size: 11px; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #1a3a5c;
    border-radius: 1px;
    background: #0a1320;
}
QCheckBox::indicator:checked {
    background: #00d4ff;
    border: 1px solid #00f0ff;
}

QSlider::groove:horizontal {
    height: 4px;
    background: #0a1320;
    border: 1px solid #1a3a5c;
    border-radius: 1px;
}
QSlider::handle:horizontal {
    background: #00d4ff;
    width: 12px;
    margin: -5px 0;
    border-radius: 6px;
    border: 1px solid #00f0ff;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #0099ff, stop:1 #00f0ff);
    border-radius: 1px;
}

QScrollBar:vertical {
    background: #0a1320; width: 8px; border: none;
}
QScrollBar::handle:vertical {
    background: #1a3a5c; min-height: 20px; border-radius: 4px;
}
QScrollBar::handle:vertical:hover { background: #00d4ff; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QScrollBar:horizontal {
    background: #0a1320; height: 8px; border: none;
}
QScrollBar::handle:horizontal {
    background: #1a3a5c; min-width: 20px; border-radius: 4px;
}
QScrollBar::handle:horizontal:hover { background: #00d4ff; }

QToolTip {
    background: #0d1828;
    color: #00f0ff;
    border: 1px solid #00d4ff;
    padding: 4px 8px;
    font-size: 11px;
    font-family: Consolas, monospace;
}

QStatusBar {
    background: #0a1018;
    color: #4a5a6a;
    border-top: 1px solid #1a3a5c;
    font-size: 10px;
    font-family: Consolas, monospace;
}
"""

DEFAULT_VIDEO = resource_path(
    "pose_analysis", "videos", "demo.mp4")


# ==================== 配置 ====================

@dataclass
class AppConfig:
    """检测/追踪/角度 全部可调参数"""
    # 反光点检测
    min_brightness: int = 50
    min_radius: int = 3
    max_radius: int = 20
    min_distance: int = 20
    min_circularity: float = 0.5
    min_fill_ratio: float = 0.4
    radius_tolerance: float = 0.35
    target_count: int = 8
    # 追踪
    max_jump: float = 80.0
    max_lost_frames: int = 30
    roi_radius: int = 60           # 局部 ROI 搜索半径 (小=快+稳)
    track_max_move: float = 35.0   # 单帧最大匹配距离 (防乱跑)
    track_min_circularity: float = 0.3  # 追踪阶段圆度下限 (比标定宽松)
    lower_gate_scale: float = 3.0  # 下肢匹配半径倍数 (膝/踝/脚跟/趾骨, 覆盖蹬踏幅度)
    use_tophat: bool = True        # top-hat 变换 (反光点贴亮色背景也能分)
    tophat_floor: int = 20         # top-hat 最低响应值
    bone_weight: float = 1.0       # 骨段长度约束惩罚权重
    # 自适应亮度
    auto_brightness: bool = True
    # 角度阈值 [min, max]
    ranges: dict = field(default_factory=lambda: {
        "elbow": (60, 110), "shoulder": (5, 40),
        "hip": (100, 180), "knee": (70, 130), "ankle": (80, 110),
    })
    # 平滑
    smooth_window: int = 5

    def to_marker_config(self, brightness: int = -1) -> MarkerConfig:
        b = self.min_brightness if brightness < 0 else brightness
        return MarkerConfig(
            min_radius=self.min_radius, max_radius=self.max_radius,
            min_distance=self.min_distance, min_circularity=self.min_circularity,
            min_fill_ratio=self.min_fill_ratio, min_brightness=b,
            radius_tolerance=self.radius_tolerance,
            target_count=self.target_count,
        )


# ==================== 核心逻辑 ====================

class AngleSmoother:
    """移动平均平滑"""
    def __init__(self, win):
        self._win = max(2, win)
        self._buf = {n: deque(maxlen=self._win) for n in JOINT_NAMES}

    def reset(self):
        self._buf = {n: deque(maxlen=self._win) for n in JOINT_NAMES}

    def smooth(self, angles):
        out = {}
        for n, a in angles.items():
            if a is None:
                out[n] = None
                continue
            self._buf[n].append(a)
            buf = self._buf[n]
            out[n] = sum(buf) / len(buf)
        return out


# MotionPredictor 已移至 tracking_engine.py (α-β 滤波器版)


def compute_angles(pixels):
    """5 个关节 2D 角度 (度), 缺失返回 None"""
    angles = {}
    if not pixels:
        return {n: None for n in JOINT_NAMES}
    # 像素→伪3D (z=0, 角度只用 X/Y)
    pts = {}
    for jid, (cx, cy) in pixels.items():
        pts[jid] = (float(cx), float(cy))

    def has(*ids):
        return all(j in pts for j in ids)

    if has(JointID.WRIST, JointID.ELBOW, JointID.SHOULDER):
        angles["elbow"] = _angle_2d(
            pts[JointID.WRIST], pts[JointID.ELBOW], pts[JointID.SHOULDER])
    if has(JointID.ELBOW, JointID.SHOULDER, JointID.HIP):
        angles["shoulder"] = _angle_2d(
            pts[JointID.ELBOW], pts[JointID.SHOULDER], pts[JointID.HIP])
    if has(JointID.SHOULDER, JointID.HIP, JointID.KNEE):
        angles["hip"] = _angle_2d(
            pts[JointID.SHOULDER], pts[JointID.HIP], pts[JointID.KNEE])
    if has(JointID.HIP, JointID.KNEE, JointID.ANKLE):
        angles["knee"] = _angle_2d(
            pts[JointID.HIP], pts[JointID.KNEE], pts[JointID.ANKLE])
    if has(JointID.KNEE, JointID.ANKLE, JointID.HEEL, JointID.PHALANX):
        kx, ky = pts[JointID.KNEE]
        ax, ay = pts[JointID.ANKLE]
        hx, hy = pts[JointID.HEEL]
        px, py = pts[JointID.PHALANX]
        v1 = np.array([ax - kx, ay - ky])
        v2 = np.array([px - hx, py - hy])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-6 and n2 > 1e-6:
            cos = float(np.clip(v1.dot(v2) / (n1 * n2), -1, 1))
            angles["ankle"] = float(np.degrees(np.arccos(cos)))
    for n in JOINT_NAMES:
        angles.setdefault(n, None)
    return angles


def _angle_2d(p_a, p_vertex, p_b):
    """三点 2D 角度, p_vertex 为顶点"""
    ax, ay = p_a
    vx, vy = p_vertex
    bx, by = p_b
    v1 = np.array([ax - vx, ay - vy])
    v2 = np.array([bx - vx, by - vy])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos = float(np.clip(v1.dot(v2) / (n1 * n2), -1, 1))
    return float(np.degrees(np.arccos(cos)))


def _in_range(name, ang, ranges):
    if ang is None:
        return True
    lo, hi = ranges[name]
    return lo <= ang <= hi


# ── HUD 底纹缓存 (扫描线+网格, 按分辨率缓存, 避免每帧重绘) ──
_hud_cache = {}


def _get_hud_overlay(h, w):
    """获取缓存的 HUD 底纹 overlay (扫描线 + 网格), 同分辨率只生成一次"""
    key = (h, w)
    if key not in _hud_cache:
        ov = np.zeros((h, w, 3), dtype=np.uint8)
        # 扫描线 (每 6px 一条)
        for y in range(0, h, 6):
            cv2.line(ov, (0, y), (w, y), (0, 30, 40), 1)
        # 网格 (64px 间距)
        for gx in range(0, w, 64):
            cv2.line(ov, (gx, 0), (gx, h), (0, 20, 30), 1)
        for gy in range(0, h, 64):
            cv2.line(ov, (0, gy), (w, gy), (0, 20, 30), 1)
        _hud_cache[key] = ov
    return _hud_cache[key]


def draw_visual(frame, pixels, angles, ranges, missing=False, frame_idx=0):
    """在帧上绘制全息骨骼 + HUD, 返回 BGR 画面 (性能优化版)

    优化: HUD底纹缓存 + 单层光晕统一blend (31次copy→1次)
    """
    import math

    if frame.ndim == 2:
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        vis = frame.copy()

    h, w = vis.shape[:2]
    t = time.time()

    # ═══ HUD 底纹 (缓存, 仅一次 add) ═══
    cv2.add(vis, _get_hud_overlay(h, w), dst=vis)

    # ═══ 四角切角 HUD ═══
    cl, co, ci = 24, 6, 10
    _corners = [(co, co), (w - co, co), (co, h - co), (w - co, h - co)]
    _ch = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
    for (cx, cy), (dx, dy) in zip(_corners, _ch):
        cv2.line(vis, (cx, cy), (cx + cl * dx, cy), HOLO_NODE, 2)
        cv2.line(vis, (cx, cy), (cx, cy + cl * dy), HOLO_NODE, 2)
        ix, iy = cx + cl * dx + 2 * dx, cy + cl * dy + 2 * dy
        cv2.line(vis, (ix, iy), (ix + ci * dx, iy), (0, 80, 100), 1)
        cv2.line(vis, (ix, iy), (ix, iy + ci * dy), (0, 80, 100), 1)

    # ═══ HUD 文字 ═══
    cv2.putText(vis, "KINEX  TRACKING", (12, 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.45, HOLO_NODE, 1)
    cv2.putText(vis, f"FRAME:{frame_idx:05d}", (w - 120, 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 160, 180), 1)
    cv2.putText(vis, f"{w}x{h}", (w - 70, h - 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.35, (0, 100, 120), 1)

    # 底部刻度线
    cv2.line(vis, (co, h - co), (w - co, h - co), (0, 40, 60), 1)
    for mx in range(co, w - co, 80):
        cv2.line(vis, (mx, h - co), (mx, h - co - 4), (0, 60, 80), 1)

    # 追踪状态
    if missing:
        if int(t * 4) % 2 == 0:
            cv2.putText(vis, "!! TRACKING LOST", (w // 2 - 90, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, HOLO_SKEL_BAD, 2)
    else:
        cv2.putText(vis, "● LOCKED", (12, h - 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, HOLO_GREEN_BGR, 1)

    # 中心准星 (无目标时)
    if not pixels:
        cx, cy = w // 2, h // 2
        cv2.drawMarker(vis, (cx, cy), (0, 100, 130), cv2.MARKER_CROSS, 30, 1)
        cv2.circle(vis, (cx, cy), 20, (0, 80, 100), 1)
        cv2.putText(vis, "AWAITING TARGET", (cx - 60, cy + 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 100, 120), 1)
        return vis

    # ═══ 单层光晕 overlay (所有骨骼+节点光晕统一绘制, 仅 blend 一次) ═══
    glow = np.zeros_like(vis)

    for a_id, b_id, jname in SEGMENTS:
        if a_id in pixels and b_id in pixels:
            color = HOLO_SKEL_OK if _in_range(jname, angles.get(jname), ranges) \
                else HOLO_SKEL_BAD
            pa, pb = tuple(pixels[a_id]), tuple(pixels[b_id])
            cv2.line(glow, pa, pb, color, 8)  # 外光晕

    pulse = 0.5 + 0.5 * (0.5 + 0.5 * math.sin(t * 3.0))
    r_pulse = int(16 + 4 * pulse)
    for jid, (x, y) in pixels.items():
        cv2.circle(glow, (int(x), int(y)), r_pulse, HOLO_NODE, -1)

    # 统一 blend 一次 (替代原来 30 次 copy+addWeighted)
    cv2.addWeighted(glow, 0.15, vis, 0.92, 0, vis)

    # ═══ 亮线 + 节点 (直接画在 vis 上, 无需 copy) ═══
    for a_id, b_id, jname in SEGMENTS:
        if a_id in pixels and b_id in pixels:
            color = HOLO_SKEL_OK if _in_range(jname, angles.get(jname), ranges) \
                else HOLO_SKEL_BAD
            pa, pb = tuple(pixels[a_id]), tuple(pixels[b_id])
            cv2.line(vis, pa, pb, color, 2)
            mx, my = (pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2
            cv2.circle(vis, (mx, my), 2, color, -1)

    for jid, (x, y) in pixels.items():
        pt = (int(x), int(y))
        cv2.circle(vis, pt, 5, HOLO_NODE, -1)
        cv2.circle(vis, pt, 8, HOLO_NODE, 1)
        cv2.circle(vis, pt, 3, (0, 10, 20), -1)
        cv2.putText(vis, POINT_LABEL[jid], (pt[0] + 11, pt[1] - 9),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, HOLO_NODE, 1)

    # ═══ 角度数值 (带背景框) ═══
    for jname, jid in ANGLE_LABELS:
        if jid not in pixels:
            continue
        ang = angles.get(jname)
        if ang is None:
            continue
        x, y = int(pixels[jid][0]), int(pixels[jid][1])
        color = HOLO_SKEL_OK if _in_range(jname, ang, ranges) else HOLO_SKEL_BAD
        tx, ty = x + 11, y + 19
        cv2.rectangle(vis, (tx - 2, ty - 10), (tx + 28, ty + 2),
                      (0, 15, 25), -1)
        cv2.rectangle(vis, (tx - 2, ty - 10), (tx + 28, ty + 2), color, 1)
        cv2.putText(vis, f"{ang:.0f}", (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)

    # ═══ 侧边数据条 ═══
    bar_x = w - 30
    bar_y0 = 40
    bar_h = 16
    for i, jname in enumerate(JOINT_NAMES):
        ang = angles.get(jname)
        lo, hi = ranges[jname]
        by = bar_y0 + i * (bar_h + 2)
        cv2.rectangle(vis, (bar_x, by), (bar_x + 22, by + bar_h),
                      (0, 20, 30), -1)
        cv2.rectangle(vis, (bar_x, by), (bar_x + 22, by + bar_h),
                      (0, 60, 80), 1)
        if ang is not None:
            ratio = max(0, min(1, (ang - lo) / max(1, hi - lo)))
            fill_w = int(22 * ratio)
            c = HOLO_GREEN_BGR if lo <= ang <= hi else HOLO_SKEL_BAD
            cv2.rectangle(vis, (bar_x, by), (bar_x + fill_w, by + bar_h), c, -1)
        cv2.putText(vis, JOINT_CN[jname][0], (bar_x + 24, by + 12),
                    cv2.FONT_HERSHEY_DUPLEX, 0.35, (0, 120, 140), 1)

    return vis


def auto_assign_ids(candidates):
    """
    自动标定: 按位置启发式将 8 个候选点分配到 JointID 1~8

    策略 (侧视骑行姿态):
        1. 按 x 中位数分成 左/右 两组
        2. 右组(前侧, x大): 腕(1)/肘(2)/肩(3) — 按 y 排序
        3. 左组(后侧, x小): 髋(4)/膝(5)/踝(6) + 脚跟(7)/趾骨(8)
        4. 脚跟/趾骨: y 最大的两个, x 小=脚跟, x 大=趾骨
        5. 如果分组不均匀则回退到纯 y 排序
    """
    if len(candidates) < 8:
        return None
    # 取最亮的 8 个
    top = sorted(candidates, key=lambda c: -c.brightness)[:8]
    if len(top) < 8:
        return None

    # 按 x 最大间隔分组 (比中位数更鲁棒)
    xs_sorted = sorted(set(c.cx for c in top))
    if len(xs_sorted) >= 2:
        gaps = [(xs_sorted[i + 1] - xs_sorted[i],
                 xs_sorted[i], xs_sorted[i + 1])
                for i in range(len(xs_sorted) - 1)]
        max_gap = max(gaps, key=lambda g: g[0])
        x_boundary = (max_gap[1] + max_gap[2]) / 2.0
    else:
        x_boundary = xs_sorted[0] if xs_sorted else 0
    # 右组按 y 降序: y最大→腕(1), y中间→肘(2), y最小→肩(3)
    right = sorted([c for c in top if c.cx >= x_boundary], key=lambda c: -c.cy)
    left = sorted([c for c in top if c.cx < x_boundary], key=lambda c: c.cy)

    click_map = {}

    # 右组(前侧) 3 个: 腕/肘/肩 (y 从大到小)
    # 左组(后侧) 5 个: 髋/膝/踝 + 脚跟/趾骨
    if len(right) == 3 and len(left) == 5:
        click_map[CALIB_ORDER[0]] = (right[0].cx, right[0].cy)  # 腕
        click_map[CALIB_ORDER[1]] = (right[1].cx, right[1].cy)  # 肘
        click_map[CALIB_ORDER[2]] = (right[2].cx, right[2].cy)  # 肩
        click_map[CALIB_ORDER[3]] = (left[0].cx, left[0].cy)    # 髋
        click_map[CALIB_ORDER[4]] = (left[1].cx, left[1].cy)    # 膝
        click_map[CALIB_ORDER[5]] = (left[2].cx, left[2].cy)    # 踝
        # 最下方两个: x 小=脚跟, x 大=趾骨
        foot = sorted(left[3:5], key=lambda c: c.cx)
        click_map[CALIB_ORDER[6]] = (foot[0].cx, foot[0].cy)    # 脚跟
        click_map[CALIB_ORDER[7]] = (foot[1].cx, foot[1].cy)    # 趾骨
    else:
        # 回退: 纯 y 排序
        top.sort(key=lambda c: c.cy)
        click_map[CALIB_ORDER[0]] = (top[0].cx, top[0].cy)  # 腕
        click_map[CALIB_ORDER[1]] = (top[1].cx, top[1].cy)  # 肘
        click_map[CALIB_ORDER[2]] = (top[2].cx, top[2].cy)  # 肩
        click_map[CALIB_ORDER[3]] = (top[3].cx, top[3].cy)  # 髋
        click_map[CALIB_ORDER[4]] = (top[4].cx, top[4].cy)  # 膝
        click_map[CALIB_ORDER[5]] = (top[5].cx, top[5].cy)  # 踝
        foot = sorted(top[6:8], key=lambda c: c.cx)
        click_map[CALIB_ORDER[6]] = (foot[0].cx, foot[0].cy)  # 脚跟
        click_map[CALIB_ORDER[7]] = (foot[1].cx, foot[1].cy)  # 趾骨

    return click_map if len(click_map) == 8 else None


# ==================== 视频显示 Label ====================

class _LegacyVideoLabel(QLabel):  # 已被 realtime_pipeline.VideoLabel 取代, 可删除
    """视频画面, 支持鼠标点击标定"""
    clicked = pyqtSignal(float, float)  # 画面坐标 (x, y)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setStyleSheet(
            f"background-color:{HOLO_BG}; color:{HOLO_DIM};"
            f"border:1px solid {HOLO_BORDER};"
            f"font-family:Consolas,monospace; font-size:12px;")
        self.setText("◈  等待信号  ◈\n\n请加载视频或连接相机")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_w = 0
        self._frame_h = 0

    def set_frame_size(self, w, h):
        """记录原始帧尺寸, 用于鼠标坐标转换"""
        self._frame_w = w
        self._frame_h = h

    def mousePressEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.MouseButton.LeftButton:
            pm = self.pixmap()
            if pm is None or pm.isNull() or self._frame_w == 0:
                return
            pw, ph = pm.width(), pm.height()
            lw, lh = self.width(), self.height()
            # 缩放后的 pixmap 在 label 中居中
            ox = (lw - pw) / 2
            oy = (lh - ph) / 2
            px = ev.position().x() - ox
            py = ev.position().y() - oy
            if px < 0 or py < 0 or px >= pw or py >= ph:
                return
            # 缩放 pixmap 坐标 → 原始帧坐标
            fx = px * self._frame_w / pw
            fy = py * self._frame_h / ph
            self.clicked.emit(fx, fy)
        super().mousePressEvent(ev)



# ==================== 主窗口 ====================

class PoseAnalysisGUI(QMainWindow):
    """骑行姿态运动捕捉 GUI 主窗口"""

    def __init__(self):
        super().__init__()
        self._state_lock = threading.RLock()   # 保护追踪/指标状态 (处理线程 <-> GUI)
        self._cfg_dirty = False
        self._metrics_ts = 0.0
        self._ui_ts = 0.0
        self._prof = None
        self._vid_idx = 0
        self._cam_idx = 0
        self._cam_reads = 0
        self.setWindowTitle("◈ KINEX  ·  骑行全息追踪系统")
        self.setMinimumSize(1280, 760)
        self.setStyleSheet(HOLO_QSS)

        self._cfg = AppConfig()
        self._detector = MarkerDetector(self._cfg.to_marker_config())
        self._tracker = MarkerTracker(
            max_jump=self._cfg.max_jump, max_lost_frames=self._cfg.max_lost_frames)
        self._smoother = AngleSmoother(self._cfg.smooth_window)
        self._motion = MotionPredictor()  # α-β 滤波器
        self._associator = MarkerAssociator(self._cfg)  # top-hat + 骨长约束匹配
        self._mp_tracker = MediaPipeTracker(self._cfg)  # MediaPipe AI 姿态追踪
        self._metrics = MetricsCollector()  # 实时指标采集
        self._calib_radius = 0.0   # 标定时的参考半径
        self._calib_pos = {}       # 标定位置 {jid_int: (x, y)}
        self._lost = {}            # 连续丢失帧数 {jid_int: int}
        self._last_pixels = {}     # 上一帧像素坐标 (后备)
        self._first_candidates = []  # 首帧候选圆点 (辅助标定)
        self._track_ts = 0.0       # 上次追踪时间戳 (计算 dt 用)
        self._nominal_interval = 1.0 / 30.0  # 标称帧间隔 (秒)

        self._cap = None
        self._video_path = None
        self._is_camera = False       # 相机模式 (实时预览)
        self._camera_device = None    # CameraDevice 实例 (相机模式专用)
        self._frame_idx = 0
        self._total_frames = 0
        self._playing = False
        self._calibrated = False
        self._manual_mode = False
        self._calib_clicks = []
        self._first_frame = None

        self._init_ui()
        self._init_timer()

        # 设备重试定时器 (相机断线自动重连, 指数退避)
        self._device_retry_timer = QTimer()
        self._device_retry_timer.timeout.connect(self._on_device_retry)
        self._device_retry_count = 0
        self._device_retry_max = 5

        # 自动加载默认视频
        if os.path.isfile(DEFAULT_VIDEO):
            self._load_video(DEFAULT_VIDEO)

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        # 左侧参数面板
        left = self._build_param_panel()
        layout.addWidget(left)

        # 中间视频 + 状态
        right = self._build_video_panel()
        layout.addWidget(right, 1)

        # 右侧实时数据面板
        self._metrics_panel = self._build_metrics_panel()
        layout.addWidget(self._metrics_panel)

    # ─── 左侧参数面板 ──────────────────────────────────

    def _build_param_panel(self):
        """构建左侧参数面板 — 全息 HUD 风格"""
        panel = QFrame()
        panel.setFixedWidth(330)
        panel.setStyleSheet(
            f"QFrame{{background:{HOLO_PANEL};"
            f"border:1px solid {HOLO_BORDER};}}")
        v = QVBoxLayout(panel)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)

        # ─ 系统标题 + 副标题 ─
        title = QLabel("◈ KINEX  控制台")
        title.setStyleSheet(
            f"color:{HOLO_CYAN}; font-size:15px; font-weight:bold;"
            f"font-family:Consolas,'Courier New',monospace;"
            f"background:transparent; border:none;")
        v.addWidget(title)
        subtitle = QLabel("  全息运动分析系统")
        subtitle.setStyleSheet(
            f"color:{HOLO_DIM}; font-size:9px;"
            f"font-family:Consolas,monospace; background:transparent;")
        v.addWidget(subtitle)
        # 标题下发光线
        sep = QFrame()
        sep.setFixedHeight(2)
        sep.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {HOLO_CYAN}, stop:0.7 {HOLO_BORDER}, stop:1 transparent);")
        v.addWidget(sep)

        # ─ 视频源 ─
        grp_src = QGroupBox("◈ 数据源")
        gs = QHBoxLayout(grp_src)
        self.btn_open = QPushButton("▶ 加载视频")
        self.btn_open.clicked.connect(self._on_open_video)
        gs.addWidget(self.btn_open)
        self.btn_camera = QPushButton("◈ 连接相机")
        self.btn_camera.clicked.connect(self._on_open_camera)
        gs.addWidget(self.btn_camera)
        v.addWidget(grp_src)

        # ─ 播放控制 ─
        grp_play = QGroupBox("◈ 播放")
        gp = QHBoxLayout(grp_play)
        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.clicked.connect(self._on_play_pause)
        self.btn_play.setEnabled(False)
        gp.addWidget(self.btn_play)
        self.btn_reset = QPushButton("↻ 重置")
        self.btn_reset.clicked.connect(self._on_reset)
        gp.addWidget(self.btn_reset)
        v.addWidget(grp_play)

        # ─ 标定 ─
        grp_calib = QGroupBox("◈ 标定")
        gc = QVBoxLayout(grp_calib)
        self.btn_auto = QPushButton("⚡ 自动标定")
        self.btn_auto.clicked.connect(self._on_auto_calib)
        self.btn_auto.setEnabled(False)
        gc.addWidget(self.btn_auto)
        self.btn_manual = QPushButton("✋ 手动标定")
        self.btn_manual.setCheckable(True)
        self.btn_manual.clicked.connect(self._on_manual_mode)
        self.btn_manual.setEnabled(False)
        gc.addWidget(self.btn_manual)
        self.lbl_calib_hint = QLabel("—")
        self.lbl_calib_hint.setStyleSheet(
            f"color:{HOLO_ORANGE}; font-size:11px; padding:2px;")
        gc.addWidget(self.lbl_calib_hint)
        v.addWidget(grp_calib)

        # ─ 检测参数 ─
        grp_det = QGroupBox("◈ 检测参数")
        gd = QFormLayout(grp_det)
        self.sp_brightness = QSpinBox()
        self.sp_brightness.setRange(0, 255)
        self.sp_brightness.setValue(self._cfg.min_brightness)
        self.sp_brightness.valueChanged.connect(self._on_cfg_changed)
        gd.addRow("亮度阈值:", self.sp_brightness)

        self.sl_brightness = QSlider(Qt.Orientation.Horizontal)
        self.sl_brightness.setRange(0, 255)
        self.sl_brightness.setValue(self._cfg.min_brightness)
        self.sl_brightness.valueChanged.connect(self.sp_brightness.setValue)
        self.sp_brightness.valueChanged.connect(self.sl_brightness.setValue)
        gd.addRow("", self.sl_brightness)

        self.sp_min_r = QSpinBox()
        self.sp_min_r.setRange(1, 50)
        self.sp_min_r.setValue(self._cfg.min_radius)
        self.sp_min_r.valueChanged.connect(self._on_cfg_changed)
        gd.addRow("最小半径:", self.sp_min_r)

        self.sp_max_r = QSpinBox()
        self.sp_max_r.setRange(2, 100)
        self.sp_max_r.setValue(self._cfg.max_radius)
        self.sp_max_r.valueChanged.connect(self._on_cfg_changed)
        gd.addRow("最大半径:", self.sp_max_r)

        self.sp_min_dist = QSpinBox()
        self.sp_min_dist.setRange(5, 200)
        self.sp_min_dist.setValue(self._cfg.min_distance)
        self.sp_min_dist.valueChanged.connect(self._on_cfg_changed)
        gd.addRow("最小距离:", self.sp_min_dist)

        self.sp_circ = QDoubleSpinBox()
        self.sp_circ.setRange(0.0, 1.0)
        self.sp_circ.setSingleStep(0.05)
        self.sp_circ.setValue(self._cfg.min_circularity)
        self.sp_circ.valueChanged.connect(self._on_cfg_changed)
        gd.addRow("最小圆度:", self.sp_circ)

        self.cb_auto = QCheckBox("自适应亮度 (候选不足自动降阈值)")
        self.cb_auto.setChecked(self._cfg.auto_brightness)
        self.cb_auto.stateChanged.connect(self._on_cfg_changed)
        gd.addRow("", self.cb_auto)
        v.addWidget(grp_det)

        # ─ 角度阈值 ─
        grp_ang = QGroupBox("◈ 角度阈值 [最小, 最大]")
        ga = QGridLayout(grp_ang)
        ga.setSpacing(4)
        self._ang_spins = {}
        for i, name in enumerate(JOINT_NAMES):
            lo, hi = self._cfg.ranges[name]
            ga.addWidget(QLabel(JOINT_CN[name]), i, 0)
            sp_lo = QSpinBox()
            sp_lo.setRange(0, 180)
            sp_lo.setValue(lo)
            sp_hi = QSpinBox()
            sp_hi.setRange(0, 180)
            sp_hi.setValue(hi)
            sp_lo.valueChanged.connect(self._on_cfg_changed)
            sp_hi.valueChanged.connect(self._on_cfg_changed)
            ga.addWidget(sp_lo, i, 1)
            ga.addWidget(sp_hi, i, 2)
            self._ang_spins[name] = (sp_lo, sp_hi)
        v.addWidget(grp_ang)

        # ─ 平滑 ─
        grp_smooth = QGroupBox("◈ 平滑")
        gsm = QFormLayout(grp_smooth)
        self.sp_smooth = QSpinBox()
        self.sp_smooth.setRange(1, 30)
        self.sp_smooth.setValue(self._cfg.smooth_window)
        self.sp_smooth.valueChanged.connect(self._on_cfg_changed)
        gsm.addRow("窗口大小:", self.sp_smooth)

        self.sp_jump = QSpinBox()
        self.sp_jump.setRange(10, 300)
        self.sp_jump.setValue(int(self._cfg.max_jump))
        self.sp_jump.valueChanged.connect(self._on_cfg_changed)
        gsm.addRow("最大位移:", self.sp_jump)

        self.sp_lost = QSpinBox()
        self.sp_lost.setRange(1, 100)
        self.sp_lost.setValue(self._cfg.max_lost_frames)
        self.sp_lost.valueChanged.connect(self._on_cfg_changed)
        gsm.addRow("丢失容差:", self.sp_lost)

        self.sp_roi = QSpinBox()
        self.sp_roi.setRange(20, 200)
        self.sp_roi.setValue(self._cfg.roi_radius)
        self.sp_roi.valueChanged.connect(self._on_cfg_changed)
        gsm.addRow("ROI半径:", self.sp_roi)

        self.sp_max_move = QSpinBox()
        self.sp_max_move.setRange(10, 100)
        self.sp_max_move.setValue(int(self._cfg.track_max_move))
        self.sp_max_move.valueChanged.connect(self._on_cfg_changed)
        gsm.addRow("匹配距离:", self.sp_max_move)
        v.addWidget(grp_smooth)

        v.addStretch()
        return panel

    # ─── 右侧视频面板 ──────────────────────────────────

    def _build_video_panel(self):
        """构建中间视频面板 — 全息 HUD 框架"""
        panel = QFrame()
        panel.setStyleSheet(
            f"QFrame{{background:{HOLO_PANEL};"
            f"border:1px solid {HOLO_BORDER};}}")
        v = QVBoxLayout(panel)
        v.setContentsMargins(2, 2, 2, 2)
        v.setSpacing(0)

        # ─ HUD 标题条 (带左右装饰) ─
        hud_bar = QFrame()
        hud_bar.setFixedHeight(30)
        hud_bar.setStyleSheet(
            f"QFrame{{background:{HOLO_PANEL_2};"
            f"border-bottom:1px solid {HOLO_BORDER_G};}}")
        hb = QHBoxLayout(hud_bar)
        hb.setContentsMargins(8, 0, 8, 0)
        hb.setSpacing(6)
        lbl_hud = QLabel("◈ 实时追踪")
        lbl_hud.setStyleSheet(
            f"color:{HOLO_CYAN};font-size:13px;font-weight:bold;"
            f"font-family:Consolas,monospace;background:transparent;")
        hb.addWidget(lbl_hud)
        hb.addStretch()
        # 模式标签
        self.lbl_hud_mode = QLabel("模式:--")
        self.lbl_hud_mode.setStyleSheet(
            f"color:{HOLO_DIM};font-size:10px;"
            f"font-family:Consolas,monospace;background:transparent;")
        hb.addWidget(self.lbl_hud_mode)
        # 帧率标签
        self.lbl_hud_fps = QLabel("帧率:--")
        self.lbl_hud_fps.setStyleSheet(
            f"color:{HOLO_ORANGE};font-size:10px;"
            f"font-family:Consolas,monospace;background:transparent;")
        hb.addWidget(self.lbl_hud_fps)
        # 状态灯
        self.lbl_hud_status = QLabel("● 待机")
        self.lbl_hud_status.setStyleSheet(
            f"color:{HOLO_DIM};font-size:11px;"
            f"font-family:Consolas,monospace;background:transparent;")
        hb.addWidget(self.lbl_hud_status)
        v.addWidget(hud_bar)

        # 视频画面
        self.video_label = VideoLabel()
        self.video_label.clicked.connect(self._on_video_click)
        v.addWidget(self.video_label, 1)

        # ─ 底部状态条 (带 HUD 装饰) ─
        status_frame = QFrame()
        status_frame.setFixedHeight(24)
        status_frame.setStyleSheet(
            f"QFrame{{background:{HOLO_PANEL_2};"
            f"border-top:1px solid {HOLO_BORDER};}}")
        sf = QHBoxLayout(status_frame)
        sf.setContentsMargins(8, 0, 8, 0)
        sf.setSpacing(8)
        self.lbl_status = QLabel("就绪")
        self.lbl_status.setStyleSheet(
            f"color:{HOLO_TEXT};font-size:10px;"
            f"font-family:Consolas,monospace;background:transparent;")
        sf.addWidget(self.lbl_status)
        sf.addStretch()
        # 点数标签
        self.lbl_hud_points = QLabel("点数:0/8")
        self.lbl_hud_points.setStyleSheet(
            f"color:{HOLO_DIM};font-size:10px;"
            f"font-family:Consolas,monospace;background:transparent;")
        sf.addWidget(self.lbl_hud_points)
        v.addWidget(status_frame)
        return panel

    # ─── 实时数据面板 ──────────────────────────────────

    # 指标显示配置: (metric_name, 显示列)
    # 列: 'cur' 'max' 'min' 'avg' (1=显示, 0=不显示)
    _METRIC_LAYOUT = [
        ('revolutions',   1, 0, 0, 0),
        ('knee_angle',    1, 1, 1, 1),
        ('ankle_angle',   1, 1, 1, 1),
        ('shoulder_angle',1, 1, 1, 1),
        ('elbow_angle',   1, 1, 1, 1),
        ('back_angle',    1, 1, 1, 1),
        ('foot_knee_offset',1,1, 1, 1),
        ('hip_h_disp',    1, 1, 1, 1),
        ('hip_v_disp',    1, 1, 1, 1),
        ('knee_lateral',  1, 0, 0, 0),
        ('shoulder_lateral',1,0,0,0),
        ('hip_lateral',   1, 0, 0, 0),
        ('ankle_range',   1, 0, 0, 0),
        ('hip_closed',    1, 0, 1, 0),
    ]

    def _build_metrics_panel(self):
        """构建右侧实时数据面板 — 全息 HUD 风格"""
        panel = QFrame()
        panel.setFixedWidth(300)
        panel.setStyleSheet(
            f"QFrame {{ background:{HOLO_PANEL};"
            f"border:1px solid {HOLO_BORDER}; }}")
        v = QVBoxLayout(panel)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(1)

        # ─ 标题区 (带脉冲 LIVE 灯) ─
        hdr = QHBoxLayout()
        hdr.setSpacing(6)
        title = QLabel("◈ 数据指标")
        title.setStyleSheet(
            f"color:{HOLO_CYAN}; font-size:15px; font-weight:bold;"
            f"font-family:Consolas,monospace; background:transparent; border:none;")
        hdr.addWidget(title)
        hdr.addStretch()
        self._lbl_live = QLabel("● 实时")
        self._lbl_live.setStyleSheet(
            f"color:{HOLO_GREEN}; font-size:10px; font-weight:bold;"
            f"font-family:Consolas,monospace; background:transparent;")
        hdr.addWidget(self._lbl_live)
        v.addLayout(hdr)
        # 标题下发光线
        sep = QFrame()
        sep.setFixedHeight(2)
        sep.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {HOLO_CYAN}, stop:0.7 {HOLO_BORDER}, stop:1 transparent);")
        v.addWidget(sep)

        # ─ 表头 ─
        hdr2 = QHBoxLayout()
        hdr2.setSpacing(2)
        col_defs = [("指标", 76), ("实时", 42), ("最大", 42),
                    ("最小", 42), ("平均", 42)]
        for text, wd in col_defs:
            lbl = QLabel(text)
            lbl.setFixedWidth(wd)
            lbl.setStyleSheet(
                f"color:{HOLO_DIM}; font-size:9px;"
                f"font-family:Consolas,monospace;"
                f"background:transparent; padding:2px 0;")
            hdr2.addWidget(lbl)
        v.addLayout(hdr2)

        # ─ 每行指标 (交替背景 + 行底发光线) ─
        self._metric_labels = {}
        for row_i, (name, show_cur, show_max, show_min, show_avg) in enumerate(
                self._METRIC_LAYOUT):
            row = QHBoxLayout()
            row.setSpacing(2)

            # 行背景 (交替深浅)
            bg = HOLO_PANEL_2 if row_i % 2 == 0 else HOLO_PANEL
            row_frame = QFrame()
            row_frame.setStyleSheet(f"background:{bg};")

            lbl_n = QLabel(METRIC_CN[name])
            lbl_n.setFixedWidth(76)
            lbl_n.setStyleSheet(
                f"color:{HOLO_TEXT}; font-size:11px;"
                f"font-family:Consolas,monospace;"
                f"background:transparent; padding:1px 2px;")
            row.addWidget(lbl_n)

            labels = {'cur': None, 'max': None, 'min': None, 'avg': None}
            for key, show in [('cur', show_cur), ('max', show_max),
                              ('min', show_min), ('avg', show_avg)]:
                if show:
                    lbl = QLabel("—")
                    lbl.setFixedWidth(42)
                    if key == 'cur':
                        color = HOLO_CYAN
                    elif key == 'max':
                        color = HOLO_ORANGE
                    elif key == 'min':
                        color = HOLO_BLUE
                    else:
                        color = HOLO_GREEN
                    lbl.setStyleSheet(
                        f"color:{color}; font-size:11px;"
                        f"font-family:Consolas,monospace;"
                        f"background:transparent; padding:1px 2px;"
                        f"font-weight:bold;")
                    row.addWidget(lbl)
                    labels[key] = lbl
                else:
                    spacer = QLabel("")
                    spacer.setFixedWidth(42)
                    spacer.setStyleSheet("background:transparent;")
                    row.addWidget(spacer)

            v.addLayout(row)
            self._metric_labels[name] = labels

        v.addStretch()

        # ─ 底部状态条 ─
        sep2 = QFrame()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 transparent, stop:0.3 {HOLO_BORDER}, stop:0.7 {HOLO_BORDER},"
            f"stop:1 transparent);")
        v.addWidget(sep2)
        self._lbl_sys_status = QLabel("系统:就绪")
        self._lbl_sys_status.setStyleSheet(
            f"color:{HOLO_DIM}; font-size:9px;"
            f"font-family:Consolas,monospace; background:transparent;"
            f"padding:4px 0;")
        v.addWidget(self._lbl_sys_status)

        # 重置按钮
        self.btn_reset_metrics = QPushButton("↻ 重置数据")
        self.btn_reset_metrics.clicked.connect(self._on_reset_metrics)
        v.addWidget(self.btn_reset_metrics)

        # ─ LIVE 脉冲定时器 ─
        self._live_blink = False
        self._live_timer = QTimer()
        self._live_timer.setInterval(600)
        self._live_timer.timeout.connect(self._on_live_blink)
        self._live_timer.start()

        return panel

    def _on_live_blink(self):
        """LIVE 指示灯脉冲"""
        self._live_blink = not self._live_blink
        c = HOLO_GREEN if self._live_blink else HOLO_DIM
        self._lbl_live.setStyleSheet(
            f"color:{c}; font-size:10px; font-weight:bold;"
            f"font-family:Consolas,monospace; background:transparent;")

    def _update_metrics_display(self, display_data):
        """更新数据面板的数值"""
        for name, labels in self._metric_labels.items():
            entry = display_data.get(name, {})
            for key in ('cur', 'max', 'min', 'avg'):
                lbl = labels[key]
                if lbl is None:
                    continue
                val = entry.get(key)
                if val is None:
                    lbl.setText("—")
                elif name == 'revolutions':
                    lbl.setText(f"{int(val)}")
                else:
                    lbl.setText(f"{val:.1f}")

    def _on_reset_metrics(self):
        """重置指标统计 (流水线运行中与处理线程互斥)"""
        with self._state_lock:
            self._metrics.reset(self._calib_pos)
            disp = self._metrics.get_display()
        self._update_metrics_display(disp)

    # ─── 定时器 ───────────────────────────────────────

    def _init_timer(self):
        # 流水线驱动器: 对外兼容 QTimer 的 start()/stop(), 其余代码无需改动
        self._timer = PipelineDriver(
            self._make_source, self._process_core, parent=self)
        self._timer.result_ready.connect(self._on_pipeline_result)
        self._timer.failed.connect(self._on_pipeline_failed)
        self._timer.display_box = (self.video_label.width(),
                                   self.video_label.height())

    def _calc_fps(self):
        """计算最近 FPS (基于最近 30 帧时间戳)"""
        now = time.time()
        self._fps_ts.append(now)
        # 保留最近 30 帧
        if len(self._fps_ts) > 30:
            self._fps_ts = self._fps_ts[-30:]
        if len(self._fps_ts) < 2:
            return 0.0
        dt = self._fps_ts[-1] - self._fps_ts[0]
        if dt <= 0:
            return 0.0
        return (len(self._fps_ts) - 1) / dt

    # ─── 参数变更 ─────────────────────────────────────

    def _on_cfg_changed(self, *_):
        """GUI 线程只把控件值写进 cfg (廉价); 重建检测器/平滑器延后到处理线程帧间"""
        c = self._cfg
        c.min_brightness = self.sp_brightness.value()
        c.min_radius = self.sp_min_r.value()
        c.max_radius = max(self.sp_min_r.value() + 1, self.sp_max_r.value())
        c.min_distance = self.sp_min_dist.value()
        c.min_circularity = self.sp_circ.value()
        c.auto_brightness = self.cb_auto.isChecked()
        for name, (sp_lo, sp_hi) in self._ang_spins.items():
            lo = min(sp_lo.value(), sp_hi.value())
            hi = max(sp_lo.value(), sp_hi.value())
            c.ranges[name] = (lo, hi)
        c.smooth_window = self.sp_smooth.value()
        c.max_jump = float(self.sp_jump.value())
        c.max_lost_frames = self.sp_lost.value()
        c.roi_radius = self.sp_roi.value()
        c.track_max_move = float(self.sp_max_move.value())
        self._cfg_dirty = True
        if not self._timer.isActive():          # 流水线没在跑: 直接应用
            with self._state_lock:
                self._apply_cfg_changes()

    # ─── 视频加载 ─────────────────────────────────────

    def _on_open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "Video (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self._load_video(path)

    def _load_video(self, path):
        self._close_video()
        self._is_camera = False
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.lbl_status.setText(f"无法打开: {path}")
            return
        self._cap = cap
        self._video_path = path
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_idx = 0
        self._calibrated = False
        self._manual_mode = False
        self._calib_clicks = []
        self._tracker.reset()
        self._associator.reset()
        self._motion.reset()
        self._mp_tracker.reset()
        self._metrics.reset()
        self._calib_radius = 0.0
        self._calib_pos = {}
        self._lost = {}
        self._last_pixels = {}
        self._first_candidates = []
        # 读取首帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret:
            frame = self._resize_frame(frame)
            self._first_frame = frame
            # 检测首帧候选圆点 (辅助标定定位)
            gray = self._to_gray(frame)
            self._first_candidates = self._detect_adaptive(gray)
            self._show_frame(frame)
        self.btn_play.setEnabled(True)
        self.btn_auto.setEnabled(True)
        self.btn_manual.setEnabled(True)
        self.btn_play.setText("▶ 播放")
        self._playing = False
        self.lbl_calib_hint.setText(
            "请点击「手动标定」按顺序点击8个点\n"
            "  1腕 2肘 3肩 4髋 5膝 6踝 7脚跟 8趾骨")
        self._update_status()

    # ─── 相机连接 (复用 capture.CameraDevice) ──────────

    def _on_open_camera(self):
        """连接相机: 优先 Kinect v2, 后备 USB 相机 (多相机时可选择)"""
        self.lbl_status.setText("正在连接 Kinect v2...")
        QApplication.processEvents()
        self._close_video()

        # 1. 优先尝试 Kinect v2 (pykinect2 + Kinect SDK)
        try:
            from capture.kinect_device import KinectDevice
            device = KinectDevice()
            if device.open():
                # open() 即使 IsAvailable=False 也返回 True, 需复查
                if device.is_available():
                    self._load_camera(device)
                    return
                # Kinect 打开但不可用 (USB/供电问题), 关闭后备
                msg = device.get_status_message()
                device.close()
                self.lbl_status.setText(
                    "Kinect 传感器不可用\n" + msg)
                QApplication.processEvents()
        except ImportError:
            pass  # pykinect2 未安装, 继续尝试 USB

        # 2. 后备: USB 相机 — 枚举所有可用相机
        self.lbl_status.setText("Kinect 不可用, 正在枚举 USB 相机...")
        QApplication.processEvents()
        cameras = CameraDevice.enumerate_cameras()
        if not cameras:
            self.lbl_status.setText(
                "未找到可用相机 (Kinect 和 USB 均未检测到)")
            return

        if len(cameras) == 1:
            # 只有一个相机, 直接连接
            idx = cameras[0]["index"]
        else:
            # 多个相机: 弹出选择对话框
            idx = self._select_camera_dialog(cameras)
            if idx is None:
                self.lbl_status.setText("已取消相机选择")
                return

        self.lbl_status.setText(f"正在连接相机 {idx}...")
        QApplication.processEvents()
        device = CameraDevice(device_index=idx)
        if device.open():
            self._load_camera(device)
        else:
            self.lbl_status.setText(
                f"相机 {idx} 连接失败, 请重试或选择其他相机")

    def _select_camera_dialog(self, cameras):
        """弹出相机选择对话框, 返回选中的索引 (或 None 表示取消)

        Args:
            cameras: [{"index":0, "width":1920, "height":1080, "fps":30}, ...]
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("选择相机")
        dlg.setFixedSize(360, 280)
        dlg.setStyleSheet(self.styleSheet())

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        hint = QLabel("检测到多个相机, 请选择:")
        hint.setStyleSheet(
            f"color:{HOLO_CYAN}; font-size:13px; font-weight:bold;"
            f"font-family:Consolas,monospace; background:transparent;")
        layout.addWidget(hint)

        list_widget = QListWidget()
        list_widget.setStyleSheet(
            f"QListWidget{{background:{HOLO_PANEL}; border:1px solid {HOLO_BORDER};"
            f"color:{HOLO_TEXT}; font-family:Consolas,monospace; font-size:12px;}}"
            f"QListWidget::item:selected{{background:{HOLO_CYAN}; color:#000;}}"
            f"QListWidget::item:hover{{background:{HOLO_BORDER};}}")
        for cam in cameras:
            text = (f"相机 {cam['index']}  "
                    f"{cam['width']}x{cam['height']}  "
                    f"{cam['fps']}fps")
            list_widget.addItem(text)
        list_widget.setCurrentRow(0)
        layout.addWidget(list_widget)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btns.setStyleSheet(self.styleSheet())
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            row = list_widget.currentRow()
            if 0 <= row < len(cameras):
                return cameras[row]["index"]
        return None

    def _load_camera(self, device):
        """连接相机成功, 开始实时预览"""
        self._camera_device = device
        self._is_camera = True
        self._video_path = device.get_device_id()
        self._total_frames = 0   # 相机无固定帧数
        self._frame_idx = 0
        self._calibrated = False
        self._manual_mode = False
        self._calib_clicks = []
        self._tracker.reset()
        self._associator.reset()
        self._motion.reset()
        self._mp_tracker.reset()
        self._metrics.reset()
        self._calib_radius = 0.0
        self._calib_pos = {}
        self._lost = {}
        self._last_pixels = {}
        self._first_candidates = []
        self._device_retry_count = 0

        # 读取首帧 (优先红外, 反光点可见)
        if device.update():
            frame, gray = self._get_camera_frames()
            if frame is not None:
                self._first_frame = frame
                self._first_candidates = self._detect_adaptive(gray)
                self._show_frame(frame)

        # 相机模式: 立即开始实时预览
        self.btn_play.setEnabled(True)
        self.btn_auto.setEnabled(True)
        self.btn_manual.setEnabled(True)
        self.btn_play.setText("⏸ 暂停")
        self._playing = True
        self._timer.start()

        did = device.get_device_id()
        self.lbl_calib_hint.setText(
            f"{did} 已连接, 实时预览中\n"
            "请点击「自动标定」或「手动标定」")
        self.lbl_status.setText(f"{did} 就绪")

    def _check_camera_status(self):
        """检查相机可用性, 不可用时启动重试"""
        if self._camera_device is None:
            return
        if self._camera_device.is_available():
            return
        # 不可用: 提示 + 排定重试
        msg = self._camera_device.get_status_message()
        if not msg:
            msg = "相机无信号, 请检查连接后重试。"
        self.lbl_status.setText(msg.split("\n")[0])
        self._schedule_device_retry()

    def _schedule_device_retry(self):
        """排定一次自动重试 (指数退避)"""
        if self._device_retry_count >= self._device_retry_max:
            self.lbl_status.setText("相机多次重试失败, 请重新点击「连接相机」")
            return
        delay = min(8000, 1500 * (self._device_retry_count + 1))
        self._device_retry_count += 1
        self.lbl_status.setText(
            f"相机重试 {self._device_retry_count}/{self._device_retry_max} "
            f"({delay // 1000}s 后)…")
        self._device_retry_timer.start(delay)

    def _on_device_retry(self):
        """自动重试定时器回调"""
        self._device_retry_timer.stop()
        if self._camera_device is None:
            return
        try:
            self._camera_device.retry_open()
        except Exception:
            pass
        if self._camera_device.is_available():
            self._device_retry_count = 0
            self._playing = True
            self.btn_play.setText("⏸ 暂停")
            self._timer.start()
            self.lbl_status.setText(
                f"{self._camera_device.get_device_id()} 已恢复")
        else:
            self._check_camera_status()

    def _close_video(self):
        self._timer.stop()      # 先停流水线 (等线程退出), 再释放设备
        if self._camera_device is not None:
            self._camera_device.close()
            self._camera_device = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ─── 播放控制 ─────────────────────────────────────

    def _on_play_pause(self):
        if self._cap is None and self._camera_device is None:
            return
        if not self._calibrated and not self._is_camera:
            self.lbl_status.setText("请先标定!")
            return
        self._playing = not self._playing
        self.btn_play.setText("⏸ 暂停" if self._playing else "▶ 播放")
        if self._playing:
            self._timer.start()
        else:
            self._timer.stop()

    def _on_reset(self):
        self._pause_pipeline()
        self._calibrated = False
        self._manual_mode = False
        self._calib_clicks = []
        self._tracker.reset()
        self._associator.reset()
        self._motion.reset()
        self._mp_tracker.reset()
        self._metrics.reset()
        self._calib_radius = 0.0
        self._calib_pos = {}
        self._lost = {}
        self._last_pixels = {}
        self.btn_manual.setChecked(False)
        self._playing = False
        self._timer.stop()
        self.btn_play.setText("▶ 播放")
        if self._is_camera:
            # 相机模式: 读取当前帧作为预览 (用红外帧)
            if self._camera_device is not None:
                if self._camera_device.update():
                    frame, gray = self._get_camera_frames()
                    if frame is not None:
                        self._first_frame = frame
                        self._show_frame(frame)
            self._frame_idx = 0
        elif self._cap is not None and self._first_frame is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._frame_idx = 0
            self._show_frame(self._first_frame)
        self.lbl_calib_hint.setText(
            "请点击「手动标定」按顺序点击8个点\n"
            "  1腕 2肘 3肩 4髋 5膝 6踝 7脚跟 8趾骨")
        self._update_status()

    # ─── 标定 ─────────────────────────────────────────

    def _on_auto_calib(self):
        self._pause_pipeline()
        if self._is_camera:
            # 相机模式: 暂停后从当前帧标定 (用红外帧)
            self._playing = False
            self._timer.stop()
            self.btn_play.setText("▶ 播放")
            if self._camera_device is None or not self._camera_device.update():
                self.lbl_status.setText("相机无信号")
                return
            frame, gray = self._get_camera_frames()
            if frame is None:
                self.lbl_status.setText("相机无信号")
                return
            self._first_frame = frame
            # 相机模式: gray 已经在 _get_camera_frames 中获取, 直接用于标定
            candidates = self._detect_adaptive(gray, force_global=True)
            click_map = auto_assign_ids(candidates)
            if click_map is None:
                self.lbl_status.setText(
                    f"自动标定失败: 仅检测到 {len(candidates)} 个候选, 需 8 个")
                return
            ok = self._tracker.calibrate(candidates, click_map)
            if ok:
                self._calibrated = True
                self._calib_radius = float(np.median(
                    [c.radius for c in candidates]))
                self._save_calib_pos(click_map)
                self._init_tracker(click_map)
                self.lbl_calib_hint.setText(
                    "自动标定完成, 请播放后检查编号是否正确\n"
                    "如编号不对请「重置标定」后用「手动标定」")
                self.lbl_status.setText("标定完成, 点击「播放」开始")
            else:
                self.lbl_status.setText("自动标定: 匹配不足, 请尝试手动标定")
            return
        if self._first_frame is None:
            return
        gray = self._to_gray(self._first_frame)
        candidates = self._detect_adaptive(gray, force_global=True)
        click_map = auto_assign_ids(candidates)
        if click_map is None:
            self.lbl_status.setText(
                f"自动标定失败: 仅检测到 {len(candidates)} 个候选, 需 8 个")
            return
        ok = self._tracker.calibrate(candidates, click_map)
        if ok:
            self._calibrated = True
            self._calib_radius = float(np.median(
                [c.radius for c in candidates]))
            self._save_calib_pos(click_map)
            self._init_tracker(click_map)
            self.lbl_calib_hint.setText(
                "自动标定完成, 请播放后检查编号是否正确\n"
                "如编号不对请「重置标定」后用「手动标定」")
            self.lbl_status.setText("标定完成, 点击「播放」开始")
        else:
            self.lbl_status.setText("自动标定: 匹配不足, 请尝试手动标定")

    def _on_manual_mode(self):
        if self._cap is None and self._camera_device is None:
            self.btn_manual.setChecked(False)
            return
        self._manual_mode = self.btn_manual.isChecked()
        self._calib_clicks = []
        if self._manual_mode:
            self._playing = False
            self._timer.stop()
            self.btn_play.setText("▶ 播放")
            # 相机模式: 捕获当前帧用于标定 (用红外帧)
            if self._is_camera and self._camera_device is not None:
                if self._camera_device.update():
                    frame, gray = self._get_camera_frames()
                    if frame is not None:
                        self._first_frame = frame
            self.lbl_calib_hint.setText(
                f"请点击第1个点: {POINT_CN[1]} (画面绿色圈为检测到的候选)")
            # 刷新画面显示候选圆圈
            if self._first_frame is not None:
                self._show_frame(self._first_frame)
        else:
            self.lbl_calib_hint.setText("手动标定已取消")
            if self._first_frame is not None:
                self._show_frame(self._first_frame)

    def _on_video_click(self, x, y):
        if not self._manual_mode:
            return
        if len(self._calib_clicks) >= 8:
            return
        self._calib_clicks.append((int(x), int(y)))
        n = len(self._calib_clicks)
        if n < 8:
            self.lbl_calib_hint.setText(
                f"已标定 {n}/8, 请点击第{n+1}个点: {POINT_CN[n+1]}")
            # 刷新画面显示已点击的点
            if self._first_frame is not None:
                self._show_frame(self._first_frame)
        else:
            self.lbl_calib_hint.setText("正在标定...")
            if self._first_frame is not None:
                self._show_frame(self._first_frame)
            self._do_manual_calib()

    def _do_manual_calib(self):
        self._pause_pipeline()
        if self._first_frame is None or len(self._calib_clicks) < 8:
            return
        gray = self._to_gray(self._first_frame)
        candidates = self._detect_adaptive(gray, force_global=True)
        click_map = {jid: pt for jid, pt
                     in zip(CALIB_ORDER, self._calib_clicks)}
        ok = self._tracker.calibrate(candidates, click_map)
        if ok:
            self._calibrated = True
            self._calib_radius = float(np.median(
                [c.radius for c in candidates])) if candidates else 0
            self._save_calib_pos(click_map)
            self._init_tracker(click_map)
            self._manual_mode = False
            self.btn_manual.setChecked(False)
            self.lbl_calib_hint.setText("手动标定成功 ✓")
            self.lbl_status.setText("标定完成, 点击「播放」开始")
        else:
            self.lbl_calib_hint.setText("标定失败, 请重试")
            self.lbl_status.setText(
                f"候选 {len(candidates)} 个, 点击未匹配到足够圆点")

    # ─── 逐帧处理 ─────────────────────────────────────

    def _make_source(self):
        """流水线启动时调用 (GUI 线程): 返回 (读帧函数, 节拍间隔秒, 是否背压)"""
        self._timer.display_box = (self.video_label.width(),
                                   self.video_label.height())
        self._track_ts = 0.0          # 重置帧间隔计时
        if self._is_camera and self._camera_device is not None:
            self._cam_idx = self._frame_idx
            self._cam_reads = 0
            self._nominal_interval = 1.0 / 30.0       # 相机默认 30fps
            return self._camera_read_once, 0.0, False       # 相机: 丢旧帧, 保低延迟
        if self._cap is not None:
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            if not (1.0 <= fps <= 240.0):
                fps = 30.0
            self._nominal_interval = 1.0 / fps        # 视频标称帧间隔
            self._vid_idx = self._frame_idx
            return self._video_read_once, 1.0 / fps, True   # 视频: 原生 fps, 不丢帧
        return None

    def _camera_read_once(self):
        """[采集线程] 读一帧相机 (红外优先). 无帧返回 None, 设备故障抛异常"""
        dev = self._camera_device
        if dev is None:
            return None
        self._cam_reads += 1
        if self._cam_reads % 60 == 0 and not dev.is_available():
            raise RuntimeError("相机已断开")
        if not dev.update():
            return None
        frame, gray = self._get_camera_frames()
        if frame is None:
            return None
        self._cam_idx += 1
        return frame, gray, self._cam_idx

    def _video_read_once(self):
        """[采集线程] 解码一帧视频 (含缩放+灰度), 到结尾自动循环"""
        cap = self._cap
        if cap is None:
            return None
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._vid_idx = 0
            ret, frame = cap.read()
            if not ret:
                return None
        self._vid_idx += 1
        frame = self._resize_frame(frame)
        return frame, self._to_gray(frame), self._vid_idx

    def _pause_pipeline(self):
        """标定/重置等需要独占设备与追踪状态的操作之前调用"""
        if self._timer.isActive():
            self._timer.stop()
        if self._playing:
            self._playing = False
            self.btn_play.setText("▶ 播放")

    def _apply_cfg_changes(self):
        """重建检测器/平滑器 (流水线运行时由处理线程在帧间调用)"""
        c = self._cfg
        self._cfg_dirty = False
        self._detector = MarkerDetector(c.to_marker_config())
        self._tracker._max_jump = c.max_jump
        self._tracker._max_lost = c.max_lost_frames
        if self._smoother._win != max(2, c.smooth_window):
            self._smoother = AngleSmoother(c.smooth_window)

    def _process_core(self, frame, gray, idx):
        """
        [处理线程] 一帧的全部重活: 追踪 → 角度 → 指标 → 画 HUD → 缩放 → BGRA
        返回给 GUI 的只有一个现成的图像和几个数字。
        """
        t0 = time.perf_counter()
        metrics_disp = None
        with self._state_lock:
            if self._cfg_dirty:
                self._apply_cfg_changes()
            # 计算实际帧间隔 dt (相对于标称帧间隔的倍数, 1.0=正常)
            now_ts = time.perf_counter()
            if self._track_ts > 0 and self._nominal_interval > 0:
                dt = (now_ts - self._track_ts) / self._nominal_interval
                dt = float(np.clip(dt, 0.25, 3.0))
            else:
                dt = 1.0
            self._track_ts = now_ts
            if self._calibrated:
                # MediaPipe AI 姿态追踪 + 反光点精化; 失败回退到反光点追踪
                pixels, _src = self._mp_tracker.track_with_refine(frame, gray)
                if not pixels:
                    pixels = self._fallback_track(gray, dt)
                detected = len(pixels)
                self._last_pixels = pixels
            else:
                # 标定阶段: 全局检测 + tracker.update
                candidates = self._detect_adaptive(gray)
                pixels = self._tracker.update(candidates)
                detected = len(candidates)
            missing = len(pixels) < 8
            angles = self._smoother.smooth(compute_angles(pixels))
            if self._calibrated:
                self._metrics.update(pixels, angles)
                now = time.perf_counter()
                if now - self._metrics_ts >= 0.1:        # 指标面板 10Hz 刷新
                    self._metrics_ts = now
                    metrics_disp = self._metrics.get_display()
            ranges = dict(self._cfg.ranges)
        t1 = time.perf_counter()

        h, w = frame.shape[:2]
        vis = draw_visual(frame, pixels, angles, ranges,
                          missing=missing, frame_idx=idx)
        bw, bh = self._timer.display_box
        image = to_display_bgra(vis, bw, bh)
        t2 = time.perf_counter()
        return {
            "image": image, "src_w": w, "src_h": h, "frame_idx": idx,
            "detected": detected, "tracked": len(pixels), "angles": angles,
            "metrics": metrics_disp,
            "track_ms": (t1 - t0) * 1000.0, "draw_ms": (t2 - t1) * 1000.0,
        }

    def _on_pipeline_result(self, r):
        """[GUI 线程] 贴图 + 低频刷新文字, 不做任何重活"""
        self._frame_idx = r["frame_idx"]
        self.video_label.show_array(r["image"], r["src_w"], r["src_h"])
        if r["metrics"] is not None:
            self._update_metrics_display(r["metrics"])
        now = time.perf_counter()
        if now - self._ui_ts >= 0.2:                      # 状态条/HUD 标签 5Hz
            self._ui_ts = now
            self._prof = (r["track_ms"], r["draw_ms"])
            self._update_status(detected=r["detected"], tracked=r["tracked"],
                                angles=r["angles"])

    def _on_pipeline_failed(self, msg):
        """[GUI 线程] 采集线程报告设备故障 (流水线已自动停止)"""
        self._playing = False
        self.btn_play.setText("▶ 播放")
        self.lbl_status.setText(msg)
        if self._is_camera and self._camera_device is not None:
            self._check_camera_status()
            if (self._camera_device is not None
                    and self._camera_device.is_available()):
                self._schedule_device_retry()

    def _init_tracker(self, click_map):
        """标定后初始化追踪引擎 (associator + motion + mp_tracker)"""
        calib_pos = {int(jid): (x, y) for jid, (x, y) in click_map.items()}
        gray = None
        if self._first_frame is not None:
            gray = self._to_gray(self._first_frame)
        self._associator.set_calibration(calib_pos, self._calib_radius, gray)
        self._motion.seed(calib_pos)
        self._mp_tracker.calibrate(calib_pos)
        self._metrics.reset(calib_pos)  # 重置指标采集
        self._update_metrics_display(self._metrics.get_display())
        self._lost = {jid: 0 for jid in calib_pos}
        self._last_pixels = dict(calib_pos)

    def _fallback_track(self, gray, dt=1.0):
        """MediaPipe 失败时的反光点追踪后备"""
        tracked = list(self._calib_pos.keys())
        matched = self._associator.associate(
            gray, self._motion, tracked, self._last_pixels, self._lost, dt)
        # 只接受离标定不太远的匹配
        safe = {}
        for jid_int, (x, y) in matched.items():
            cx, cy = self._calib_pos.get(jid_int, (None, None))
            if cx is not None:
                lim = self._associator._calib_limit(jid_int)
                if np.hypot(x - cx, y - cy) > lim:
                    continue
            safe[int(jid_int)] = (x, y)
        self._motion.update(safe, dt)
        pixels = {}
        for jid_int in tracked:
            if jid_int in matched:
                x, y = matched[jid_int]
                pixels[jid_int] = (int(x), int(y))
                self._lost[jid_int] = 0
            else:
                self._lost[jid_int] = self._lost.get(jid_int, 0) + 1
                # 使用 MotionPredictor 内部的 lost_age 判断是否需要 reseed
                # (lost_age 包含被异常门拒绝的帧, 比外部 _lost 更准确)
                internal_lost = self._motion.lost_age(jid_int)
                if internal_lost > 8 or self._lost[jid_int] > 5:
                    cx, cy = self._calib_pos.get(jid_int, (None, None))
                    if cx is not None:
                        self._motion.reseed(jid_int, cx, cy)
                        self._lost[jid_int] = 0
                pred = self._motion.predict(jid_int, dt)
                if pred is not None:
                    cx, cy = self._calib_pos.get(jid_int, (None, None))
                    if cx is not None:
                        lim = self._associator._calib_limit(jid_int)
                        if np.hypot(pred[0] - cx, pred[1] - cy) > lim:
                            pred = (cx, cy)
                    pixels[jid_int] = (int(pred[0]), int(pred[1]))
        return pixels

    def _save_calib_pos(self, click_map):
        """保存标定位置和下肢骨骼长度"""
        self._calib_pos = {int(jid): (x, y) for jid, (x, y) in click_map.items()}
        self._calib_bones = {}
        # 下肢骨骼链: 髋-膝, 膝-踝, 踝-脚跟, 踝-趾骨, 脚跟-趾骨
        for a, b in [(4, 5), (5, 6), (6, 7), (6, 8), (7, 8)]:
            if a in self._calib_pos and b in self._calib_pos:
                ax, ay = self._calib_pos[a]
                bx, by = self._calib_pos[b]
                self._calib_bones[(a, b)] = float(np.hypot(ax - bx, ay - by))

    def _fix_lower_joints(self, matched):
        """
        骨骼长度约束修正下肢关节

        两步策略 (避免全排列过拟合):
            1. 髋-膝交换验证: 只验证 4/5 交换 (2种排列)
               膝运动到髋附近时, 全局贪心匹配可能把膝分给髋的位置
            2. 踝/脚跟/趾骨修正: 6/7/8 全排列 (6种)
               位置太近 (25-53px) 容易互相混淆
        """
        if not self._calib_bones:
            return
        # ── 步骤1: 髋-膝交换验证 ──
        self._swap_verify(matched, [4, 5])
        # ── 步骤2: 踝/脚跟/趾骨修正 ──
        self._perm_verify(matched, [6, 7, 8])

    def _swap_verify(self, matched, ids):
        """交换验证: 只验证两个关节是否需要交换"""
        if not all(jid in matched for jid in ids):
            return
        positions = [matched[jid] for jid in ids]
        idx = {jid: i for i, jid in enumerate(ids)}

        def calc_err(p):
            err = 0.0
            for (a, b), calib_len in self._calib_bones.items():
                if a in idx and b in idx:
                    cur = float(np.hypot(p[idx[a]][0] - p[idx[b]][0],
                                         p[idx[a]][1] - p[idx[b]][1]))
                    err += abs(cur - calib_len)
            return err

        orig_err = calc_err(positions)
        swapped = [positions[1], positions[0]]
        swap_err = calc_err(swapped)
        if swap_err < orig_err - 15:
            matched[ids[0]] = swapped[0]
            matched[ids[1]] = swapped[1]

    def _perm_verify(self, matched, ids):
        """全排列验证: 对多个关节尝试所有排列"""
        if not all(jid in matched for jid in ids):
            return
        positions = [matched[jid] for jid in ids]
        idx = {jid: i for i, jid in enumerate(ids)}

        def calc_err(p):
            err = 0.0
            for (a, b), calib_len in self._calib_bones.items():
                if a in idx and b in idx:
                    cur = float(np.hypot(p[idx[a]][0] - p[idx[b]][0],
                                         p[idx[a]][1] - p[idx[b]][1]))
                    err += abs(cur - calib_len)
            return err

        orig_err = calc_err(positions)
        best_perm = tuple(range(len(ids)))
        best_err = orig_err
        for perm in itertools.permutations(range(len(ids))):
            p = [positions[perm[i]] for i in range(len(ids))]
            err = calc_err(p)
            if err < best_err:
                best_err = err
                best_perm = perm
        if best_err < orig_err - 10 and best_perm != tuple(range(len(ids))):
            for i, jid in enumerate(ids):
                matched[jid] = positions[best_perm[i]]

    def _update_tracker_direct(self, matched):
        """
        直接用全局贪心匹配结果更新 tracker, 不经过最近邻匹配

        matched: {jid: (x, y)} 来自 _detect_local_roi 的全局贪心匹配
        返回: {jid: (x, y)} 当前帧像素坐标 (含沿用位置)
        """
        # 骨骼长度约束修正下肢关节
        self._fix_lower_joints(matched)
        pixels = {}
        for jid in self._tracker.tracked_ids:
            if jid in matched:
                x, y = matched[jid]
                self._tracker._last_pos[jid] = (float(x), float(y))
                self._tracker._lost[jid] = 0
                pixels[jid] = (int(x), int(y))
            else:
                # 未匹配: 沿用上一帧位置 (短暂丢失)
                lost = self._tracker._lost.get(jid, 0) + 1
                self._tracker._lost[jid] = lost
                if lost <= self._tracker._max_lost:
                    px = self._tracker._last_pos[jid]
                    pixels[jid] = (int(px[0]), int(px[1]))
        return pixels

    # ─── 自适应检测 ───────────────────────────────────

    def _detect_candidates(self, gray):
        """
        候选检测: 用卡尔曼预测位置做 ROI 中心, 收集去重候选

        返回: [(gx, gy, radius, brightness)] 供 TrackingEngine 匈牙利匹配
        """
        c = self._cfg
        h, w = gray.shape[:2]
        positions = self._engine.get_positions()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        raw = []  # [(gx, gy, lr, brightness)]

        for jid_int, (px, py) in positions.items():
            is_lower = jid_int >= 5
            roi_r = c.roi_radius * (2 if is_lower else 1)
            lost_cnt = self._engine._lost.get(jid_int, 0)
            if lost_cnt > 0:
                roi_r = min(roi_r * 2, w // 3)
            y1 = max(0, int(py - roi_r))
            y2 = min(h, int(py + roi_r + 1))
            x1 = max(0, int(px - roi_r))
            x2 = min(w, int(px + roi_r + 1))
            if x2 <= x1 or y2 <= y1:
                continue
            roi = gray[y1:y2, x1:x2]
            # 自适应阈值
            mx = float(roi.max())
            thr = max(20, int(mx * 0.4))
            _, binary = cv2.threshold(roi, thr, 255, cv2.THRESH_BINARY)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 3:
                    continue
                (lcx, lcy), lr = cv2.minEnclosingCircle(cnt)
                lr = float(lr)
                if not (2 <= lr <= c.max_radius):
                    continue
                per = cv2.arcLength(cnt, True)
                if per <= 0:
                    continue
                circ = 4 * np.pi * area / (per * per)
                if circ < 0.3:
                    continue
                gx = int(lcx + x1)
                gy = int(lcy + y1)
                # 标定位置约束
                if jid_int in self._calib_pos:
                    ccx, ccy = self._calib_pos[jid_int]
                    d_calib = float(np.hypot(gx - ccx, gy - ccy))
                    limit = (60 if jid_int <= 4 else
                             120 if jid_int <= 6 else 160)
                    if d_calib > limit:
                        continue
                # 亮度
                ry1 = max(0, int(lcy - lr))
                ry2 = min(roi.shape[0], int(lcy + lr + 1))
                rx1 = max(0, int(lcx - lr))
                rx2 = min(roi.shape[1], int(lcx + lr + 1))
                if ry2 > ry1 and rx2 > rx1:
                    br = float(np.mean(roi[ry1:ry2, rx1:rx2]))
                else:
                    br = 0.0
                raw.append((gx, gy, lr, br))

        # 去重: 位置相近的候选只保留亮度最高的
        raw.sort(key=lambda c4: -c4[3])
        deduped = []
        for gx, gy, lr, br in raw:
            if any(np.hypot(gx - dx, gy - dy) < c.min_distance
                   for dx, dy, _, _ in deduped):
                continue
            deduped.append((gx, gy, lr, br))
        return deduped

    def _detect_adaptive(self, gray, force_global=False):
        """
        检测反光点

        追踪阶段: 纯局部 ROI 检测 (绕过严格过滤, 靠位置约束)
            真实视频反光点亮度可能接近背景, 全局低阈值会引入噪点;
            局部 ROI 在每个已知位置附近取最亮圆点, 靠位置约束区分真实反光点。

        标定阶段: 全局检测 + 自适应降阈值

        force_global: True 时强制走全局检测 (标定/重标定场景,
            需要 list[CircleCandidate] 而非局部 ROI 的 dict 结果)
        """
        c = self._cfg
        # 追踪阶段: 纯局部 ROI 检测
        if self._tracker.calibrated and not force_global:
            return self._detect_local_roi(gray, [])
        # 标定阶段: 全局检测
        candidates = self._detector.detect(gray)
        if len(candidates) >= c.target_count:
            return candidates
        if c.auto_brightness:
            cur = c.min_brightness
            while len(candidates) < c.target_count and cur > 10:
                cur = max(10, cur - 20)
                candidates = MarkerDetector(
                    c.to_marker_config(brightness=cur)).detect(gray)
        return candidates

    def _detect_local_roi(self, gray, existing):
        """
        局部 ROI 检测: 自适应阈值 + 全局贪心匹配

        两阶段策略 (解决踝/脚跟/趾骨位置太近互相抢候选):
            1. 收集: 对每个关节, 在 ROI 内收集所有候选 (不只最佳)
            2. 全局贪心匹配: 所有 (关节, 候选, 评分) 按评分降序,
               依次分配 — 已分配关节跳过, 已占用位置跳过
               → 踝对自己位置评分最高, 先分配; 脚跟不能再匹配踝的位置
        """
        c = self._cfg
        h, w = gray.shape[:2]
        existing_pos = [(cd.cx, cd.cy) for cd in existing]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # ── 第一阶段: 收集每个关节的所有候选 ──
        all_matches = []  # [(score, jid, CircleCandidate), ...]

        for jid in self._tracker.tracked_ids:
            pos = self._tracker._last_pos.get(jid)
            if pos is None:
                continue
            # 关节自适应: 下肢搜索范围加倍
            is_lower = int(jid) >= 5
            base_roi = c.roi_radius * (2 if is_lower else 1)
            base_max = c.track_max_move * (2 if is_lower else 1)
            # 丢失时再扩大
            lost_cnt = self._tracker._lost.get(jid, 0)
            if lost_cnt > 0:
                cur_roi = min(base_roi * 2, w // 3)
                cur_max = float(min(base_roi * 2, w // 3))
            else:
                cur_roi = base_roi
                cur_max = base_max

            # 搜索中心: 预测位置优先, 上一帧位置后备
            pred = self._motion.predict(jid)
            centers = []
            if pred:
                centers.append(pred)
            centers.append(pos)

            found = False
            for cx, cy in centers:
                if found:
                    break
                y1 = max(0, int(cy - cur_roi))
                y2 = min(h, int(cy + cur_roi + 1))
                x1 = max(0, int(cx - cur_roi))
                x2 = min(w, int(cx + cur_roi + 1))
                if x2 <= x1 or y2 <= y1:
                    continue
                roi = gray[y1:y2, x1:x2]

                # 自适应阈值
                max_bright = float(roi.max())
                thr = max(20, int(max_bright * 0.4))
                _, binary = cv2.threshold(roi, thr, 255, cv2.THRESH_BINARY)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 3:
                        continue
                    (lcx, lcy), lr = cv2.minEnclosingCircle(cnt)
                    lr = float(lr)
                    if not (2 <= lr <= c.max_radius):
                        continue
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter <= 0:
                        continue
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity < 0.3:
                        continue
                    gx = int(lcx + x1)
                    gy = int(lcy + y1)
                    d = float(np.hypot(gx - cx, gy - cy))
                    if d > cur_max:
                        continue
                    # 标定位置约束: 打破正反馈循环
                    # 按关节区分阈值 — 下肢远端运动幅度更大
                    jid_int = int(jid)
                    if jid_int in self._calib_pos:
                        ccx, ccy = self._calib_pos[jid_int]
                        d_calib = float(np.hypot(gx - ccx, gy - ccy))
                        # 1-4: 60px, 5-6: 100px, 7-8: 140px
                        calib_limit = 60 if jid_int <= 4 else (
                            100 if jid_int <= 6 else 140)
                        if d_calib > calib_limit:
                            continue
                    ry1 = max(0, int(lcy - lr))
                    ry2 = min(roi.shape[0], int(lcy + lr + 1))
                    rx1 = max(0, int(lcx - lr))
                    rx2 = min(roi.shape[1], int(lcx + lr + 1))
                    if ry2 > ry1 and rx2 > rx1:
                        brightness = float(np.mean(roi[ry1:ry2, rx1:rx2]))
                    else:
                        brightness = 0.0
                    fill_ratio = (float(area / (np.pi * lr * lr))
                                  if lr > 0 else 0.0)
                    # 评分: 距离近(40%) + 亮度(25%) + 圆度(20%) + 大小一致(15%)
                    dist_s = 1.0 - d / cur_max if cur_max > 0 else 0.0
                    bright_s = brightness / 255.0
                    circ_s = circularity
                    size_s = 1.0
                    if self._calib_radius > 0:
                        size_s = max(0.0, 1.0 - abs(lr - self._calib_radius)
                                     / self._calib_radius)
                    score = (dist_s * 0.4 + bright_s * 0.25
                             + circ_s * 0.2 + size_s * 0.15)
                    all_matches.append((score, jid, gx, gy,
                                        CircleCandidate(
                                            cx=gx, cy=gy, radius=lr,
                                            area=float(area),
                                            circularity=float(circularity),
                                            fill_ratio=fill_ratio,
                                            brightness=brightness)))
                    found = True

        # ── 第二阶段: 全局贪心匹配 ──
        all_matches.sort(key=lambda m: -m[0])  # 按评分降序
        assigned_jids = set()
        matched = {}  # {jid: (gx, gy)} — 带关节ID, 不再经过 tracker.update
        for score, jid, gx, gy, cc in all_matches:
            if jid in assigned_jids:
                continue  # 该关节已分配
            # 检查位置是否已被占用
            if any(np.hypot(gx - ex, gy - ey) < c.min_distance
                   for ex, ey in existing_pos):
                continue
            matched[jid] = (gx, gy)
            assigned_jids.add(jid)
            existing_pos.append((gx, gy))
        return matched

    # ─── 工具 ─────────────────────────────────────────

    def _ir_to_bgr_gray(self, ir):
        """Kinect 红外帧 (16-bit/8-bit, 512x424) → (960宽 BGR, 960宽 gray)

        红外帧中反光点清晰可见 (反射 Kinect 投射的红外光)。
        返回放大后的 BGR (给 MediaPipe) 和 gray (给 top-hat)。
        """
        if ir.dtype == np.uint16:
            gray = (ir >> 8).astype(np.uint8)
        else:
            gray = ir.copy() if ir.ndim == 2 else cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
        # 放大到 960 宽度 (512→960, 保持比例 424→794)
        h, w = gray.shape[:2]
        if w < 960:
            new_w = 960
            new_h = max(1, int(h * new_w / w))
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return bgr, gray

    def _get_camera_frames(self):
        """相机模式: 获取帧 (优先红外, 反光点可见)

        Returns: (bgr, gray) 或 (None, None)
        """
        if self._camera_device is None:
            return None, None
        ir = self._camera_device.get_ir_frame()
        if ir is not None and ir.size > 0:
            return self._ir_to_bgr_gray(ir)
        # 回退彩色帧
        color = self._camera_device.get_color_frame()
        if color is not None:
            color = self._resize_frame(color)
            return color, self._to_gray(color)
        return None, None

    def _to_gray(self, frame):
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _resize_frame(self, frame):
        """缩放帧到合理大小 (宽度<=960), 减少检测/渲染开销"""
        h, w = frame.shape[:2]
        if w <= 960:
            return frame
        new_w = 960
        new_h = max(1, int(h * new_w / w))
        return cv2.resize(frame, (new_w, new_h),
                          interpolation=cv2.INTER_AREA)

    def _show_frame(self, cv_img, candidates=None):
        """显示 OpenCV 帧到 QLabel (性能优化: 避免不必要的全帧 copy)"""
        if cv_img is None:
            return

        # 仅在需要绘制标定标记时才 copy, 否则直接使用
        needs_overlay = (
            (not self._calibrated and self._first_candidates) or
            self._manual_mode
        )
        if needs_overlay:
            vis = (cv_img.copy() if cv_img.ndim == 3
                   else cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR))

            # 未标定: 候选圆圈 (直接画, 无光晕 copy)
            if not self._calibrated and self._first_candidates:
                for c in self._first_candidates:
                    cv2.circle(vis, (c.cx, c.cy), int(c.radius) + 2,
                               HOLO_NODE, 1)
                    cv2.circle(vis, (c.cx, c.cy), 2, HOLO_NODE, -1)

            # 手动标定: 十字准星 + 外环 + 编号 (直接画, 无光晕 copy)
            if self._manual_mode:
                for i, (x, y) in enumerate(self._calib_clicks):
                    pt = (int(x), int(y))
                    cv2.drawMarker(vis, pt, HOLO_NODE,
                                   cv2.MARKER_CROSS, 20, 1)
                    cv2.circle(vis, pt, 10, HOLO_NODE, 2)
                    cv2.putText(vis, POINT_EN[i + 1], (pt[0] + 12, pt[1] - 12),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5,
                                HOLO_NODE, 1)
                n = len(self._calib_clicks)
                if n < 8:
                    cv2.putText(vis, f"CLICK #{n+1}: {POINT_EN[n+1]}",
                                (12, 25), cv2.FONT_HERSHEY_DUPLEX,
                                0.5, HOLO_NODE, 1)
                elif n == 8:
                    cv2.putText(vis, "CALIBRATING...", (12, 25),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5,
                                HOLO_GREEN_BGR, 1)
        else:
            vis = cv_img

        h, w = vis.shape[:2]
        self.video_label.show_array(vis, w, h)

    def _update_status(self, detected=0, tracked=0, angles=None):
        """更新底部状态条 + HUD 标签 (setStyleSheet 仅在状态变化时调用)"""
        src_on = self._cap is not None or self._camera_device is not None
        # ─ 底部状态条 ─
        if self._is_camera:
            parts = [f"相机 帧:{self._frame_idx}"]
        else:
            parts = [f"帧:{self._frame_idx}/{self._total_frames}"]
        if src_on:
            parts.append(f"检测:{detected}")
            if self._calibrated:
                parts.append(f"追踪:{tracked}/8")
                if angles:
                    for n in JOINT_NAMES:
                        a = angles.get(n)
                        if a is not None:
                            lo, hi = self._cfg.ranges[n]
                            flag = "正常" if lo <= a <= hi else "超限"
                            parts.append(f"{n}:{a:.0f}({flag})")
        prof = self._prof
        if prof is not None and self._timer.isActive():
            parts.append(
                f"采集{self._timer.cap_fps:.0f}/处理{self._timer.proc_fps:.0f}fps"
                f" 追踪{prof[0]:.0f}ms 绘制{prof[1]:.0f}ms")
        self.lbl_status.setText("  |  ".join(parts))

        # ─ HUD 状态灯 ─
        if hasattr(self, 'lbl_hud_status'):
            if src_on and self._playing:
                color = HOLO_GREEN if self._calibrated else HOLO_ORANGE
                text = "● 追踪中" if self._calibrated else "● 预览中"
            elif src_on:
                color = HOLO_DIM
                text = "● 就绪"
            else:
                color = HOLO_DIM
                text = "● 待机"
            if text != self.lbl_hud_status.text():
                self.lbl_hud_status.setText(text)
                self.lbl_hud_status.setStyleSheet(
                    f"color:{color};font-size:11px;"
                    f"font-family:Consolas,monospace;background:transparent;")

        # ─ HUD 模式标签 ─
        if hasattr(self, 'lbl_hud_mode'):
            if self._is_camera:
                mode = "模式:相机"
            elif self._cap is not None:
                mode = "模式:视频"
            else:
                mode = "模式:--"
            if mode != self.lbl_hud_mode.text():
                self.lbl_hud_mode.setText(mode)

        # ─ HUD 帧率 (流水线实测处理帧率) ─
        if hasattr(self, 'lbl_hud_fps'):
            self.lbl_hud_fps.setText(f"帧率:{self._timer.proc_fps:.0f}")

        # ─ HUD 点数 ─
        if hasattr(self, 'lbl_hud_points'):
            self.lbl_hud_points.setText(f"点数:{tracked}/8")

        # ─ 数据面板系统状态 ─
        if hasattr(self, '_lbl_sys_status'):
            if src_on and self._playing:
                sys_txt = f"系统:运行  帧:{self._frame_idx}"
            elif src_on:
                sys_txt = "系统:就绪"
            else:
                sys_txt = "系统:待机"
            self._lbl_sys_status.setText(sys_txt)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if hasattr(self, '_timer'):
            self._timer.display_box = (self.video_label.width(),
                                       self.video_label.height())
        if self._first_frame is not None and not self._playing:
            self._show_frame(self._first_frame)

    def closeEvent(self, ev):
        self._timer.stop()
        self._close_video()
        super().closeEvent(ev)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # 全息主题: 深色调色板
    pal = app.palette()
    from PyQt6.QtGui import QColor
    pal.setColor(pal.ColorRole.Window, QColor(HOLO_BG))
    pal.setColor(pal.ColorRole.WindowText, QColor(HOLO_TEXT))
    pal.setColor(pal.ColorRole.Base, QColor(HOLO_PANEL))
    pal.setColor(pal.ColorRole.Text, QColor(HOLO_TEXT))
    pal.setColor(pal.ColorRole.Button, QColor(HOLO_PANEL_2))
    pal.setColor(pal.ColorRole.ButtonText, QColor(HOLO_CYAN))
    pal.setColor(pal.ColorRole.Highlight, QColor(HOLO_BORDER))
    app.setPalette(pal)
    gui = PoseAnalysisGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
