# -*- coding: utf-8 -*-
"""
骑行姿态实时运动学分析程序 (2D)

⚠ 重要声明 (务必阅读):
    本程序计算的是 2D 图像平面内的角度 (仅 X/Y 像素投影),
    不是真实三维物理角度。当关节存在侧向运动或相机视角偏斜时,
    2D 角度会与真实角度存在偏差。如需真实三维关节角度,
    必须先完成相机标定, 并用 3D 空间坐标 (如 Kinect 深度空间,
    device.get_camera_pos) 配合 core.geometry.angle_3d 计算。

依赖安装:
    pip install opencv-python numpy

运行:
    python pose_analysis/main.py

复用的当前系统模块 (不重复造轮子):
    vision.MarkerDetector   8 分光点捕捉 + 多层干扰排除 (形状/亮度/大小一致性/遮罩)
    vision.MarkerTracker    首帧标定 + 跨帧身份保持 (最近邻+运动预测+短暂丢失沿用)
    vision.pixels_to_joints 像素 → 3D (无设备时伪 3D, z=0)
    core.models.JointID     编号 1~8
    core.geometry.angle_2d  三点 2D 角度 (顶点为中点)
    capture.VideoDevice     本地视频读取 (循环播放)
"""

# ==================== 可配置参数 (代码最上方) ====================

# 输入源: True=合成红外演示视频(无需文件, 立即运行); False=读取 VIDEO_PATH
DEMO_MODE = True
VIDEO_PATH = "video/your_ir_video.mp4"   # DEMO_MODE=False 时填真实红外视频路径

# 反光点检测参数 (映射到 MarkerConfig)
BRIGHTNESS_THRESHOLD = 120    # 反光点亮度下限 (0-255)
MIN_CONTOUR_AREA = 5          # 标记点最小轮廓面积
MAX_CONTOUR_AREA = 500        # 标记点最大轮廓面积
MIN_RADIUS = 3                # 反光点半径下限 (px)
MAX_RADIUS = 15               # 反光点半径上限 (px)

# 每个关节角度阈值 [最小值, 最大值] (单位: 度), 超出则红色告警
# 注: 示例阈值适配骑行 2D 投影 (肩角偏小/髋角偏大); 真实场景需按实测标定
ELBOW_RANGE    = [60, 110]    # 肘
SHOULDER_RANGE = [5, 40]     # 肩 (2D 下偏小)
HIP_RANGE      = [100, 180]  # 髋 (2D 下偏大)
KNEE_RANGE     = [70, 130]   # 膝 (踏车周期变化, 展示告警切换)
ANKLE_RANGE    = [80, 110]    # 踝

# 帧平滑窗口 (越大越平滑但滞后越大)
SMOOTH_WINDOW = 5

# ====================================================================

import sys
import os
from collections import deque

# 将项目根加入 sys.path, 以复用 vision/core/capture 等现有模块
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import cv2

from vision import MarkerDetector, MarkerConfig, MarkerTracker, pixels_to_joints
from core.models import JointID
from core.geometry import angle_2d
from capture.video_device import VideoDevice


# 标记点编号 (严格固定 1~8); OpenCV putText 不支持中文, 故用数字编号
POINT_LABEL = {
    JointID.WRIST: "1", JointID.ELBOW: "2", JointID.SHOULDER: "3",
    JointID.HIP: "4", JointID.KNEE: "5", JointID.ANKLE: "6",
    JointID.HEEL: "7", JointID.PHALANX: "8",
}
# 编号 → 中文名 (仅用于控制台提示)
POINT_CN = {
    1: "腕关节", 2: "肘关节", 3: "肩关节", 4: "髋关节",
    5: "膝关节", 6: "踝关节", 7: "脚跟", 8: "趾骨",
}

# 关节角度阈值表
JOINT_RANGES = {
    "elbow": ELBOW_RANGE, "shoulder": SHOULDER_RANGE,
    "hip": HIP_RANGE, "knee": KNEE_RANGE, "ankle": ANKLE_RANGE,
}

# 骨骼连线: (起点, 终点, 所属关节名) — 颜色按所属关节是否超限
SEGMENTS = [
    (JointID.WRIST,    JointID.ELBOW,    "elbow"),
    (JointID.ELBOW,    JointID.SHOULDER, "shoulder"),
    (JointID.SHOULDER, JointID.HIP,      "hip"),
    (JointID.HIP,      JointID.KNEE,     "knee"),
    (JointID.KNEE,     JointID.ANKLE,    "ankle"),
    (JointID.ANKLE,    JointID.HEEL,     "ankle"),
    (JointID.ANKLE,    JointID.PHALANX,  "ankle"),
]

# 角度数值标注: (关节名, 标注位置关节)
ANGLE_LABELS = [
    ("elbow", JointID.ELBOW), ("shoulder", JointID.SHOULDER),
    ("hip", JointID.HIP), ("knee", JointID.KNEE),
    ("ankle", JointID.ANKLE),
]

GREEN = (0, 255, 0)    # 正常 (BGR)
RED   = (0, 0, 255)    # 告警


def _in_range(name, ang):
    """角度是否在阈值范围内 (缺失时不告警)"""
    if ang is None:
        return True
    lo, hi = JOINT_RANGES[name]
    return lo <= ang <= hi


# --- 合成红外演示帧 (DEMO_MODE 用, 模拟 8 个分光点踏车运动) ---
# 基准位置: 上肢/髋/膝/踝; 脚跟/趾骨随踝同步 (相对偏移固定),
# 保证脚部三点相对距离恒定, 避免运动中两点过近被去重/连通。
# 相邻点间距均 > 30px (大于 2*radius + min_distance 安全裕度)。
_DEMO_BASE = {
    JointID.WRIST: (320, 180), JointID.ELBOW: (270, 210),
    JointID.SHOULDER: (200, 150), JointID.HIP: (270, 280),
    JointID.KNEE: (350, 310), JointID.ANKLE: (350, 380),
}
_FOOT_OFFSET = {   # 脚跟/趾骨相对踝的偏移 (脚部同步运动)
    JointID.HEEL: (-30, 15), JointID.PHALANX: (45, 5),
}


def make_demo_frame(idx, w=512, h=424):
    """生成一帧合成红外画面 (黑底 + 8 个亮圆点做踏车运动)

    Returns:
        (ir_frame, known_points): 已知位置用于 demo 自动标定
    """
    img = np.zeros((h, w), dtype=np.uint8)
    t = idx * 0.12
    pts = {}
    # 上肢/髋/膝/踝: 各自周期摆动 (腿部幅度大模拟踏车)
    for jid, (bx, by) in _DEMO_BASE.items():
        amp = 20 if jid.value >= 5 else (3 if jid == JointID.HIP else 5)
        phase = jid.value * 0.6
        x = int(bx + amp * np.cos(t + phase))
        y = int(by + amp * np.sin(t + phase) * 0.5)
        pts[jid] = (x, y)
    # 脚跟/趾骨随踝同步 (相对偏移固定, 三点一起运动)
    ax, ay = pts[JointID.ANKLE]
    for jid, (dx, dy) in _FOOT_OFFSET.items():
        pts[jid] = (ax + dx, ay + dy)
    # 画 8 个亮圆点
    for x, y in pts.values():
        cv2.circle(img, (x, y), 8, 240, -1)
    return img, pts


# --- 角度平滑器 (移动平均, 避免角度剧烈跳动) ---
class AngleSmoother:
    def __init__(self, win):
        self._buf = {name: deque(maxlen=max(2, win)) for name in JOINT_RANGES}

    def smooth(self, angles):
        out = {}
        for name, ang in angles.items():
            if ang is None:
                out[name] = None
                continue
            self._buf[name].append(ang)
            buf = self._buf[name]
            out[name] = sum(buf) / len(buf)
        return out


# --- 关节角度计算 (2D 像素平面) ---
def compute_angles(pixels):
    """
    基于 2D 像素坐标计算 5 个关节角度 (度)

    pixels: {JointID: (cx, cy)}
    """
    angles = {}
    if not pixels:
        return {n: None for n in JOINT_RANGES}

    # 像素 → Point3D (无设备伪 3D, z=0; angle_2d 仅用 X/Y)
    # 注: pixels_to_joints 对 X/Y 做同向缩放+平移, 不影响角度
    joints = pixels_to_joints(pixels, device=None)

    def has(*ids):
        return all(j in joints for j in ids)

    # 肘关节: 点1(腕) - 点2(肘) - 点3(肩), 顶点=肘
    if has(JointID.WRIST, JointID.ELBOW, JointID.SHOULDER):
        angles["elbow"] = angle_2d(joints[JointID.WRIST],
                                   joints[JointID.ELBOW],
                                   joints[JointID.SHOULDER])
    # 肩关节: 点2(肘) - 点3(肩) - 点4(髋), 顶点=肩
    if has(JointID.ELBOW, JointID.SHOULDER, JointID.HIP):
        angles["shoulder"] = angle_2d(joints[JointID.ELBOW],
                                      joints[JointID.SHOULDER],
                                      joints[JointID.HIP])
    # 髋关节: 点3(肩) - 点4(髋) - 点5(膝), 顶点=髋
    if has(JointID.SHOULDER, JointID.HIP, JointID.KNEE):
        angles["hip"] = angle_2d(joints[JointID.SHOULDER],
                                  joints[JointID.HIP],
                                  joints[JointID.KNEE])
    # 膝关节: 点4(髋) - 点5(膝) - 点6(踝), 顶点=膝
    if has(JointID.HIP, JointID.KNEE, JointID.ANKLE):
        angles["knee"] = angle_2d(joints[JointID.HIP],
                                   joints[JointID.KNEE],
                                   joints[JointID.ANKLE])
    # 踝关节: 点5(膝) - 点6(踝) - 点7(脚跟) - 点8(趾骨)
    #   小腿向量 = 膝→踝, 脚掌向量 = 脚跟→趾骨, 求二者夹角
    if has(JointID.KNEE, JointID.ANKLE, JointID.HEEL, JointID.PHALANX):
        k = joints[JointID.KNEE]
        a = joints[JointID.ANKLE]
        h = joints[JointID.HEEL]
        p = joints[JointID.PHALANX]
        v_shin = np.array([a.x - k.x, a.y - k.y])   # 小腿方向
        v_foot = np.array([p.x - h.x, p.y - h.y])   # 脚掌方向
        n1, n2 = np.linalg.norm(v_shin), np.linalg.norm(v_foot)
        if n1 > 1e-6 and n2 > 1e-6:
            cos = float(np.clip(v_shin.dot(v_foot) / (n1 * n2), -1.0, 1.0))
            angles["ankle"] = float(np.degrees(np.arccos(cos)))

    # 缺失关节填 None
    for n in JOINT_RANGES:
        angles.setdefault(n, None)
    return angles


# --- 可视化绘制 ---
def draw_visual(ir_frame, pixels, angles, missing=False):
    """在红外帧上绘制骨骼/编号/角度, 返回 BGR 画面"""
    vis = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)

    if missing:
        cv2.putText(vis, "Marker LOST!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)

    if not pixels:
        return vis

    # 1) 骨骼连线 (颜色按所属关节是否超限)
    for a_id, b_id, jname in SEGMENTS:
        if a_id in pixels and b_id in pixels:
            ang = angles.get(jname)
            color = GREEN if _in_range(jname, ang) else RED
            cv2.line(vis, pixels[a_id], pixels[b_id], color, 2)

    # 2) 标记点圆圈 + 编号 1~8
    for jid, (x, y) in pixels.items():
        cv2.circle(vis, (x, y), 10, (255, 255, 0), 1)
        cv2.putText(vis, POINT_LABEL[jid], (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 3) 关节角度数值 (颜色按是否超限)
    for jname, jid in ANGLE_LABELS:
        if jid not in pixels:
            continue
        ang = angles.get(jname)
        if ang is None:
            continue
        x, y = pixels[jid]
        color = GREEN if _in_range(jname, ang) else RED
        cv2.putText(vis, f"{jname}:{ang:.0f}", (x + 10, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis


# --- 首帧标定 (鼠标交互, 按 1~8 顺序点击) ---
class Calibrator:
    """首帧手动标定: 依次点击 8 个点, 按 1~8 编号顺序"""

    ORDER = [JointID.WRIST, JointID.ELBOW, JointID.SHOULDER, JointID.HIP,
             JointID.KNEE, JointID.ANKLE, JointID.HEEL, JointID.PHALANX]

    def __init__(self):
        self.clicks = []

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.clicks) < 8:
            self.clicks.append((x, y))

    def click_map(self):
        return {jid: pt for jid, pt in zip(self.ORDER, self.clicks)}

    @property
    def done(self):
        return len(self.clicks) >= 8

    @property
    def next_label(self):
        i = len(self.clicks)
        if i < 8:
            return f"Click #{i+1} ({POINT_CN[i+1]})"
        return "Calibration done"


def main():
    # 检测器配置 (面积上下限已直接传入, 不必再换算半径)
    cfg = MarkerConfig(
        min_radius=MIN_RADIUS, max_radius=MAX_RADIUS,
        min_area=float(MIN_CONTOUR_AREA), min_brightness=BRIGHTNESS_THRESHOLD,
    )
    detector = MarkerDetector(cfg)
    tracker = MarkerTracker(max_jump=80, max_lost_frames=15)
    smoother = AngleSmoother(SMOOTH_WINDOW)
    calib = Calibrator()

    win = "riding_pose (ESC to quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, calib.on_mouse)

    device = None
    if not DEMO_MODE:
        device = VideoDevice(video_path=VIDEO_PATH)
        if not device.open():
            print(f"无法打开视频: {VIDEO_PATH}")
            return

    frame_idx = 0
    calibrated = False
    print("=" * 50)
    print("骑行姿态实时运动学分析 (2D)")
    print("  DEMO_MODE=True: 合成红外视频, 自动标定, 立即演示")
    print("  DEMO_MODE=False: 真实视频, 依次点击8个点完成标定")
    print("  按 ESC 退出")
    print("  注: 本程序为 2D 平面角度, 非真实三维角度")
    print("=" * 50)

    while True:
        # 取帧
        if DEMO_MODE:
            ir, demo_pts = make_demo_frame(frame_idx)
        else:
            if not device.update():
                continue
            ir = device.get_ir_frame()
            if ir is None:
                continue
            demo_pts = None

        if ir is None:
            break
        if ir.ndim == 3:
            ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)

        # 捕捉 + 干扰排除
        candidates = detector.detect(ir)

        if not calibrated:
            # --- 标定阶段 ---
            if DEMO_MODE and demo_pts is not None:
                tracker.calibrate(candidates, demo_pts)   # demo 自动标定
                calibrated = True
            else:
                if calib.done:
                    tracker.calibrate(candidates, calib.click_map())
                    calibrated = True

            # 标定前画面: 显示候选 + 提示
            vis = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
            for c in candidates:
                cv2.circle(vis, (c.cx, c.cy), int(c.radius) + 2,
                           (0, 255, 0), 1)
            if not DEMO_MODE:
                cv2.putText(vis, calib.next_label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.imshow(win, vis)
        else:
            # --- 追踪 + 分析阶段 ---
            pixels = tracker.update(candidates)           # {JointID:(x,y)}
            missing = len(pixels) < 8
            angles = compute_angles(pixels)               # 5 关节角度
            angles = smoother.smooth(angles)              # 平滑
            vis = draw_visual(ir, pixels, angles, missing=missing)
            cv2.imshow(win, vis)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break
        frame_idx += 1

    if device:
        device.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
