# -*- coding: utf-8 -*-
"""
骑行者模型 (无 Qt 依赖): 把左 / 右侧相机的数据汇总, 生成"正面膝盖图"所需的几何。

数据来源 (哪些是测量的, 哪些是模拟的):
    测量   垂直方向: 髋/膝/踝/脚跟/趾骨相对髋的高度 (侧视图和正面图共用竖直轴)
           膝关节屈曲角 (侧视)
           曲柄相位 (由踝点绕曲柄中心的角度得到)
    估计   只接一侧相机时, 另一条腿 = 同一条腿在"相位 + 180°"时的姿态
           (曲柄两侧相差 180°, 用近侧腿按相位建立的模板回放)
    模拟   左右方向的膝盖偏摆: 侧面相机看不到, 由模型生成:
               偏摆 = 幅度 × 屈曲程度 × (0.5 + 0.5·sin 相位)
           幅度默认 1.2 cm, 可调; 若以后有真实横向数据 (如 Kinect 深度),
           通过 lateral_cm 传入即可替换模拟值。

约定: 正面图是"面对骑行者"的视角, 骑行者右腿显示在画面左侧。
"""
import math
from collections import deque

import numpy as np

JID = {"hip": 4, "knee": 5, "ankle": 6, "heel": 7, "toe": 8}
_BINS = 24
TH_DEG = 4.0          # 正面膝角偏差阈值: ≥ +TH 外翻(红), ≤ -TH 内翻(蓝)


def detect_side(pixels):
    """画面里手腕(1)在髋(4)右边 → 骑行者朝右 → 相机在其右侧 → 'R'; 反之 'L'"""
    p1, p4 = pixels.get(1), pixels.get(4)
    if p1 is None or p4 is None:
        return None
    return "R" if p1[0] > p4[0] else "L"


def _angle(a, b, c):
    """以 b 为顶点, a-b-c 的夹角 (度)"""
    v1, v2 = np.asarray(a, float) - b, np.asarray(c, float) - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(v1.dot(v2) / (n1 * n2), -1, 1))))


class _Leg:
    """一侧腿: 最新采样 + 按曲柄相位分箱的姿态模板"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.t = None
        self.rel = None              # {jid: (dx, dy)} 以大腿长为单位, 原点=髋
        self.u = None                # 大腿 2D 长度 (px, EMA)
        self.phase = None
        self.last_key = None
        self.ring = deque(maxlen=64)             # 踝点相对位置, 用来估曲柄中心
        self.tpl = np.zeros((_BINS, 8))          # knee/ankle/heel/toe 的 (dx,dy)
        self.cnt = np.zeros(_BINS)

    def push(self, pixels, t):
        if any(j not in pixels for j in JID.values()):
            return
        hip = np.asarray(pixels[4], float)
        u = float(np.hypot(*(np.asarray(pixels[5], float) - hip)))
        if u < 10:
            return
        self.u = u if self.u is None else 0.9 * self.u + 0.1 * u
        rel = {j: tuple((np.asarray(pixels[j], float) - hip) / self.u)
               for j in (5, 6, 7, 8)}
        self.t, self.rel = t, rel
        key = np.array([pixels[j] for j in (4, 5, 6, 7, 8)], float).ravel()
        if self.last_key is not None and np.abs(key - self.last_key).max() < 0.6:
            return                                # 重复帧: 只刷新时间, 不更新统计
        self.last_key = key
        self.ring.append(rel[6])
        self.phase = self._phase(rel[6])
        if self.phase is not None:
            b = int(self.phase / (2 * math.pi) * _BINS) % _BINS
            vec = np.array([*rel[5], *rel[6], *rel[7], *rel[8]])
            self.tpl[b] = vec if self.cnt[b] == 0 else 0.5 * self.tpl[b] + 0.5 * vec
            self.cnt[b] += 1

    def _phase(self, ankle_rel):
        if len(self.ring) < 6:
            return None
        c = np.mean(self.ring, axis=0)
        return math.atan2(ankle_rel[1] - c[1], ankle_rel[0] - c[0]) % (2 * math.pi)

    def template(self, phase):
        """按相位取姿态 (已填充的箱之间循环插值); 覆盖不足返回 None"""
        filled = np.where(self.cnt > 0)[0]
        if len(filled) < 4 or phase is None:
            return None
        x = phase / (2 * math.pi) * _BINS
        xs = np.concatenate([filled - _BINS, filled, filled + _BINS]).astype(float)
        out = []
        for k in range(8):
            ys = np.tile(self.tpl[filled, k], 3)
            out.append(float(np.interp(x, xs, ys)))
        return {5: (out[0], out[1]), 6: (out[2], out[3]),
                7: (out[4], out[5]), 8: (out[6], out[7])}


class RiderModel:
    def __init__(self, thigh_cm=43.0, hip_w_cm=18.0, foot_w_cm=20.0,
                 sway_cm=1.2, stale_s=0.6, simulate=True):
        self.thigh_cm, self.hip_w_cm, self.foot_w_cm = thigh_cm, hip_w_cm, foot_w_cm
        self.sway_cm, self.stale_s, self.simulate = sway_cm, stale_s, simulate
        self.lateral_cm = {}                      # 外部真实横向偏摆 {'L': cm, 'R': cm}
        self.side = None
        self.legs = {"L": _Leg(), "R": _Leg()}
        self._last_t = -1e9

    def reset(self):
        for leg in self.legs.values():
            leg.reset()
        self._last_t = -1e9

    def set_calibration(self, calib_pos, side=None):
        """标定后调用; side 缺省时按朝向自动判断 (单相机)"""
        self.side = side or detect_side({int(j): p for j, p in calib_pos.items()})
        return self.side

    def push(self, pixels, t, side=None):
        side = side or self.side or detect_side(pixels)
        if side not in self.legs:
            return
        if t < self._last_t - 0.5:                # 视频循环回绕: 清掉旧时间戳
            for leg in self.legs.values():
                leg.t = None
        self._last_t = t
        self.legs[side].push({int(j): p for j, p in pixels.items()}, t)

    # ── 生成正面图几何 ──
    def front_state(self, now):
        fresh = {s: (leg.t is not None and 0 <= now - leg.t <= self.stale_s
                     and leg.rel is not None)
                 for s, leg in self.legs.items()}
        out = {"geom": {"hip_w": self.hip_w_cm, "foot_w": self.foot_w_cm},
               "simulated": self.simulate, "L": None, "R": None}
        for s in ("L", "R"):
            other = "R" if s == "L" else "L"
            if fresh[s]:
                rel, phase, src = self.legs[s].rel, self.legs[s].phase, "measured"
            elif fresh[other] and self.legs[other].phase is not None:
                phase = (self.legs[other].phase + math.pi) % (2 * math.pi)
                rel = self.legs[other].template(phase)
                src = "estimated"
                if rel is None:
                    continue
            else:
                continue
            out[s] = self._leg_geometry(s, rel, phase, src)
        return out

    def _leg_geometry(self, s, rel, phase, src):
        cm = self.thigh_cm
        sign = -1.0 if s == "R" else 1.0          # 面对骑行者: 右腿在画面左侧
        hip = (sign * self.hip_w_cm / 2.0, 0.0)
        ankle_dy = max(1.0, rel[6][1] * cm)
        knee_dy = float(np.clip(rel[5][1] * cm, 0.0, ankle_dy))
        ankle = (sign * self.foot_w_cm / 2.0, ankle_dy)
        frac = knee_dy / ankle_dy
        neutral_x = hip[0] + (ankle[0] - hip[0]) * frac
        theta = _angle((0.0, 0.0), rel[5], rel[6])              # 侧视膝内角
        flex = float(np.clip((170.0 - theta) / 110.0, 0.0, 1.0))
        if s in self.lateral_cm:                                 # 真实数据优先
            dev_cm, mode = float(self.lateral_cm[s]), "measured"
        elif self.simulate:
            ph = 0.0 if phase is None else phase
            dev_cm = self.sway_cm * flex * (0.5 + 0.5 * math.sin(ph))
            mode = "simulated"
        else:
            dev_cm, mode = 0.0, "neutral"
        knee = (neutral_x - sign * dev_cm, knee_dy)              # 向中线为内翻/外翻方向
        ang = _angle(hip, knee, ankle)
        dev_deg = (180.0 - ang) * (1.0 if dev_cm >= 0 else -1.0)
        status = ("valgus" if dev_deg >= TH_DEG
                  else "varus" if dev_deg <= -TH_DEG else "neutral")
        return {"source": src, "lateral_mode": mode, "hip": hip, "knee": knee,
                "ankle": ankle, "dev_cm": dev_cm, "dev_deg": dev_deg,
                "status": status, "flex": flex, "phase": phase}
