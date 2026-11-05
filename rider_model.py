# -*- coding: utf-8 -*-
"""
骑行者模型 (无 Qt 依赖): 把左 / 右侧相机的数据汇总, 生成"正面图"所需的几何。

髋、肩、膝都会动, 不是固定点。各量的数据来源:

  测量 (侧面相机直接给出, 竖直轴与正面图共用)
        髋 / 肩 / 腕 / 膝 / 踝 的竖直位置 → 骨盆上下起伏、肩上下起伏、手高度
        膝屈曲角、曲柄相位 (踝点绕曲柄中心的角度)
  估计   只接一侧相机时, 另一条腿 = 同一条腿在"相位 + 180°"时的姿态
  模拟   侧面相机看不到的"左右"量, 由实测的两腿伸展差驱动:
            平衡量 balance = (右腿伸展 - 左腿伸展) / 近期幅度   ∈ [-1, 1]
            骨盆左右平移 = -balance × hip_sway_cm      (重心移向发力腿一侧)
            骨盆倾斜     = +balance × hip_tilt_deg      (伸展侧髋略抬高)
            肩左右平移   = 骨盆平移 × shoulder_follow  (躯干跟随)
            肩线倾斜     = -0.6 × 骨盆倾斜            (躯干反向补偿)
            膝左右偏摆   = 幅度 × 屈曲 × (0.5 + 0.5·sin 相位)
        若有真实横向数据 (如 Kinect 深度), 通过 lateral_cm / hip_lateral_cm /
        shoulder_lateral_cm 传入即可替换对应的模拟值。

约定: 正面图是"面对骑行者"的视角, 骑行者右腿显示在画面左侧; x 向右为正, y 向下为正。
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
    """一侧相机的数据: 最新采样 + 姿态模板 + 髋/肩/手的竖直位置"""

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
        self.hip_ref = None          # 髋高度的慢速均值 (px), 用来求上下起伏
        self.bob = 0.0               # 髋上下起伏 (cm, 向下为正)
        self.sh_dy = None            # 肩相对髋的竖直位置 (cm, 向上为负)
        self.hand_dy = None          # 腕相对髋的竖直位置 (cm)

    def push(self, pixels, t, thigh_cm=43.0):
        if any(j not in pixels for j in JID.values()):
            return
        hip = np.asarray(pixels[4], float)
        u = float(np.hypot(*(np.asarray(pixels[5], float) - hip)))
        if u < 10:
            return
        self.u = u if self.u is None else 0.9 * self.u + 0.1 * u
        k = thigh_cm / self.u                            # cm / px
        rel = {j: tuple((np.asarray(pixels[j], float) - hip) / self.u)
               for j in (5, 6, 7, 8)}
        self.t, self.rel = t, rel
        if self.hip_ref is None:
            self.hip_ref = hip[1]
        self.bob = float((hip[1] - self.hip_ref) * k)
        self.sh_dy = float((pixels[3][1] - hip[1]) * k) if 3 in pixels else None
        self.hand_dy = float((pixels[1][1] - hip[1]) * k) if 1 in pixels else None
        key = np.array([pixels[j] for j in (4, 5, 6, 7, 8)], float).ravel()
        if self.last_key is not None and np.abs(key - self.last_key).max() < 0.6:
            return                                # 重复帧: 只刷新时间, 不更新统计
        self.last_key = key
        self.hip_ref = 0.97 * self.hip_ref + 0.03 * hip[1]
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
                 sway_cm=1.2, stale_s=0.6, simulate=True,
                 hip_sway_cm=1.5, hip_tilt_deg=3.0,
                 shoulder_w_cm=38.0, shoulder_follow=0.8, hand_w_cm=42.0):
        self.thigh_cm, self.hip_w_cm, self.foot_w_cm = thigh_cm, hip_w_cm, foot_w_cm
        self.sway_cm, self.stale_s, self.simulate = sway_cm, stale_s, simulate
        self.hip_sway_cm, self.hip_tilt_deg = hip_sway_cm, hip_tilt_deg
        self.shoulder_w_cm, self.shoulder_follow = shoulder_w_cm, shoulder_follow
        self.hand_w_cm = hand_w_cm
        self.lateral_cm = {}                      # 膝: 外部真实横向偏摆 {'L': cm, 'R': cm}
        self.hip_lateral_cm = None                # 骨盆: 外部真实左右平移 (cm)
        self.shoulder_lateral_cm = None           # 肩: 外部真实左右平移 (cm)
        self.side = None
        self.legs = {"L": _Leg(), "R": _Leg()}
        self._last_t = -1e9
        self._amp = 5.0                           # 两腿伸展差的近期幅度 (cm)

    def reset(self):
        for leg in self.legs.values():
            leg.reset()
        self._last_t = -1e9
        self._amp = 5.0

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
        self.legs[side].push({int(j): p for j, p in pixels.items()}, t, self.thigh_cm)

    def other_leg_px(self, hip_px, side=None):
        """另一条腿在侧视图里的估计位置 {5,6,7,8: (x, y)} (像素), 供虚拟骨骼画"远侧腿"。
        近侧腿在"相位 +180°"时的姿态; 模板覆盖不足时返回 None"""
        leg = self.legs.get(side or self.side)
        if leg is None or leg.phase is None or leg.u is None or hip_px is None:
            return None
        rel = leg.template((leg.phase + math.pi) % (2 * math.pi))
        if rel is None:
            return None
        return {j: (hip_px[0] + rel[j][0] * leg.u, hip_px[1] + rel[j][1] * leg.u)
                for j in (5, 6, 7, 8)}

    # ── 生成正面图几何 ──
    def front_state(self, now):
        fresh = {s: (leg.t is not None and 0 <= now - leg.t <= self.stale_s
                     and leg.rel is not None)
                 for s, leg in self.legs.items()}
        out = {"geom": {"hip_w": self.hip_w_cm, "foot_w": self.foot_w_cm},
               "simulated": self.simulate, "L": None, "R": None,
               "pelvis": None, "shoulder": None, "hands": None}
        sel = {}
        for s in ("L", "R"):
            other = "R" if s == "L" else "L"
            if fresh[s]:
                sel[s] = (self.legs[s].rel, self.legs[s].phase, "measured")
            elif fresh[other] and self.legs[other].phase is not None:
                phase = (self.legs[other].phase + math.pi) % (2 * math.pi)
                rel = self.legs[other].template(phase)
                if rel is not None:
                    sel[s] = (rel, phase, "estimated")
        if not sel:
            return out
        meas = [self.legs[s] for s in ("L", "R") if fresh[s]]

        def _mean(vals):
            vals = [v for v in vals if v is not None]
            return float(np.mean(vals)) if vals else None

        bob = _mean([l.bob for l in meas]) or 0.0               # 实测: 髋上下起伏
        sh_dy = _mean([l.sh_dy for l in meas])                  # 实测: 肩相对髋高度
        hand_dy = _mean([l.hand_dy for l in meas])              # 实测: 手相对髋高度

        # 平衡量: 两腿伸展差 (均来自实测/估计的竖直位置)
        balance = 0.0
        if "L" in sel and "R" in sel:
            diff = (sel["R"][0][6][1] - sel["L"][0][6][1]) * self.thigh_cm
            self._amp = max(self._amp * 0.998, abs(diff), 5.0)
            balance = float(np.clip(diff / self._amp, -1.0, 1.0))

        if self.hip_lateral_cm is not None:
            shift, mode = float(self.hip_lateral_cm), "measured"
        elif self.simulate:
            shift, mode = -balance * self.hip_sway_cm, "simulated"
        else:
            shift, mode = 0.0, "neutral"
        tilt = balance * self.hip_tilt_deg if (self.simulate and mode != "measured") else 0.0
        hw = self.hip_w_cm / 2.0
        ct, st = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
        hips = {"L": (shift + hw * ct, bob + hw * st),            # 倾斜 > 0: 画面右侧髋下沉
                "R": (shift - hw * ct, bob - hw * st)}
        out["pelvis"] = {"center": (shift, bob), "tilt_deg": tilt, "shift_cm": shift,
                         "bob_cm": bob, "mode": mode, "hip_w": self.hip_w_cm,
                         "hips": hips}
        if sh_dy is not None:
            if self.shoulder_lateral_cm is not None:
                s_shift = float(self.shoulder_lateral_cm)
            else:
                s_shift = shift * self.shoulder_follow if self.simulate else 0.0
            s_tilt = -0.6 * tilt
            sc = (s_shift, bob + sh_dy)
            sw = self.shoulder_w_cm / 2.0
            cs, ss = math.cos(math.radians(s_tilt)), math.sin(math.radians(s_tilt))
            out["shoulder"] = {"center": sc, "tilt_deg": s_tilt, "shift_cm": s_shift,
                               "width": self.shoulder_w_cm, "mode": mode,
                               "pts": {"L": (sc[0] + sw * cs, sc[1] + sw * ss),
                                       "R": (sc[0] - sw * cs, sc[1] - sw * ss)}}
        if hand_dy is not None:
            out["hands"] = {"y": bob + hand_dy, "w": self.hand_w_cm}
        for s, (rel, phase, src) in sel.items():
            out[s] = self._leg_geometry(s, rel, phase, src, hips[s], bob)
        return out

    def _leg_geometry(self, s, rel, phase, src, hip, bob):
        cm = self.thigh_cm
        sign = -1.0 if s == "R" else 1.0          # 面对骑行者: 右腿在画面左侧
        ankle_rel = max(1.0, rel[6][1] * cm)
        knee_rel = float(np.clip(rel[5][1] * cm, 0.0, ankle_rel))
        ankle = (sign * self.foot_w_cm / 2.0, bob + ankle_rel)    # 脚踩在踏板上, 左右固定
        knee_y = bob + knee_rel
        frac = float(np.clip((knee_y - hip[1]) / max(1.0, ankle[1] - hip[1]), 0.0, 1.0))
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
        knee = (neutral_x - sign * dev_cm, knee_y)               # 向中线为内翻/外翻方向
        ang = _angle(hip, knee, ankle)
        dev_deg = (180.0 - ang) * (1.0 if dev_cm >= 0 else -1.0)
        status = ("valgus" if dev_deg >= TH_DEG
                  else "varus" if dev_deg <= -TH_DEG else "neutral")
        return {"source": src, "lateral_mode": mode, "hip": hip, "knee": knee,
                "ankle": ankle, "dev_cm": dev_cm, "dev_deg": dev_deg,
                "status": status, "flex": flex, "phase": phase}
