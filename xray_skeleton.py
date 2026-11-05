# -*- coding: utf-8 -*-
"""
虚拟人体骨骼 (X 光全息风): 由 8 个追踪点实时生成, 半透明发光地叠加在画面上。

与旧"图层蒙版"的区别: 蒙版是乘法 (压暗画面, 会遮住真人); 这里是加法发光叠加,
画面亮度只会增加, 不会被挡住。

生成的结构 (均由 8 个点按人体比例推导):
    头骨+下颌+颈椎 / 脊柱(带弯曲) / 肋骨+胸骨 / 骨盆 / 肩胛
    上臂 / 前臂(桡骨+尺骨) / 手
    股骨(含股骨头) / 髌骨 / 胫骨+腓骨 / 足 (跟骨 / 跖骨扇形)
    另一条腿 (估计) 以更淡的一层绘制
关节光晕 (肘/肩/髋/膝/踝) 按角度阈值显示 蓝=偏小 绿=正常 红=偏大。

环境变量:
    KINEX_XRAY=0            关闭
    KINEX_XRAY_ALPHA=0.85   整体强度 (0~1.5)
    KINEX_XRAY_STATUS=0     关节光晕不按红绿蓝着色 (统一青色)
"""
import math
import os

import numpy as np
import cv2

ENABLED = os.environ.get("KINEX_XRAY", "1").strip() != "0"
ALPHA = float(os.environ.get("KINEX_XRAY_ALPHA", "0.85"))
STATUS = os.environ.get("KINEX_XRAY_STATUS", "1").strip() != "0"

# BGR
FILL = (88, 108, 26)           # 长骨填充 (暗青, 叠加后微亮)
FILL_BIG = (40, 52, 12)        # 颅骨/骨盆/肩胛等大块区域: 更暗, 靠轮廓表达, 避免成实心块
EDGE = (255, 255, 175)         # 骨骼轮廓 (亮青白)
CORE = (255, 255, 230)         # 关节球 / 高光
HALO = {"low": (255, 140, 0), "ok": (0, 240, 128), "high": (0, 51, 255)}
HALO_CYAN = (255, 212, 0)
AA = cv2.LINE_AA

JOINT_RANGE = {2: "elbow", 3: "shoulder", 4: "hip", 5: "knee", 6: "ankle"}


def _u(v):
    n = float(np.hypot(v[0], v[1]))
    return np.asarray(v, float) / n if n > 1e-6 else np.array([1.0, 0.0])


def _perp(v):
    return np.array([-v[1], v[0]])


def _i(p):
    return (int(round(p[0])), int(round(p[1])))


def _status(name, ang, ranges):
    if ang is None or not ranges or name not in ranges:
        return "ok"
    lo, hi = ranges[name]
    return "low" if ang < lo else ("high" if ang > hi else "ok")


def _bone(L, p, q, w0, w1, pinch=0.65, cap=1.25, fill=FILL, edge=EDGE):
    """带骨骺膨大的长骨: 两端宽、骨干收窄"""
    p, q = np.asarray(p, float), np.asarray(q, float)
    t = _u(q - p)
    n = _perp(t)
    wm = pinch * 0.5 * (w0 + w1)
    m = 0.5 * (p + q)
    poly = np.array([p + n * w0, m + n * wm, q + n * w1,
                     q - n * w1, m - n * wm, p - n * w0]).astype(np.int32)
    cv2.fillPoly(L, [poly], fill, AA)
    cv2.polylines(L, [poly], True, edge, 1, AA)
    for c, r in ((p, w0 * cap), (q, w1 * cap)):
        cv2.circle(L, _i(c), max(1, int(r)), fill, -1, AA)
        cv2.circle(L, _i(c), max(1, int(r)), edge, 1, AA)


def _bez(a, c, b, n=12):
    ts = np.linspace(0.0, 1.0, n)[:, None]
    return ((1 - ts) ** 2) * a + 2 * (1 - ts) * ts * c + (ts ** 2) * b


def _ell(L, c, axes, ang, fill=FILL, edge=EDGE, th=1):
    cv2.ellipse(L, _i(c), (max(1, int(axes[0])), max(1, int(axes[1]))),
                ang, 0, 360, fill, -1, AA)
    cv2.ellipse(L, _i(c), (max(1, int(axes[0])), max(1, int(axes[1]))),
                ang, 0, 360, edge, th, AA)


def _leg(L, hip, knee, ankle, heel, toe, u, d, fill=FILL, edge=EDGE):
    """股骨 / 髌骨 / 胫骨 / 腓骨 / 足"""
    hip, knee, ankle = (np.asarray(x, float) for x in (hip, knee, ankle))
    thigh = _u(knee - hip)
    nf = _perp(thigh)
    if nf[0] * d < 0:
        nf = -nf                                    # 朝向前方的法线
    _bone(L, hip, knee, 0.105 * u, 0.090 * u, 0.62, fill=fill, edge=edge)
    _ell(L, knee + nf * 0.115 * u, (0.075 * u, 0.095 * u),
         math.degrees(math.atan2(thigh[1], thigh[0])), fill, edge)      # 髌骨
    shank = _u(ankle - knee)
    _bone(L, knee, ankle, 0.082 * u, 0.068 * u, 0.58, fill=fill, edge=edge)
    ns = _perp(shank)
    if ns[0] * d > 0:
        ns = -ns                                    # 腓骨在后外侧
    f0, f1 = knee + shank * 0.14 * u + ns * 0.085 * u, ankle + ns * 0.045 * u
    cv2.line(L, _i(f0), _i(f1), edge, 1, AA)
    cv2.line(L, _i(f0 + ns * 0.012 * u), _i(f1 + ns * 0.012 * u), fill, 2, AA)
    if heel is not None and toe is not None:
        heel, toe = np.asarray(heel, float), np.asarray(toe, float)
        a0 = ankle + (toe - ankle) * 0.38
        cv2.fillPoly(L, [np.array([ankle, heel, a0]).astype(np.int32)], FILL_BIG, AA)
        cv2.polylines(L, [np.array([ankle, heel, a0]).astype(np.int32)], True, edge, 1, AA)
        _ell(L, heel, (0.085 * u, 0.07 * u), 0, fill, edge)             # 跟骨
        ft = _u(toe - a0)
        fn = _perp(ft)
        for k in np.linspace(-1, 1, 5):                                  # 跖骨扇形
            e = toe + fn * k * 0.065 * u - ft * abs(k) * 0.02 * u
            cv2.line(L, _i(a0), _i(e), edge, 1, AA)
            cv2.circle(L, _i(e), 2, edge, -1, AA)
    cv2.circle(L, _i(hip), int(0.125 * u), CORE, 1, AA)                 # 股骨头
    cv2.circle(L, _i(ankle), int(0.07 * u), CORE, 1, AA)


class XraySkeleton:
    def __init__(self, alpha=None, status=None):
        self.alpha = ALPHA if alpha is None else alpha
        self.status = STATUS if status is None else status
        self._u = None                                # 大腿长度 EMA (像素)

    def render(self, frame, pixels, angles=None, ranges=None, other_leg=None):
        if not ENABLED or not pixels:
            return frame
        P = {int(j): np.asarray(p, float) for j, p in pixels.items()}
        if not all(j in P for j in (3, 4, 5, 6)):
            return frame
        vis = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        h, w = vis.shape[:2]
        u_now = float(np.hypot(*(P[5] - P[4])))
        if u_now > 20:
            self._u = u_now if self._u is None else 0.92 * self._u + 0.08 * u_now
        u = self._u or 114.0
        sh, hip = P[3], P[4]
        T = _u(sh - hip)
        d = 1.0 if (P.get(1, sh)[0] - hip[0]) >= 0 else -1.0     # 朝向: 手腕在髋前方
        Nc = d * np.array([-T[1], T[0]])                          # 胸侧法线 (朝前下)

        L = np.zeros_like(vis)                                    # 近侧骨骼 (亮)
        F = np.zeros_like(vis)                                    # 远侧腿 (淡)
        pts = [P[j] for j in P]

        # ── 另一条腿 (估计): 画在淡层 ──
        if other_leg and all(j in other_leg for j in (5, 6)):
            ol = {j: np.asarray(p, float) for j, p in other_leg.items()}
            _leg(F, hip, ol[5], ol[6], ol.get(7), ol.get(8), u, d,
                 fill=(100, 120, 30), edge=(220, 220, 140))
            pts += list(ol.values())

        # ── 脊柱 (带弯曲) + 肋骨 + 胸骨 ──
        base = hip + (sh - hip) * 0.10
        neck = sh + np.array([-d * 0.12 * u, -0.05 * u])
        mid = 0.5 * (base + neck)
        ctrl = mid - Nc * 0.14 * float(np.hypot(*(neck - base)))
        spine = _bez(base, ctrl, neck, 17)
        for k in range(len(spine) - 1):
            tg = spine[k + 1] - spine[k]
            ang = math.degrees(math.atan2(tg[1], tg[0]))
            _ell(L, 0.5 * (spine[k] + spine[k + 1]), (0.05 * u, 0.034 * u), ang)
        rib_ends = []
        for t_ in np.linspace(0.40, 0.92, 8):
            i0 = int(t_ * (len(spine) - 1))
            S = spine[i0]
            tg = _u(spine[min(i0 + 1, len(spine) - 1)] - spine[max(i0 - 1, 0)])
            Nr = d * np.array([-tg[1], tg[0]])
            ln = u * (0.30 + 0.12 * math.sin(math.pi * (t_ - 0.30) / 0.70))
            E = S + Nr * ln + tg * 0.10 * ln
            C = S + Nr * ln * 0.50 - tg * 0.30 * ln
            cv2.polylines(L, [_bez(S, C, E, 10).astype(np.int32)], False, EDGE, 1, AA)
            cv2.polylines(L, [_bez(S, C, E, 10).astype(np.int32)], False, FILL, 2, AA)
            rib_ends.append(E)
        cv2.polylines(L, [np.array(rib_ends).astype(np.int32)], False, EDGE, 2, AA)
        pts += rib_ends + [neck]

        # ── 骨盆 ──
        ca = math.degrees(math.atan2(T[1], T[0]))
        _ell(L, hip + T * 0.10 * u, (0.15 * u, 0.24 * u), ca, FILL_BIG)
        _ell(L, hip - T * 0.05 * u - np.array([d * 0.04 * u, 0]), (0.08 * u, 0.11 * u), ca, FILL_BIG)

        # ── 头颈 ──
        # 头的位置随躯干前倾程度变化: 直立 → 在颈上方; 俯身 → 在肩前下方
        lean = float(np.clip((abs(T[0]) - 0.30) / 0.60, 0.0, 1.0))
        head_dir = _u((1 - lean) * np.array([0.15 * d, -1.0]) + lean * np.array([d * 0.91, 0.41]))
        hc = neck + head_dir * (0.55 + 0.21 * lean) * u
        for k in np.linspace(0.10, 0.90, 4):                       # 颈椎
            _ell(L, neck + (hc - neck) * k, (0.040 * u, 0.028 * u),
                 math.degrees(math.atan2(*(hc - neck)[::-1])))
        cv2.circle(L, _i(hc), int(0.27 * u), FILL_BIG, -1, AA)
        cv2.circle(L, _i(hc), int(0.27 * u), EDGE, 1, AA)
        jaw = np.array([hc + [d * 0.03 * u, 0.10 * u], hc + [d * 0.19 * u, 0.12 * u],
                        hc + [d * 0.30 * u, 0.22 * u], hc + [d * 0.23 * u, 0.34 * u],
                        hc + [d * 0.07 * u, 0.30 * u]])
        cv2.fillPoly(L, [jaw.astype(np.int32)], FILL_BIG, AA)
        cv2.polylines(L, [jaw.astype(np.int32)], True, EDGE, 1, AA)
        cv2.circle(L, _i(hc + [d * 0.14 * u, -0.01 * u]), max(1, int(0.04 * u)), EDGE, 1, AA)
        pts += [hc + [0, -0.3 * u], hc + [d * 0.34 * u, 0.3 * u]]

        # ── 肩胛 + 手臂 ──
        sc = np.array([sh, sh + [-d * 0.34 * u, 0.04 * u], sh + [-d * 0.24 * u, 0.26 * u]])
        cv2.fillPoly(L, [sc.astype(np.int32)], FILL_BIG, AA)
        cv2.polylines(L, [sc.astype(np.int32)], True, EDGE, 1, AA)
        if 2 in P:
            _bone(L, sh, P[2], 0.075 * u, 0.065 * u, 0.7)
            if 1 in P:
                fa = _u(P[1] - P[2])
                fn = _perp(fa)
                _bone(L, P[2], P[1] + fn * 0.02 * u, 0.034 * u, 0.028 * u, 0.8)
                _bone(L, P[2], P[1] - fn * 0.02 * u, 0.034 * u, 0.028 * u, 0.8)
                for k in np.linspace(-1, 1, 4):                     # 手
                    e = P[1] + fa * 0.27 * u + fn * k * 0.07 * u
                    cv2.line(L, _i(P[1]), _i(e), EDGE, 1, AA)
                    cv2.circle(L, _i(e), 2, EDGE, -1, AA)
                pts.append(P[1] + fa * 0.30 * u)

        # ── 近侧腿 ──
        _leg(L, hip, P[5], P[6], P.get(7), P.get(8), u, d)

        # ── 关节球 + 状态光晕 ──
        for jid, name in JOINT_RANGE.items():
            if jid not in P:
                continue
            st = _status(name, (angles or {}).get(name), ranges)
            col = HALO[st] if self.status else HALO_CYAN
            r = int(0.115 * u)
            cv2.circle(L, _i(P[jid]), r, col, 2, AA)
            cv2.circle(L, _i(P[jid]), max(2, r // 3), CORE, -1, AA)

        # ── 合成: 发光 + 加法叠加 (只增亮, 不遮挡) ──
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = int(0.5 * u) + 12
        x0, y0 = max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad)
        x1, y1 = min(w, int(max(xs)) + pad), min(h, int(max(ys)) + pad)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return vis
        sl = (slice(y0, y1), slice(x0, x1))
        layer = cv2.addWeighted(L[sl], 1.0, F[sl], 0.45, 0)
        glow = cv2.GaussianBlur(layer, (0, 0), max(3.0, 0.05 * u))
        add = cv2.addWeighted(layer, self.alpha, glow, 0.85 * self.alpha, 0)
        out = vis.copy()
        out[sl] = cv2.add(out[sl], add)
        return out


def render_xray(frame, pixels, angles=None, ranges=None, other_leg=None, _inst=[]):
    """无状态便捷入口 (内部复用一个实例)"""
    if not _inst:
        _inst.append(XraySkeleton())
    return _inst[0].render(frame, pixels, angles, ranges, other_leg)
