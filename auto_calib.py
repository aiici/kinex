# -*- coding: utf-8 -*-
"""
自动标定 (自动标识 8 个点): 朝向感知版

旧版问题: 写死"手臂在画面右边、腿在左边", 且脚跟/趾骨按"x 小=脚跟"判断,
          相机换到另一侧 (骑行者朝左) 时整体镜像错乱; 直接取最亮 8 点,
          车架高光比某个反光点更亮时会把点位带乱。

新版:
    1. 在最亮的 K 个候选里枚举 8 点子集, 对"朝右 / 朝左"两种假设分别评分,
       选最像人体的一组 (骨长比例、上下顺序、前后方向、反光点大小一致)
    2. 朝向 d = +1 (朝右) / -1 (朝左) 决定谁是前方: 手臂在前, 趾骨在脚跟前方
    3. 对选中的方案做两两交换的局部优化, 修正相近点 (踝/脚跟/趾骨) 的混淆
    4. 同时给出相机侧别: 骑行者朝右 → 相机拍到其右侧 'R'; 朝左 → 'L'

用法:
    click_map = auto_assign_ids(candidates)     # {JointID: (x, y)} 或 None
    AUTO_INFO  -> {"side": "R"/"L", "facing": "right"/"left", "label": "...", "score": ...}
candidates 需要有 cx, cy, radius, brightness 属性 (CircleCandidate 即可)。
"""
import itertools
import math

try:                                        # 仓库内正常使用
    from core.models import JointID
except Exception:                           # 脱离仓库测试时的后备
    import enum

    class JointID(enum.IntEnum):
        WRIST = 1; ELBOW = 2; SHOULDER = 3; HIP = 4
        KNEE = 5; ANKLE = 6; HEEL = 7; PHALANX = 8

CALIB_ORDER = [JointID.WRIST, JointID.ELBOW, JointID.SHOULDER, JointID.HIP,
               JointID.KNEE, JointID.ANKLE, JointID.HEEL, JointID.PHALANX]

AUTO_INFO = {}
_TOP_K = 12                                 # 参与枚举的最亮候选数


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _band(r, lo, hi):
    """比值 r 落在 [lo, hi] 之外的程度"""
    return max(0.0, lo - r, r - hi)


def _layout_penalty(P, d):
    """P: {1..8: (x, y)}, d=+1 朝右 / -1 朝左。越小越像真实人体布置"""
    th = _d(P[4], P[5])                     # 大腿长度 (髋-膝), 作为尺度
    if th < 15:
        return 1e6
    pen = 0.0
    # ── 骨长比例 ──
    pen += 3.0 * _band(_d(P[4], P[5]) / max(_d(P[5], P[6]), 1e-6), 0.6, 1.6)   # 大腿/小腿
    pen += 2.0 * _band(_d(P[3], P[2]) / max(_d(P[2], P[1]), 1e-6), 0.5, 2.0)   # 上臂/前臂
    pen += 2.0 * _band(_d(P[3], P[4]) / th, 0.6, 2.2)                           # 躯干/大腿
    for a, b in ((6, 7), (6, 8), (7, 8)):                                      # 足部骨段
        pen += 2.0 * _band(_d(P[a], P[b]) / th, 0.10, 0.85)
    # ── 上下顺序 (图像 y 向下为正), 以大腿长归一 ──
    for up, low in ((3, 2), (2, 1), (4, 5), (5, 6), (6, 7), (6, 8), (3, 4)):
        pen += 1.5 * max(0.0, (P[up][1] - P[low][1]) / th) \
            if up != 3 or low != 4 else 0.5 * max(0.0, (P[up][1] - P[low][1]) / th)
    # ── 前后方向 ──
    for back, front in ((3, 1), (7, 8), (4, 1), (3, 2), (2, 1)):               # 肩→腕、跟→趾、髋→腕、肩→肘→腕
        pen += 2.0 * max(0.0, -d * (P[front][0] - P[back][0]) / th)
    # ── 肘角: 手臂不会在肘部折成锐角 (排除"肩-肘-腕"来回折返的错误组合) ──
    v1 = (P[3][0] - P[2][0], P[3][1] - P[2][1])
    v2 = (P[1][0] - P[2][0], P[1][1] - P[2][1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 > 1e-6 and n2 > 1e-6:
        cosv = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        pen += 1.5 * max(0.0, (100.0 - math.degrees(math.acos(cosv))) / 60.0)
    pen += 0.6 * max(0.0, -d * (P[5][0] - P[4][0]) / th)                       # 膝在髋前方 (弱)
    return pen


def _heuristic_assign(pts, d):
    """pts: 8 个 (x, y). 按朝向 d 把前方 3 个当手臂, 其余当腿, 返回 {jid: (x, y)}"""
    idx = sorted(range(8), key=lambda i: -d * pts[i][0])        # 前方在前
    arms = sorted(idx[:3], key=lambda i: pts[i][1])             # y 小→大: 肩 肘 腕
    legs = sorted(idx[3:], key=lambda i: pts[i][1])             # y 小→大: 髋 膝 + 三个足部点
    hip, knee, foot = legs[0], legs[1], legs[2:]
    ankle = min(foot, key=lambda i: pts[i][1])                  # 足部三点里最高的是踝
    rest = sorted([i for i in foot if i != ankle], key=lambda i: d * pts[i][0])
    heel, toe = rest[0], rest[1]                                # 朝向后方的是脚跟
    order = {3: arms[0], 2: arms[1], 1: arms[2], 4: hip, 5: knee,
             6: ankle, 7: heel, 8: toe}
    return {j: pts[i] for j, i in order.items()}


def _refine(P, d, rounds=3):
    """两两交换标签的局部搜索, 直到不再改善"""
    best, bp = _layout_penalty(P, d), dict(P)
    for _ in range(rounds):
        improved = False
        for a, b in itertools.combinations(range(1, 9), 2):
            Q = dict(bp)
            Q[a], Q[b] = Q[b], Q[a]
            s = _layout_penalty(Q, d)
            if s < best - 1e-9:
                best, bp, improved = s, Q, True
        if not improved:
            break
    return best, bp


def auto_assign_ids(candidates):
    """返回 {JointID: (x, y)}; 候选不足 8 个或找不到合理布置时返回 None"""
    AUTO_INFO.clear()
    if len(candidates) < 8:
        return None
    pool = sorted(candidates, key=lambda c: -c.brightness)[:_TOP_K]
    rads = sorted(c.radius for c in pool)
    med_r = rads[len(rads) // 2] or 1.0

    best = None                                         # (score, d, subset, P)
    for subset in itertools.combinations(range(len(pool)), 8):
        cs = [pool[i] for i in subset]
        pts = [(c.cx, c.cy) for c in cs]
        # 反光点大小应一致; 大小偏离中位数越多越像杂物
        r_pen = 1.5 * sum(abs(c.radius - med_r) / med_r for c in cs) / 8.0
        for d in (1, -1):
            P = _heuristic_assign(pts, d)
            s = _layout_penalty(P, d) + r_pen
            if best is None or s < best[0]:
                best = (s, d, subset, P)
    if best is None:
        return None
    s, d, subset, P = best
    s, P = _refine(P, d)
    facing = "right" if d > 0 else "left"
    side = "R" if d > 0 else "L"
    AUTO_INFO.update(side=side, facing=facing, score=round(s, 2),
                     label=f"{'右' if side == 'R' else '左'}侧相机, 骑行者朝{'右' if d > 0 else '左'}")
    return {CALIB_ORDER[j - 1]: (float(P[j][0]), float(P[j][1])) for j in range(1, 9)}