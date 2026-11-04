# -*- coding: utf-8 -*-
# ==================== tracking_engine.py 追踪引擎 (无 Qt 依赖) ====================
#
# 高速运动修复版, 相对旧版的改动:
#   1. MotionPredictor: α-β → α-β-γ (恒加速度, 衰减记忆增益), 能跟上蹬踏的向心加速度;
#      异常门放宽, 且连续被拒 max_reject 帧后强制重新锁定 (消除"拒绝→减速→更落后"死循环);
#      丢失衰减不再连乘; dt 上限 4.0
#   2. 标定约束: 仅丢失>=3 帧时才把预测拉回标定点; 活动范围由实测自学习 (有上限)
#   3. 删除"左右侧"惩罚 (侧视下前后摆动会穿过身体中线, 反而误伤正确候选)
#   4. _assign_cost 把漏检计入代价, 足部共模修正在高速时不再被误拒
#   5. 候选提取随速度自适应 (结构元变大, 圆度/对比度/半径放宽) 对付运动模糊
#   6. 关联层不再二次收紧异常门 (之前同一个测量被卡两次)

import numpy as np
import cv2

BONE_PAIRS = ((4, 5), (5, 6), (6, 7), (6, 8), (7, 8))

JOINT_CHAIN = {
    "upper": (1, 2, 3),
    "lower": (4, 5, 6),
    "foot":  (6, 7, 8),
}

DT_MIN, DT_MAX = 0.25, 4.0


class MotionPredictor:
    """恒加速度 α-β-γ 滤波器 (衰减记忆形式), 每个关节一个。

    接口兼容: seed/reseed/predict/speed/velocity/age/lost_age/update/reset
    """

    def __init__(self, theta=0.5, damping=0.92, max_speed=120.0,
                 max_accel=60.0, outlier_gate=6.0, max_reject=2):
        self.theta = float(theta)
        self.damping = float(damping)
        self.max_speed = float(max_speed)
        self.max_accel = float(max_accel)
        self.outlier_gate = float(outlier_gate)
        self.max_reject = int(max_reject)
        # jid -> [x, y, vx, vy, ax, ay, age, lost, reject]
        self._s = {}

    def reset(self):
        self._s = {}

    def seed(self, positions):
        self._s = {int(j): [float(p[0]), float(p[1]), 0.0, 0.0, 0.0, 0.0, 0, 0, 0]
                   for j, p in positions.items()}

    def reseed(self, jid, x, y):
        self._s[int(jid)] = [float(x), float(y), 0.0, 0.0, 0.0, 0.0, 0, 0, 0]

    def predict(self, jid, dt=1.0):
        s = self._s.get(int(jid))
        if s is None:
            return None
        dt = float(np.clip(dt, DT_MIN, DT_MAX))
        return (s[0] + s[2] * dt + 0.5 * s[4] * dt * dt,
                s[1] + s[3] * dt + 0.5 * s[5] * dt * dt)

    def speed(self, jid):
        s = self._s.get(int(jid))
        return 0.0 if s is None else float(np.hypot(s[2], s[3]))

    def velocity(self, jid):
        s = self._s.get(int(jid))
        return (0.0, 0.0) if s is None else (float(s[2]), float(s[3]))

    def age(self, jid):
        s = self._s.get(int(jid))
        return 0 if s is None else int(s[6])

    def lost_age(self, jid):
        s = self._s.get(int(jid))
        return 0 if s is None else int(s[7])

    def _gains(self, spd):
        # 速度越快 θ 越小 → 增益越大 → 响应越快
        th = float(np.clip(self.theta - 0.004 * spd, 0.30, self.theta))
        return (1.0 - th ** 3,
                1.5 * (1.0 - th) ** 2 * (1.0 + th),
                0.5 * (1.0 - th) ** 3)

    @staticmethod
    def _limit(vx, vy, lim):
        n = float(np.hypot(vx, vy))
        if n > lim > 0:
            k = lim / n
            return vx * k, vy * k
        return vx, vy

    def update(self, measured, dt=1.0):
        dt = float(np.clip(dt, DT_MIN, DT_MAX))
        meas = {int(j): (float(p[0]), float(p[1])) for j, p in measured.items()}

        for j, s in self._s.items():
            x, y, vx, vy, ax, ay, age, lost, rej = s
            px = x + vx * dt + 0.5 * ax * dt * dt
            py = y + vy * dt + 0.5 * ay * dt * dt
            pvx, pvy = vx + ax * dt, vy + ay * dt
            z = meas.get(j)

            if z is None:
                lost += 1
                s[0], s[1] = px, py
                s[2], s[3] = pvx * self.damping, pvy * self.damping  # 固定系数, 不连乘
                s[4], s[5] = ax * 0.5, ay * 0.5
                if lost >= 12:
                    s[2] = s[3] = s[4] = s[5] = 0.0
                s[7], s[8] = lost, 0
                continue

            rx, ry = z[0] - px, z[1] - py
            resid = float(np.hypot(rx, ry))
            spd = float(np.hypot(pvx, pvy))
            gate = max(25.0, self.outlier_gate * (3.0 + 0.3 * spd))

            if resid > gate:
                if rej < self.max_reject:
                    s[0], s[1], s[2], s[3] = px, py, pvx, pvy
                    s[7], s[8] = lost + 1, rej + 1
                    continue
                # 连续被拒仍然出现 → 当作真实运动, 重新锁定
                s[0], s[1] = z
                s[2], s[3] = self._limit(pvx + 0.5 * rx / dt,
                                         pvy + 0.5 * ry / dt, self.max_speed)
                s[4] = s[5] = 0.0
                s[6], s[7], s[8] = age + 1, 0, 0
                continue

            a_, b_, g_ = self._gains(spd)
            nvx = pvx + b_ * rx / dt
            nvy = pvy + b_ * ry / dt
            nax = ax + 2.0 * g_ * rx / (dt * dt)
            nay = ay + 2.0 * g_ * ry / (dt * dt)
            nvx, nvy = self._limit(nvx, nvy, self.max_speed)
            nax, nay = self._limit(nax, nay, self.max_accel)

            s[0], s[1] = px + a_ * rx, py + a_ * ry
            s[2], s[3], s[4], s[5] = nvx, nvy, nax, nay
            s[6], s[7], s[8] = age + 1, 0, 0

        for j, z in meas.items():
            if j not in self._s:
                self._s[j] = [z[0], z[1], 0.0, 0.0, 0.0, 0.0, 1, 0, 0]


class Blob:
    """一个候选反光点"""
    __slots__ = ("x", "y", "r", "area", "circ", "contrast")

    def __init__(self, x, y, r, area, circ, contrast):
        self.x, self.y, self.r = x, y, r
        self.area, self.circ, self.contrast = area, circ, contrast


def _merge_rects(rects):
    rects = [list(r) for r in rects]
    changed = True
    while changed:
        changed = False
        out = []
        while rects:
            a = rects.pop()
            i = 0
            while i < len(rects):
                b = rects[i]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    a = [min(a[0], b[0]), min(a[1], b[1]),
                         max(a[2], b[2]), max(a[3], b[3])]
                    rects.pop(i)
                    changed = True
                    i = 0
                else:
                    i += 1
            out.append(a)
        rects = out
    return [tuple(r) for r in rects]


def estimate_marker_radius(gray, positions, half=20):
    h, w = gray.shape[:2]
    rs = []
    for (cx, cy) in positions:
        x1, y1 = max(0, int(cx) - half), max(0, int(cy) - half)
        x2, y2 = min(w, int(cx) + half + 1), min(h, int(cy) + half + 1)
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue
        win = gray[y1:y2, x1:x2]
        lo, hi = float(np.percentile(win, 20)), float(win.max())
        if hi - lo < 15:
            continue
        _, bw = cv2.threshold(win, int(lo + 0.6 * (hi - lo)), 255,
                              cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        best, best_d = None, 1e9
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 3:
                continue
            pt = cv2.pointPolygonTest(
                cnt, (float(cx - x1), float(cy - y1)), True)
            d = max(0.0, -pt)
            if d < best_d:
                best, best_d = area, d
        if best is not None and best_d <= 6:
            rs.append(float(np.sqrt(best / np.pi)) + 0.5)
    return float(np.median(rs)) if rs else 0.0


class MarkerAssociator:
    """把候选反光点分配给 8 个关节 (代价 + 骨长约束的精确搜索)"""
    MISS_COST = 6.0
    TOP_K = 4
    NODE_BUDGET = 4000
    FOOT = (6, 7, 8)
    CLUTTER_TOL = 3.5   # 静止杂物位置容差 (px): 车架高光质心会随光照漂移几像素

    def __init__(self, cfg):
        self.cfg = cfg
        self.bones = {}
        self.r_ref = 0.0
        self.contrast_ref = 0.0
        self.clutter = []
        self._calib_pos = {}
        self._bw_scale = 1.0     # 骨长惩罚倍率 (宽搜索时加大)
        self.jumped = set()      # 本帧经宽搜索接受、发生大位移的关节
        self._max_dev = {}      # 各关节实测到过的最大偏离标定位置 (px), 自学习活动范围
        self.debug = {}
        self._se_cache = {}

    def reset(self):
        self.bones = {}
        self.r_ref = 0.0
        self.contrast_ref = 0.0
        self.clutter = []
        self._calib_pos = {}
        self._max_dev = {}
        self.debug = {}

    @staticmethod
    def _base_limit(jid):
        if jid <= 4:
            return 80
        return 230          # 下肢 (膝/踝/脚跟/趾骨) 蹬踏幅度大, 且可能整帧跳变

    def _calib_limit(self, jid):
        """候选离标定位置的最大允许距离 (px); 随实测活动范围自动扩展, 上限 2 倍基础值"""
        base = self._base_limit(jid)
        seen = self._max_dev.get(jid, 0.0)
        return float(min(max(base, 1.3 * seen + 20.0), 2.0 * base))

    def set_calibration(self, calib_pos, calib_radius=0.0, gray=None):
        self._calib_pos = {int(j): (float(p[0]), float(p[1]))
                           for j, p in calib_pos.items()}
        self._max_dev = {}
        self.bones = {}
        for a, b in BONE_PAIRS:
            if a in calib_pos and b in calib_pos:
                self.bones[(a, b)] = float(np.hypot(
                    calib_pos[a][0] - calib_pos[b][0],
                    calib_pos[a][1] - calib_pos[b][1]))
        r = 0.0
        if gray is not None:
            r = estimate_marker_radius(gray, list(calib_pos.values()))
        self.r_ref = r if r > 0 else float(calib_radius)
        self.contrast_ref = 0.0
        self.clutter = []
        if gray is None:
            return
        H, W = gray.shape[:2]
        pts = list(calib_pos.values())
        rects = [(max(0, int(x) - 30), max(0, int(y) - 30),
                  min(W, int(x) + 31), min(H, int(y) + 31)) for x, y in pts]
        near = self.extract_blobs(gray, _merge_rects(rects))
        cs = []
        for (x, y) in pts:
            cand = [b.contrast for b in near
                    if np.hypot(b.x - x, b.y - y) <= max(6.0, self.r_ref + 3.0)]
            if cand:
                cs.append(max(cand))
        if cs:
            self.contrast_ref = float(np.median(cs))
        clear = max(8.0, 2.0 * self.r_ref)
        allb = self.extract_blobs(gray, [(0, 0, W, H)])
        cref = self.contrast_ref
        for b in allb:
            if cref > 0 and b.contrast < 0.25 * cref:
                continue
            if all(np.hypot(b.x - x, b.y - y) > clear for x, y in pts):
                self.clutter.append((b.x, b.y))

    # ─── 候选提取 ─────────────────────────────────────

    def _structuring_element(self, blur=0.0):
        c = self.cfg
        r = (self.r_ref if self.r_ref >= 2
             else float(np.clip((c.min_radius + c.max_radius) / 3.0, 3, 8)))
        k = int(np.clip(int(round(1.8 * 2 * r)) | 1, 9, 45))
        # 运动模糊: 反光点被拉成条纹, 结构元随之变大, 否则条纹被当背景减掉
        k = int(np.clip((k + int(0.6 * blur)) | 1, 9, 61))
        se = self._se_cache.get(k)
        if se is None:
            se = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            self._se_cache[k] = se
        return se, k

    def extract_blobs(self, gray, rects, blur=0.0):
        """在给定矩形窗内提取候选反光点; blur=当前最大运动速度(px/帧)"""
        c = self.cfg
        H, W = gray.shape[:2]
        se, k = self._structuring_element(blur)
        pad = k // 2 + 1
        ref_r = max(2.0, self.r_ref if self.r_ref > 0 else c.min_radius)
        floor = max(6, int(c.tophat_floor * (0.6 if ref_r <= 6.0 else 0.8)))
        r_max = c.max_radius * (1.6 if ref_r <= 6.0 else 1.3)
        r_max *= 1.0 + min(0.5, blur / 80.0)                      # 条纹半径略放宽
        if self.r_ref >= 2:
            # 真实反光点半径不会比标定半径大太多; 更大的圆是背景纹理
            r_max = min(r_max, max(2.5 * self.r_ref, self.r_ref + 8.0))
        circ_min = c.track_min_circularity * (1.0 - min(0.3, blur / 100.0))
        blobs = []
        for (x1, y1, x2, y2) in rects:
            X1, Y1 = max(0, x1 - pad), max(0, y1 - pad)
            X2, Y2 = min(W, x2 + pad), min(H, y2 + pad)
            sub = cv2.GaussianBlur(gray[Y1:Y2, X1:X2], (3, 3), 0)
            img = (cv2.morphologyEx(sub, cv2.MORPH_TOPHAT, se)
                   if c.use_tophat else sub)
            ox, oy = x1 - X1, y1 - Y1
            img = img[oy:oy + (y2 - y1), ox:ox + (x2 - x1)]
            if img.size == 0:
                continue
            peak = int(img.max())
            if peak < floor:
                continue
            for frac in (0.45, 0.22, 0.10):
                thr = max(floor, int(peak * frac))
                _, bw = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)
                cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    area = cv2.contourArea(cnt)
                    if area < 3:
                        continue
                    (lx, ly), lr = cv2.minEnclosingCircle(cnt)
                    if not (2.0 <= lr <= r_max):
                        continue
                    per = cv2.arcLength(cnt, True)
                    if per <= 0:
                        continue
                    circ = min(1.0, 4.0 * np.pi * area / (per * per))
                    if circ < circ_min:
                        continue
                    gx, gy = lx + x1, ly + y1
                    merge_r = max(4.0, 0.8 * lr)
                    if any(np.hypot(gx - b.x, gy - b.y) < merge_r
                           for b in blobs):
                        continue
                    bx, by, bw_, bh_ = cv2.boundingRect(cnt)
                    contrast = float(img[by:by + bh_, bx:bx + bw_].max())
                    blobs.append(Blob(gx, gy, float(lr), float(area),
                                      float(circ), contrast))
        return blobs

    # ─── 关联 ─────────────────────────────────────────

    def associate(self, gray, motion, tracked, last_pos, lost, dt=1.0):
        """返回 {jid_int: (x, y)} 本帧真正匹配上的关节"""
        c = self.cfg
        H, W = gray.shape[:2]
        max_gate = max(40.0, min(W, H) / 3.0)

        st = {}
        rects = []
        for j in tracked:
            pred = motion.predict(j, dt) or last_pos.get(j)
            if pred is None:
                continue
            vx, vy = motion.velocity(j)
            lost_j = lost.get(j, 0)
            spd = motion.speed(j)
            base = c.track_max_move * (c.lower_gate_scale if j >= 5 else 1.0)
            gate = min(base * (1.0 + 0.35 * lost_j) + 0.5 * spd,
                       max(max_gate, base))
            # 跟踪稳定的关节收紧搜索门, 减少被相邻垃圾候选抢走
            if lost_j == 0 and motion.age(j) >= 3:
                gate = min(gate, max(35.0, 2.0 * spd + 25.0))
            cx, cy = self._calib_pos.get(j, (None, None))
            limit = self._calib_limit(j)
            if cx is not None:
                # 只有丢失>=3 帧才把 coast 出去的预测拉回, 正常跟踪时不干预
                if lost_j >= 3:
                    d_calib = float(np.hypot(pred[0] - cx, pred[1] - cy))
                    if d_calib > limit * 0.8:
                        pred = (float(cx), float(cy))
                gate = min(gate, float(limit))
            sigma = (4.0 + 0.3 * spd + 3.0 * lost_j
                     + max(0, 3 - motion.age(j)) * 5.0)
            sigma = min(max(sigma, 6.0), gate / 2.0)
            st[j] = (float(pred[0]), float(pred[1]), gate, sigma,
                     float(vx), float(vy))
            win = max(float(c.roi_radius), gate + c.max_radius + 4.0)
            rx1, ry1 = max(0, int(pred[0] - win)), max(0, int(pred[1] - win))
            rx2, ry2 = min(W, int(pred[0] + win) + 1), min(H, int(pred[1] + win) + 1)
            if rx2 > rx1 and ry2 > ry1:
                rects.append((rx1, ry1, rx2, ry2))

        blur = max((motion.speed(j) for j in tracked), default=0.0)
        blobs = (self.extract_blobs(gray, _merge_rects(rects), blur)
                 if rects else [])
        blobs = self._filter_blobs(blobs, blur)
        if len(blobs) > 40:                       # 只留对比度最高的 40 个
            blobs.sort(key=lambda b: -b.contrast)
            blobs = blobs[:40]

        assign = self._assign(st, blobs)
        assign = self._protect_identity(assign, st, blobs)

        cost1 = self._assign_cost(assign, st, blobs)
        st2 = self._refine_foot(st, blobs, assign)
        if st2 is not None:
            assign2 = self._assign(st2, blobs)
            assign2 = self._protect_identity(assign2, st2, blobs)
            cost2 = self._assign_cost(assign2, st2, blobs)
            if cost2 <= cost1 * 1.15:
                st = st2
                assign = assign2

        # ── 宽搜索: 正常匹配的下肢不合理 (漏检/对比度低/骨长不符) 时启用,
        #    用于画面跳变、蹬踏位移超出预测门的场景 ──
        self.jumped = set()
        st_norm = st
        if self.bones and self._calib_pos:
            s_norm = self._lower_score(assign, blobs)
            if s_norm > 2.5:
                res = self._wide_stage(gray, st, blobs, last_pos)
                if res is not None:
                    st_w, blobs_w, assign_w = res
                    s_w = self._lower_score(assign_w, blobs_w)
                    if s_w + 0.5 < s_norm:
                        for j in (5, 6, 7, 8):
                            bi = assign_w.get(j)
                            if bi is None or j not in st_norm:
                                continue
                            b = blobs_w[bi]
                            if np.hypot(b.x - st_norm[j][0],
                                        b.y - st_norm[j][1]) > st_norm[j][2]:
                                self.jumped.add(j)
                        st, blobs, assign = st_w, blobs_w, assign_w

        # 直接用 gate 判定, 不再二次收紧 (MotionPredictor 内部另有门)
        matched = {}
        for j, bi in assign.items():
            b = blobs[bi]
            px, py = st[j][0], st[j][1]
            if np.hypot(b.x - px, b.y - py) <= st[j][2]:
                matched[j] = (b.x, b.y)
                c0 = self._calib_pos.get(j)
                if c0 is not None:
                    dev = float(np.hypot(b.x - c0[0], b.y - c0[1]))
                    if dev > self._max_dev.get(j, 0.0):
                        self._max_dev[j] = dev

        for j in self.jumped:           # 大位移: 滤波器直接重置到新位置
            if j in matched:
                motion.reseed(j, matched[j][0], matched[j][1])

        self.debug = {
            "blobs": [(b.x, b.y, b.r) for b in blobs],
            "pred": {j: (v[0], v[1]) for j, v in st.items()},
            "gate": {j: v[2] for j, v in st.items()},
            "clutter": list(self.clutter),
            "blur": blur,
        }
        return matched

    def _lower_score(self, assign, blobs):
        """下肢 (4~8) 匹配合理性评分, 越小越好: 漏检 + 弱对比度/异常大小 + 骨长偏差"""
        cref = self.contrast_ref
        sc = 0.0
        for j in (5, 6, 7, 8):
            bi = assign.get(j)
            if bi is None:
                sc += 4.0
                continue
            b = blobs[bi]
            # 真实反光点对比度≈标定值 (≥0.93·cref) 且半径≥0.7·r_ref;
            # 暗/小的碎片明显扣分, 逼迫宽搜索去找真正的强点
            weak = False
            if cref > 0:
                ratio = b.contrast / cref
                sc += 10.0 * max(0.0, 0.93 - ratio)
                weak = ratio < 0.93
            if self.r_ref >= 2:
                sc += 2.0 * min(1.0, abs(b.r - self.r_ref) / self.r_ref)
                weak = weak or b.r < 0.7 * self.r_ref
            if weak:
                sc += 2.0
        for (a, b), L in self.bones.items():
            if a < 4 or a not in assign or b not in assign:
                continue
            ia, ib = assign[a], assign[b]
            if ia is None or ib is None:
                continue
            d = float(np.hypot(blobs[ia].x - blobs[ib].x,
                               blobs[ia].y - blobs[ib].y))
            tol = 0.35 * L + 12.0 if L <= 60 else 0.15 * L + 2.0
            sc += min((abs(d - L) / tol) ** 2, 9.0)
        return sc

    def _wide_stage(self, gray, st, blobs, last_pos):
        """下肢宽搜索: 在整个下肢活动区提强候选, 搜索门放大, 靠对比度+骨长选"""
        lower = [j for j in (5, 6, 7, 8) if j in st and j in self._calib_pos]
        if not lower:
            return None
        H, W = gray.shape[:2]
        R = 240
        xs = [self._calib_pos[j][0] for j in lower]
        ys = [self._calib_pos[j][1] for j in lower]
        rect = (max(0, int(min(xs) - R)), max(0, int(min(ys) - R)),
                min(W, int(max(xs) + R)), min(H, int(max(ys) + R)))
        extra = self._filter_blobs(self.extract_blobs(gray, [rect]))
        allb = list(blobs)
        for e in extra:
            if not any(np.hypot(e.x - b.x, e.y - b.y) < 3.0 for b in allb):
                allb.append(e)
        cref = self.contrast_ref
        if cref > 0:
            allb = [b for b in allb if b.contrast >= 0.7 * cref]
        if len(allb) > 40:
            allb.sort(key=lambda b: -b.contrast)
            allb = allb[:40]
        st_w = dict(st)
        for j in lower:
            cx, cy = last_pos.get(j) or (st[j][0], st[j][1])
            st_w[j] = (float(cx), float(cy), 240.0, 90.0, 0.0, 0.0)
        self._bw_scale = 2.0
        try:
            assign_w = self._assign(st_w, allb, wide=True)
        finally:
            self._bw_scale = 1.0
        return st_w, allb, assign_w

    def _filter_blobs(self, blobs, blur=0.0):
        cref = self.contrast_ref
        tol = self.CLUTTER_TOL
        rel = 0.25 * max(0.6, 1.0 - blur / 80.0)   # 条纹能量分散, 对比度下限随速度降低
        keep = []
        for b in blobs:
            if cref > 0 and b.contrast < rel * cref:
                continue
            if any(abs(b.x - cx) <= tol and abs(b.y - cy) <= tol
                   for cx, cy in self.clutter):
                continue
            keep.append(b)
        return keep

    def _assign(self, st, blobs, wide=False):
        cref = self.contrast_ref
        options = {}
        for j, (px, py, gate, sigma, vx, vy) in st.items():
            cx, cy = self._calib_pos.get(j, (None, None))
            limit = self._calib_limit(j)
            spd = float(np.hypot(vx, vy))
            lst = []
            for bi, b in enumerate(blobs):
                d = float(np.hypot(b.x - px, b.y - py))
                if d > gate:
                    continue
                if (cx is not None and not (wide and j >= 5)
                        and np.hypot(b.x - cx, b.y - cy) > limit):
                    continue
                size = (min(1.5, abs(b.r - self.r_ref) / self.r_ref)
                        if self.r_ref >= 2 else 0.0)
                con = (min(1.0, max(0.0, 1.0 - b.contrast / cref
                                    - (0.05 if wide else 0.0)))
                       if cref > 0 else 0.0)
                cost = (0.5 * (d / sigma) ** 2 + 0.25 * (1.0 - b.circ)
                        + 0.3 * size + (2.5 if wide else 0.8) * con)
                # 宽搜索: 暗/小的碎片大幅扣分 (真实反光点对比度≈标定值、半径≈标定值)
                if wide and cref > 0 and (
                        b.contrast < 0.93 * cref or
                        (self.r_ref >= 2 and b.r < 0.7 * self.r_ref)):
                    cost += 3.0
                # 高速时轻微惩罚反方向候选
                if spd > 5.0 and d > 1.0:
                    dot = vx * (b.x - px) + vy * (b.y - py)
                    if dot < 0:
                        cost += 0.3 * min(1.5, -dot / (spd * d))
                # (已删除: 左右侧中线惩罚, 侧视下会误伤正确候选)
                if cost < self.MISS_COST:
                    lst.append((cost, bi))
            lst.sort()
            options[j] = lst[:self.TOP_K]
        return self._solve(list(st.keys()), options, blobs)

    def _refine_foot(self, st, blobs, assign):
        inn = {}
        for j in self.FOOT:
            if j in assign and j in st:
                b = blobs[assign[j]]
                inn[j] = (b.x - st[j][0], b.y - st[j][1])
        if len(inn) < 2:
            return None

        mx = float(np.median([v[0] for v in inn.values()]))
        my = float(np.median([v[1] for v in inn.values()]))

        foot_bones = [L for (a, b), L in self.bones.items()
                      if a in self.FOOT and b in self.FOOT]
        ref_len = float(np.median(foot_bones)) if foot_bones else 40.0
        tol = max(6.0, 0.25 * ref_len + 2.0 * self.r_ref)

        agree = [j for j, (ix, iy) in inn.items()
                 if np.hypot(ix - mx, iy - my) <= tol]
        if len(agree) < 2:
            return None

        res_angles = []
        for j in agree:
            ix, iy = inn[j]
            if np.hypot(ix, iy) > 3.0:
                res_angles.append(np.arctan2(iy, ix))
        if len(res_angles) >= 2:
            for i in range(len(res_angles)):
                for k in range(i + 1, len(res_angles)):
                    diff = abs(res_angles[i] - res_angles[k])
                    diff = min(diff, 2 * np.pi - diff)
                    if diff > np.pi / 2:
                        return None

        dx = float(np.mean([inn[j][0] for j in agree]))
        dy = float(np.mean([inn[j][1] for j in agree]))
        if np.hypot(dx, dy) < 2.0:
            return None

        new = dict(st)
        for j in self.FOOT:
            if j in st:
                px, py, gate, sigma, vx, vy = st[j]
                new[j] = (px + dx, py + dy, gate, max(6.0, 0.5 * sigma), vx, vy)

        for (a, b), L in self.bones.items():
            if a in self.FOOT and b in self.FOOT and a in new and b in new:
                d_new = float(np.hypot(new[a][0] - new[b][0],
                                       new[a][1] - new[b][1]))
                tol_b = 0.35 * L + 12.0 if L <= 60 else 0.15 * L + 2.0
                if abs(d_new - L) > tol_b * 1.5:
                    return None
        return new

    def _protect_identity(self, assign, st, blobs):
        if not assign or not self.bones:
            return assign
        result = dict(assign)
        for (a, b), L in self.bones.items():
            if a not in result or b not in result:
                continue
            ba, bb = result[a], result[b]
            if ba is None or bb is None:
                continue
            pa, pb = st.get(a), st.get(b)
            if pa is None or pb is None:
                continue
            oa, ob = blobs[ba], blobs[bb]
            cur = (np.hypot(oa.x - pa[0], oa.y - pa[1]) +
                   np.hypot(ob.x - pb[0], ob.y - pb[1]) +
                   2.0 * abs(np.hypot(oa.x - ob.x, oa.y - ob.y) - L))
            sw = (np.hypot(ob.x - pa[0], ob.y - pa[1]) +
                  np.hypot(oa.x - pb[0], oa.y - pb[1]) +
                  2.0 * abs(np.hypot(ob.x - oa.x, ob.y - oa.y) - L))
            if sw < cur - 8.0:
                result[a], result[b] = bb, ba
        return result

    def _assign_cost(self, assign, st, blobs):
        """方案总代价; 漏检按半个 gate 计, 否则匹配少的方案总显得更便宜"""
        if not st:
            return float("inf")
        total = 0.0
        for j in st:
            bi = assign.get(j)
            if bi is None:
                total += 0.5 * st[j][2]
            else:
                b = blobs[bi]
                total += float(np.hypot(b.x - st[j][0], b.y - st[j][1]))
        for (a, b), L in self.bones.items():
            ba, bb = assign.get(a), assign.get(b)
            if ba is not None and bb is not None:
                d = float(np.hypot(blobs[ba].x - blobs[bb].x,
                                   blobs[ba].y - blobs[bb].y))
                total += abs(d - L) * 2.0
        return total

    # ─── 带骨长约束的分配 ─────────────────────────────

    def _solve(self, joints, options, blobs):
        parent = {j: j for j in joints}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        owner = {}
        for j in joints:
            for _, bi in options.get(j, []):
                if bi in owner:
                    union(j, owner[bi])
                else:
                    owner[bi] = j
        for (a, b) in self.bones:
            if a in parent and b in parent:
                union(a, b)

        groups = {}
        for j in joints:
            groups.setdefault(find(j), []).append(j)

        result = {}
        for grp in groups.values():
            result.update(self._solve_group(grp, options, blobs))
        return result

    def _solve_group(self, group, options, blobs):
        w = self.cfg.bone_weight * self._bw_scale
        miss = self.MISS_COST
        order = sorted(group, key=lambda j: (len(options.get(j, [])), j))
        bones_of = {j: [] for j in group}
        for (a, b), L in self.bones.items():
            if a in bones_of and b in bones_of:
                bones_of[a].append((b, L))
                bones_of[b].append((a, L))

        def bone_pen(d, L):
            tol = 0.35 * L + 12.0 if L <= 60 else 0.15 * L + 2.0
            e = abs(d - L) / tol
            return w * 0.5 * min(e * e, 9.0)

        best = [float("inf"), {}]
        nodes = [0]
        assign = {}
        used = set()

        def dfs(i, cost):
            if cost >= best[0]:
                return
            if i == len(order):
                best[0] = cost
                best[1] = dict(assign)
                return
            nodes[0] += 1
            if nodes[0] > self.NODE_BUDGET:
                return
            j = order[i]
            for cst, bi in options.get(j, []):
                if bi in used:
                    continue
                b = blobs[bi]
                add = cst
                for (o, L) in bones_of[j]:
                    ob = assign.get(o)
                    if ob is not None:
                        add += bone_pen(float(np.hypot(b.x - blobs[ob].x,
                                                       b.y - blobs[ob].y)), L)
                assign[j] = bi
                used.add(bi)
                dfs(i + 1, cost + add)
                used.discard(bi)
                del assign[j]
            assign[j] = None
            dfs(i + 1, cost + miss)
            del assign[j]

        dfs(0, 0.0)
        return {j: bi for j, bi in best[1].items() if bi is not None}
