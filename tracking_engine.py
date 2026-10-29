# ==================== tracking_engine.py 追踪引擎 (无 Qt 依赖, 可脱离界面单独测试) ====================
#
# 逐帧流程:
#   1. MotionPredictor      自适应 α-β 滤波器给出每个关节"本帧应出现的位置"
#                           (支持实际帧间隔 dt, 动态纠正强度, 异常测量过滤, 速度限幅)
#   2. MarkerAssociator
#        a. 在预测位置附近的窗口里提取候选反光点 (局部对比度 top-hat,
#           反光点贴在亮色鞋面/衣物上也能分开)
#        b. 剔除"杂物": 标定帧里就存在的静止亮点、对比度远低于反光点的碎片
#        c. 对 (关节, 候选) 计算代价: 到预测点的距离 + 形状/大小/对比度一致性
#           + 运动方向约束 (高速时惩罚反方向候选)
#        d. 对 4~8 号点加刚性骨段长度约束, 在候选组合上做精确搜索
#           (最小总代价), 取代原来的"评分贪心 + 事后纠错"
#        e. 身份交换保护: 分配完成后对每对骨段关节做 keep-vs-swap 验证,
#           只有交换明显更优才接受, 维护运动连续性
#        f. 足部 6/7/8 是刚体: 先匹配上的点带着其余点一起修正预测, 再分配一次
#           (共模修正前检查残差方向一致性, 修正后验证骨长, 二次分配不差于第一次才接受)
#        g. 异常跳变检测: 残差过大的匹配不输出, 防止错误关联传播
#   3. 没有匹配上的关节按阻尼速度外推 (coast), 而不是原地冻结;
#      连续丢失过久停止外推, 由上层调用者 reseed 到标定位置重新捕获
#

import numpy as np
import cv2

# 刚性骨段: 侧视 2D 下长度近似恒定, 用于消除 4~8 号点之间的"身份互换"
BONE_PAIRS = ((4, 5), (5, 6), (6, 7), (6, 8), (7, 8))

# 运动链分组 (侧视单侧): 同一链内的关节在运动中可能互相混淆,
# 需要额外的身份保持约束. 足部 (6/7/8) 距离最近, 最易互换.
JOINT_CHAIN = {
    "upper": (1, 2, 3),       # 腕-肘-肩
    "lower": (4, 5, 6),       # 髋-膝-踝
    "foot":  (6, 7, 8),       # 踝-脚跟-趾骨 (刚体)
}


class MotionPredictor:
    """
    高速运动自适应 α-β 滤波器, 每个关节一个

    改进 (相比旧版恒速模型):
        1. 支持实际帧间隔 dt — 处理速度不稳定时预测位置仍准确
        2. 根据运动速度动态调整纠正强度 (高速增响应, 低速减抖动)
        3. 测量异常 (大跳变) 时不立即接受, 避免错误关联污染状态
        4. 丢失目标时阻尼外推, 连续丢失过久停止外推
        5. 速度/加速度限幅保护, 降低单帧异常造成的速度突变

    接口兼容: seed()/predict()/speed()/age()/reseed()/update() 均保留,
    新增可选 dt 参数 (默认 1.0 = 正常帧间隔) 和 lost_age() 方法.
    """

    def __init__(self, alpha=0.85, beta=0.5, damping=0.90,
                 max_speed=80.0, max_accel=35.0, outlier_gate=3.5):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.damping = float(damping)
        self.max_speed = float(max_speed)
        self.max_accel = float(max_accel)
        self.outlier_gate = float(outlier_gate)
        # jid -> [x, y, vx, vy, age, lost]
        self._s = {}

    def reset(self):
        self._s = {}

    def seed(self, positions):
        """用标定位置 / 循环起点位置初始化 (速度 0). positions: {jid: (x, y)}"""
        self._s = {int(j): [float(p[0]), float(p[1]), 0.0, 0.0, 0, 0]
                   for j, p in positions.items()}

    def reseed(self, jid, x, y):
        """重置单个关节到指定位置 (速度归零, age/lost 归零) — 丢失太久时防止 coast 跑飞"""
        self._s[int(jid)] = [float(x), float(y), 0.0, 0.0, 0, 0]

    def predict(self, jid, dt=1.0):
        """返回 dt 个帧间隔后的预测位置; dt=1.0 为正常帧间隔"""
        s = self._s.get(int(jid))
        if s is None:
            return None
        dt = float(np.clip(dt, 0.25, 3.0))
        return (s[0] + s[2] * dt, s[1] + s[3] * dt)

    def speed(self, jid):
        s = self._s.get(int(jid))
        return 0.0 if s is None else float(np.hypot(s[2], s[3]))

    def velocity(self, jid):
        """返回 (vx, vy); 供关联器做运动方向约束"""
        s = self._s.get(int(jid))
        return (0.0, 0.0) if s is None else (float(s[2]), float(s[3]))

    def age(self, jid):
        """已用实测修正过的次数 (刚初始化的关节速度不可信)"""
        s = self._s.get(int(jid))
        return 0 if s is None else int(s[4])

    def lost_age(self, jid):
        """连续丢失帧数 (含被异常门拒绝的)"""
        s = self._s.get(int(jid))
        return 0 if s is None else int(s[5])

    def update(self, measured, dt=1.0):
        """
        measured: {jid: (x, y)} 本帧实测 (只含真正匹配上的关节)
        dt: 相对于正常帧间隔的倍数 (1.0=正常)
        """
        dt = float(np.clip(dt, 0.25, 3.0))
        meas = {int(j): (float(p[0]), float(p[1]))
                for j, p in measured.items()}

        for j, s in self._s.items():
            x, y, vx, vy, age, lost = s

            px = x + vx * dt
            py = y + vy * dt

            z = meas.get(j)

            if z is None:
                # 未匹配: 沿运动方向短暂外推, 连续丢失后逐渐减速
                lost += 1
                decay = self.damping ** min(lost, 8)
                vx *= decay
                vy *= decay
                # 连续丢失过久时停止外推
                if lost >= 12:
                    vx = 0.0
                    vy = 0.0
                s[0] = px
                s[1] = py
                s[2] = vx
                s[3] = vy
                s[5] = lost
                continue

            mx, my = z
            rx = mx - px
            ry = my - py
            residual = float(np.hypot(rx, ry))

            # 预测误差越大, 越需要怀疑测量是否错配
            # 阈值随当前速度适度放宽
            spd = float(np.hypot(vx, vy))
            gate = max(12.0, self.outlier_gate * (2.0 + spd * 0.15))

            if residual > gate:
                # 大跳变: 不直接接受, 避免错误关联瞬间污染运动状态
                # 连续丢失后重新捕获由上层关联器决定是否 reseed
                s[0] = px
                s[1] = py
                s[2] = vx * self.damping
                s[3] = vy * self.damping
                s[5] = lost + 1
                continue

            # 动态纠正: 高速增响应, 低速减抖动
            alpha = float(np.clip(self.alpha + 0.0015 * spd,
                                   self.alpha, 0.96))
            beta = float(np.clip(self.beta + 0.001 * spd,
                                 self.beta, 0.75))

            nx = px + alpha * rx
            ny = py + alpha * ry

            nvx = vx + beta * rx / dt
            nvy = vy + beta * ry / dt

            # 限制速度变化, 降低单帧异常造成的速度突变
            dvx = nvx - vx
            dvy = nvy - vy
            dv = float(np.hypot(dvx, dvy))
            max_dv = self.max_accel * dt
            if dv > max_dv > 0:
                scale = max_dv / dv
                nvx = vx + dvx * scale
                nvy = vy + dvy * scale

            # 速度限幅 (保护机制, 非物理真值)
            v = float(np.hypot(nvx, nvy))
            if v > self.max_speed:
                scale = self.max_speed / v
                nvx *= scale
                nvy *= scale

            s[0] = nx
            s[1] = ny
            s[2] = nvx
            s[3] = nvy
            s[4] = age + 1
            s[5] = 0

        # 首次出现的关节初始化
        for j, z in meas.items():
            if j not in self._s:
                self._s[j] = [z[0], z[1], 0.0, 0.0, 1, 0]


class Blob:
    """一个候选反光点"""
    __slots__ = ("x", "y", "r", "area", "circ", "contrast")

    def __init__(self, x, y, r, area, circ, contrast):
        self.x, self.y, self.r = x, y, r
        self.area, self.circ, self.contrast = area, circ, contrast


def _merge_rects(rects):
    """合并相互重叠的矩形 (x1, y1, x2, y2), 避免同一区域重复处理"""
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
    """标定时直接从画面量反光点半径 (各点取中位数), 用来确定 top-hat 结构元大小"""
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
    """
    把候选反光点分配给 8 个关节

    代价 = 0.5·(d/σ)² + 0.25·(1-圆度) + 0.3·|r-r_ref|/r_ref + 0.8·(1-对比度/参考)
           └ 到预测点距离  └ 形状          └ 大小一致性        └ 对比度一致性
    4~8 号点之间再加骨段长度惩罚, 对每组互相关联的关节做带剪枝的精确搜索。
    关节宁可"未匹配"(代价 MISS_COST) 也不接受代价过高的候选。
    """
    MISS_COST = 6.0
    TOP_K = 4           # 每个关节最多保留的候选数
    NODE_BUDGET = 4000  # 搜索节点上限 (保证实时性)
    FOOT = (6, 7, 8)    # 足部刚体 (踝/脚跟/趾骨)
    CLUTTER_TOL = 1.0   # 与标定帧静止亮点的位置容差 (px) — 小则不误杀运动反光点

    def __init__(self, cfg):
        self.cfg = cfg
        self.bones = {}      # {(a, b): 标定长度 px}
        self.r_ref = 0.0     # 标定时的反光点半径
        self.contrast_ref = 0.0   # 标定时反光点的 top-hat 对比度
        self.clutter = []    # 标定帧里非标记点的静止亮点 [(x, y)]
        self._calib_pos = {} # 标定位置 {jid_int: (x, y)} — 正反馈循环防线
        self.debug = {}      # 供界面叠加显示
        self._se_cache = {}

    def reset(self):
        self.bones = {}
        self.r_ref = 0.0
        self.contrast_ref = 0.0
        self.clutter = []
        self._calib_pos = {}
        self.debug = {}

    def _calib_limit(self, jid):
        """标定位置约束: 候选离标定位置的最大允许距离 (px)
        上肢(1-4)运动小, 下肢(5-6)中等, 足部(7-8)蹬踏摆动大"""
        if jid <= 4: return 80
        if jid <= 6: return 120
        return 160

    def _body_center_x(self):
        """估算身体中线，供左右侧识别和高速时的反向候选过滤."""
        if not self._calib_pos:
            return None
        xs = [p[0] for p in self._calib_pos.values()]
        return float(np.median(xs))

    def _expected_side(self, jid, pred):
        """返回该关节在当前帧应该出现的左右侧：left/right/both.

        这里以身体中线分左右，避免高速时左/右反光点互相错配。"""
        center_x = self._body_center_x()
        if center_x is None or pred is None:
            return None
        return "left" if pred[0] < center_x else "right"

    def set_calibration(self, calib_pos, calib_radius=0.0, gray=None):
        """calib_pos: {jid_int: (x, y)}"""
        self._calib_pos = {int(j): (float(p[0]), float(p[1]))
                          for j, p in calib_pos.items()}
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
        # 标定点处的反光点对比度 (取中位数) -> 过滤低对比度碎片
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
        # 杂物图: 标定帧里不属于 8 个标记点的亮点 (车上的反光片/螺栓等).
        # 它们静止不动, 起步时速度未知, 最容易被误当成下肢标记点而"锁死"
        clear = max(8.0, 2.0 * self.r_ref)
        allb = self.extract_blobs(gray, [(0, 0, W, H)])
        cref = self.contrast_ref
        for b in allb:
            if cref > 0 and b.contrast < 0.25 * cref:
                continue
            if all(np.hypot(b.x - x, b.y - y) > clear for x, y in pts):
                self.clutter.append((b.x, b.y))

    # ─── 候选提取 ─────────────────────────────────────

    def _structuring_element(self):
        c = self.cfg
        r = (self.r_ref if self.r_ref >= 2
             else float(np.clip((c.min_radius + c.max_radius) / 3.0, 3, 8)))
        # 结构元约为反光点直径的 1.8 倍:
        #   比反光点大 → 反光点保留在 top-hat 结果里
        #   比鞋面/衣物的厚度小 → 大面积亮区被当作背景减掉
        k = int(np.clip(int(round(1.8 * 2 * r)) | 1, 9, 45))
        se = self._se_cache.get(k)
        if se is None:
            se = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            self._se_cache[k] = se
        return se, k

    def extract_blobs(self, gray, rects):
        """在给定矩形窗内提取候选反光点 (多级阈值 + 位置去重)"""
        c = self.cfg
        H, W = gray.shape[:2]
        se, k = self._structuring_element()
        pad = k // 2 + 1
        # 远距离 (约 1.6m 左右) 的反光点更暗、轮廓更小，固定阈值会被直接过滤。
        # 因此在低参考半径（小目标/远处）时主动放宽阈值并扩大半径检出范围。
        ref_r = max(2.0, self.r_ref if self.r_ref > 0 else c.min_radius)
        floor = max(6, int(c.tophat_floor * (0.6 if ref_r <= 6.0 else 0.8)))
        r_max = c.max_radius * (1.6 if ref_r <= 6.0 else 1.3)
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
            # 由高到低三级阈值: 高阈值先分出亮核, 低阈值补回偏暗的远距离点
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
                    if circ < c.track_min_circularity:
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
        """
        tracked : [jid_int]          被追踪的关节
        last_pos: {jid_int: (x, y)}  上一帧位置 (无预测时的后备)
        lost    : {jid_int: n}       已连续丢失帧数
        dt      : float              实际帧间隔 / 标称帧间隔 (1.0=正常)
        返回    : {jid_int: (x, y)}  本帧真正匹配上的关节
        """
        c = self.cfg
        H, W = gray.shape[:2]
        max_gate = max(40.0, min(W, H) / 3.0)

        st = {}
        rects = []
        for j in tracked:
            pred = motion.predict(j, dt) or last_pos.get(j)
            if pred is None:
                continue
            # 同时取速度, 供运动方向约束使用
            vx, vy = motion.velocity(j)
            lost_j = lost.get(j, 0)
            spd = motion.speed(j)
            base = c.track_max_move * (c.lower_gate_scale if j >= 5 else 1.0)
            # 匹配半径: 随速度、丢失帧数增大 (丢失后扩大搜索范围以便找回)
            gate = min(base * (1.0 + 0.35 * lost_j) + 0.5 * spd,
                       max(max_gate, base))
            # ── 标定位置约束: 打破正反馈循环 ──
            # 1) 预测回归: coast 外推会把预测位置带到远处, 必须拉回
            cx, cy = self._calib_pos.get(j, (None, None))
            limit = self._calib_limit(j)
            if cx is not None:
                d_calib = float(np.hypot(pred[0] - cx, pred[1] - cy))
                if d_calib > limit * 0.6:
                    # 预测位置已偏离标定太远 → 回归到标定位置
                    pred = (float(cx), float(cy))
                # gate 不超过标定约束范围 (防止搜索到远处的噪点)
                gate = min(gate, float(limit))
            # 距离容差 σ: 刚初始化(速度未知)或刚丢失时放宽
            sigma = (4.0 + 0.3 * spd + 3.0 * lost_j
                     + max(0, 3 - motion.age(j)) * 5.0)
            sigma = min(max(sigma, 6.0), gate / 2.0)
            # st 扩展为 6 元组: (px, py, gate, sigma, vx, vy)
            st[j] = (float(pred[0]), float(pred[1]), gate, sigma,
                     float(vx), float(vy))
            win = max(float(c.roi_radius), gate + c.max_radius + 4.0)
            rx1, ry1 = max(0, int(pred[0] - win)), max(0, int(pred[1] - win))
            rx2, ry2 = min(W, int(pred[0] + win) + 1), min(H, int(pred[1] + win) + 1)
            if rx2 > rx1 and ry2 > ry1:
                rects.append((rx1, ry1, rx2, ry2))

        blobs = self.extract_blobs(gray, _merge_rects(rects)) if rects else []
        blobs = self._filter_blobs(blobs)

        assign = self._assign(st, blobs)
        # ── 身份交换保护: 检测同一运动链内的关节是否发生互换 ──
        assign = self._protect_identity(assign, st, blobs)

        # 足部刚体共模修正: 已匹配上的点告诉我们整只脚这一帧移动了多少,
        # 据此修正其余点的预测后再分配一次 (解决起步/加速时预测偏差大的问题)
        # 共模修正前先记录第一次分配的总代价, 用于二次分配质量对比
        cost1 = self._assign_cost(assign, st, blobs)
        st2 = self._refine_foot(st, blobs, assign)
        if st2 is not None:
            assign2 = self._assign(st2, blobs)
            assign2 = self._protect_identity(assign2, st2, blobs)
            cost2 = self._assign_cost(assign2, st2, blobs)
            # 只有二次分配不比第一次差时才接受 (防止共模修正放大错误)
            if cost2 <= cost1 * 1.15:
                st = st2
                assign = assign2

        # ── 异常跳变检测: 残差过大的匹配标记为可疑, 不输出 ──
        # (MotionPredictor.update 内部也有异常门, 这里在关联层再防一道)
        matched = {}
        for j, bi in assign.items():
            px, py = st[j][0], st[j][1]
            b = blobs[bi]
            resid = float(np.hypot(b.x - px, b.y - py))
            spd = motion.speed(j)
            # 阈值随速度放宽, 但不超过 gate
            anomaly_gate = min(st[j][2],
                               max(15.0, 3.5 * (2.0 + spd * 0.15)))
            if resid <= anomaly_gate:
                matched[j] = (b.x, b.y)

        self.debug = {
            "blobs": [(b.x, b.y, b.r) for b in blobs],
            "pred": {j: (v[0], v[1]) for j, v in st.items()},
            "gate": {j: v[2] for j, v in st.items()},
            "clutter": list(self.clutter),
        }
        return matched

    def _filter_blobs(self, blobs):
        """剔除低对比度碎片和标定帧里就存在的静止杂物"""
        cref = self.contrast_ref
        tol = self.CLUTTER_TOL
        keep = []
        for b in blobs:
            if cref > 0 and b.contrast < 0.25 * cref:
                continue
            if any(abs(b.x - cx) <= tol and abs(b.y - cy) <= tol
                   for cx, cy in self.clutter):
                continue
            keep.append(b)
        return keep

    def _assign(self, st, blobs):
        """st: {j: (px, py, gate, sigma, vx, vy)} -> {j: blob_index}"""
        cref = self.contrast_ref
        options = {}
        for j, (px, py, gate, sigma, vx, vy) in st.items():
            cx, cy = self._calib_pos.get(j, (None, None))
            limit = self._calib_limit(j)
            spd = float(np.hypot(vx, vy))
            lst = []
            center_x = self._body_center_x()
            side_hint = self._expected_side(j, (px, py))
            for bi, b in enumerate(blobs):
                d = float(np.hypot(b.x - px, b.y - py))
                if d > gate:
                    continue
                # 标定位置约束: 候选离标定太远排除 (防止匹配到远处噪点)
                if cx is not None:
                    if np.hypot(b.x - cx, b.y - cy) > limit:
                        continue
                size = (min(1.5, abs(b.r - self.r_ref) / self.r_ref)
                        if self.r_ref >= 2 else 0.0)
                con = (min(1.0, max(0.0, 1.0 - b.contrast / cref))
                       if cref > 0 else 0.0)
                cost = (0.5 * (d / sigma) ** 2 + 0.25 * (1.0 - b.circ)
                        + 0.3 * size + 0.8 * con)
                # ── 运动方向约束: 高速时惩罚反方向候选 ──
                # 不完全禁止反向运动 (关节可以改变方向),
                # 但提高反方向候选的代价, 降低高速时身份互换概率
                if spd > 5.0 and d > 1.0:
                    dot = vx * (b.x - px) + vy * (b.y - py)
                    if dot < 0:
                        dir_pen = 0.3 * min(1.5, -dot / (spd * d))
                        cost += dir_pen
                # ── 左右侧强约束: 防止高速时把左侧候选误配到右侧/反之 ──
                # 以身体中线为边界，错侧候选直接显著惩罚，避免“点位错乱跟不上轨迹”。
                if side_hint is not None and center_x is not None:
                    if side_hint == "left" and b.x > center_x + 5.0:
                        cost += 3.0
                    elif side_hint == "right" and b.x < center_x - 5.0:
                        cost += 3.0
                if cost < self.MISS_COST:
                    lst.append((cost, bi))
            lst.sort()
            options[j] = lst[:self.TOP_K]
        return self._solve(list(st.keys()), options, blobs)

    def _refine_foot(self, st, blobs, assign):
        """
        足部 6/7/8 近似刚性平移: 取已匹配点的"实测-预测"偏差的共识值,
        平移其余点的预测并缩小容差. 至少 2 个点偏差一致才修正.

        改进:
        1. 残差方向一致性检查 — 各点残差方向应大致相同 (同一只脚整体平移)
        2. 动态阈值 — 根据标记点半径和骨段长度确定一致性容差, 不固定 14px
        3. 骨段长度验证 — 修正后骨段长度仍需合理, 否则拒绝修正
        """
        inn = {}
        for j in self.FOOT:
            if j in assign and j in st:
                b = blobs[assign[j]]
                inn[j] = (b.x - st[j][0], b.y - st[j][1])
        if len(inn) < 2:
            return None

        mx = float(np.median([v[0] for v in inn.values()]))
        my = float(np.median([v[1] for v in inn.values()]))

        # 动态一致性阈值: 基于标记点半径和足部短骨段长度
        # 短骨段 (踝-脚跟/踝-趾骨/脚跟-趾骨) 标定长度通常 25-60px
        foot_bones = [L for (a, b), L in self.bones.items()
                      if a in self.FOOT and b in self.FOOT]
        ref_len = float(np.median(foot_bones)) if foot_bones else 40.0
        tol = max(6.0, 0.25 * ref_len + 2.0 * self.r_ref)

        agree = [j for j, (ix, iy) in inn.items()
                 if np.hypot(ix - mx, iy - my) <= tol]
        if len(agree) < 2:
            return None

        # 残差方向一致性: 参与修正的点的残差方向应大致相同
        # (同一只脚整体平移, 不会出现一个点往左、另一个往右)
        res_angles = []
        for j in agree:
            ix, iy = inn[j]
            if np.hypot(ix, iy) > 3.0:
                res_angles.append(np.arctan2(iy, ix))
        if len(res_angles) >= 2:
            # 角度差超过 90° 说明残差方向不一致, 可能已发生身份互换
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
                new[j] = (px + dx, py + dy, gate, max(6.0, 0.5 * sigma),
                          vx, vy)

        # 骨段长度验证: 修正后的预测位置之间的距离仍需接近标定骨长
        for (a, b), L in self.bones.items():
            if a in self.FOOT and b in self.FOOT and a in new and b in new:
                d_new = float(np.hypot(new[a][0] - new[b][0],
                                       new[a][1] - new[b][1]))
                tol_b = 0.35 * L + 12.0 if L <= 60 else 0.15 * L + 2.0
                if abs(d_new - L) > tol_b * 1.5:
                    return None

        return new

    # ─── 身份交换保护 ─────────────────────────────────

    def _protect_identity(self, assign, st, blobs):
        """
        检测同一运动链内的关节是否发生了身份互换.

        原理: _solve_group 的骨长约束在两个候选点很近时惩罚差异很小,
        可能仅凭距离选错. 这里在分配完成后做二次验证:
        对每对相邻关节 (有骨段约束的), 比较保持当前分配 vs 交换分配,
        只有交换在运动连续性 + 骨段几何上明显更优时才接受.

        返回修正后的 assign (可能不变).
        """
        if not assign or not self.bones:
            return assign

        result = dict(assign)

        for (a, b), L in self.bones.items():
            if a not in result or b not in result:
                continue
            ba, bb = result[a], result[b]
            if ba is None or bb is None:
                continue

            pa = st.get(a)
            pb = st.get(b)
            if pa is None or pb is None:
                continue

            ba_obj, bb_obj = blobs[ba], blobs[bb]

            # 当前分配的代价: 预测残差 + 骨长偏差
            cur_resid = (np.hypot(ba_obj.x - pa[0], ba_obj.y - pa[1]) +
                         np.hypot(bb_obj.x - pb[0], bb_obj.y - pb[1]))
            cur_bone = abs(np.hypot(ba_obj.x - bb_obj.x,
                                    ba_obj.y - bb_obj.y) - L)
            cur_cost = cur_resid + cur_bone * 2.0

            # 交换分配的代价
            sw_resid = (np.hypot(bb_obj.x - pa[0], bb_obj.y - pa[1]) +
                        np.hypot(ba_obj.x - pb[0], ba_obj.y - pb[1]))
            sw_bone = abs(np.hypot(bb_obj.x - ba_obj.x,
                                    bb_obj.y - ba_obj.y) - L)
            sw_cost = sw_resid + sw_bone * 2.0

            # 只有交换明显更优 (余量 > 8px 等效) 才接受
            # 否则保持当前分配, 维护身份连续性
            if sw_cost < cur_cost - 8.0:
                result[a], result[b] = bb, ba

        return result

    def _assign_cost(self, assign, st, blobs):
        """计算分配方案的总代价 (预测残差 + 骨长偏差), 用于质量对比"""
        if not assign:
            return float("inf")
        total = 0.0
        for j, bi in assign.items():
            if bi is None or j not in st:
                continue
            px, py = st[j][0], st[j][1]
            b = blobs[bi]
            total += float(np.hypot(b.x - px, b.y - py))
        for (a, b), L in self.bones.items():
            if a in assign and b in assign:
                ba, bb = assign[a], assign[b]
                if ba is not None and bb is not None:
                    d = float(np.hypot(blobs[ba].x - blobs[bb].x,
                                       blobs[ba].y - blobs[bb].y))
                    total += abs(d - L) * 2.0
        return total

    # ─── 带骨长约束的分配 ─────────────────────────────

    def _solve(self, joints, options, blobs):
        """把互相有关联 (共享候选/有骨段) 的关节分组, 每组做精确搜索"""
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
        w = self.cfg.bone_weight
        miss = self.MISS_COST
        order = sorted(group, key=lambda j: (len(options.get(j, [])), j))
        bones_of = {j: [] for j in group}
        for (a, b), L in self.bones.items():
            if a in bones_of and b in bones_of:
                bones_of[a].append((b, L))
                bones_of[b].append((a, L))

        def bone_pen(d, L):
            # 短骨段(≤60px: 踝-脚跟/踝-趾骨/脚跟-趾骨)用大容差:
            # 蹬踏时 2D 投影下短骨段长度变化大 (25px→49px 很常见)
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
            assign[j] = None          # 允许该关节本帧不匹配
            dfs(i + 1, cost + miss)
            del assign[j]

        dfs(0, 0.0)
        return {j: bi for j, bi in best[1].items() if bi is not None}
