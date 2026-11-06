"""骑行姿态实时指标计算引擎

计算16项生物力学指标:
  圈数、膝关节角度(实时/最大/最小)、踝关节角度、肩部角度、手肘角度、
  水平背角度、足膝偏移量、膝关节偏摆量、肩关节偏摆量、
  髋关节水平位移、髋关节垂直位移、髋关节偏摆量、踝角度变化区间、髋闭角角度

关节编号: 1腕 2肘 3肩 4髋 5膝 6踝 7脚跟 8趾骨

约定:
  · 所有"前后"量按骑行者朝向取符号: 向前为正 (reset(calib_pos, facing) 的 facing=+1 朝右 / -1 朝左;
    不传 facing 时按标定点自动判断); 竖直量向上为正
  · 长度量统一换算成 cm: 以标定时 髋-膝(大腿) 的像素长度对应 THIGH_CM (默认 43 cm,
    环境变量 KINEX_THIGH_CM 可改; 设为 0 则保持像素)。每个相机各算各的比例
  · 足膝偏移量(实时) = 膝相对踏轴的水平偏移; KOPS值 = 曲柄水平朝前 (3 点钟) 时的足膝偏移。
    踏轴用趾骨点(8)近似 (环境变量 KINEX_SPINDLE 可改成 6/7)
  · 双相机: 每个相机一个 MetricsCollector, 用 BilateralMetrics 汇总 (圈数只取一侧, 不重复计数)
"""
import math
import os
from collections import deque

import numpy as np

THIGH_CM = float(os.environ.get("KINEX_THIGH_CM", "43"))
SPINDLE = int(os.environ.get("KINEX_SPINDLE", "8"))

# 偏摆 / 踝角度区间 的滚动窗口: 最近 N 个"不重复"采样 (重复帧不计入, 相机/视频都适用)
SWAY_WINDOW = 90

# 指标中文名
METRIC_CN = {
    'revolutions': '圈数',
    'knee_angle': '膝关节角度',
    'ankle_angle': '踝关节角度',
    'shoulder_angle': '肩部角度',
    'elbow_angle': '手肘角度',
    'back_angle': '水平背角度',
    'foot_knee_offset': '足膝偏移量',
    'kops': 'KOPS值',
    'knee_lateral': '膝关节偏摆',
    'shoulder_lateral': '肩关节偏摆',
    'hip_h_disp': '髋水平位移',
    'hip_v_disp': '髋垂直位移',
    'hip_lateral': '髋关节偏摆',
    'ankle_range': '踝角度区间',
    'hip_closed': '髋闭角角度',
}

# 需要追踪 max/min/avg 的指标
TRACK_STATS = [
    'knee_angle', 'ankle_angle', 'shoulder_angle', 'elbow_angle',
    'back_angle', 'foot_knee_offset', 'hip_h_disp', 'hip_v_disp',
    'hip_closed', 'kops',
]

# 滚动窗口型指标: 实时=相对窗口均值的偏移 (踝角度区间: 实时=窗口内最大-最小), 最大/最小=窗口内极值
RANGE_METRICS = ['knee_lateral', 'shoulder_lateral', 'hip_lateral', 'ankle_range']

# 最小值型指标 (只追踪最小) — hip_closed 已并入 TRACK_STATS, 保留接口以备扩展
MIN_METRICS = []


def compute_all_metrics(pixels, angles, calib_pos=None, facing=1, k=1.0, spindle=SPINDLE):
    """计算当前帧的所有指标

    pixels: {jid_int: (x, y)}  8个关节像素坐标
    angles: {'elbow':v, 'shoulder':v, 'hip':v, 'knee':v, 'ankle':v}
    calib_pos: {jid_int: (x, y)}  标定位置
    facing: +1 骑行者朝右 / -1 朝左; 水平类指标乘以此符号, 使"正=向前"与画面左右朝向无关
    k: cm / px 比例 (长度类指标乘以它)
    返回 dict: {metric_name: value}
    """
    m = {name: None for name in METRIC_CN}
    if not pixels or len(pixels) < 3:
        return m

    p = {jid: (float(x), float(y)) for jid, (x, y) in pixels.items()}

    # --- 已有角度 ---
    m['knee_angle'] = angles.get('knee')
    m['ankle_angle'] = angles.get('ankle')
    m['shoulder_angle'] = angles.get('shoulder')
    m['elbow_angle'] = angles.get('elbow')

    # --- 水平背角度 ---
    # 肩(3)→髋(4)连线与水平线的夹角 (0°=水平, 90°=垂直)
    if 3 in p and 4 in p:
        dx = abs(p[4][0] - p[3][0])
        dy = abs(p[4][1] - p[3][1])
        if dx + dy > 1e-6:
            m['back_angle'] = float(np.degrees(np.arctan2(dy, dx)))

    # --- 足膝偏移量 ---
    # 膝(5)相对踏轴(默认趾骨点 8)的水平偏移 (cm), 向前为正; 标准 KOPS 值见 'kops'
    sp = spindle if spindle in p else (6 if 6 in p else None)
    if 5 in p and sp is not None:
        m['foot_knee_offset'] = facing * (p[5][0] - p[sp][0]) * k

    # --- 偏摆量: X方向运动范围 ---
    # 实时值由 MetricsCollector 追踪历史后计算 range
    if 5 in p:
        m['knee_lateral'] = p[5][0]
    if 3 in p:
        m['shoulder_lateral'] = p[3][0]
    if 4 in p:
        m['hip_lateral'] = p[4][0]

    # --- 髋关节位移 (相对标定) --- 水平位移乘朝向符号 (正=向前)
    if 4 in p and calib_pos and 4 in calib_pos:
        cx, cy = calib_pos[4]
        m['hip_h_disp'] = facing * (p[4][0] - cx) * k          # 向前为正 (cm)
        m['hip_v_disp'] = -(p[4][1] - cy) * k                  # 向上为正 (cm)

    # --- 髋闭角角度 (髋关节最小角度 = 最大屈曲) ---
    m['hip_closed'] = angles.get('hip')

    return m


class MetricsCollector:
    """实时指标采集器: 追踪各项指标的实时/最大/最小/平均/圈数"""

    def __init__(self):
        self.reset()

    def reset(self, calib_pos=None, facing=None):
        """facing: +1 朝右 / -1 朝左 (与 app 的 _facing_sign 一致); None = 按标定点自动判断"""
        self._calib_pos = calib_pos or {}
        g_facing, self.k, self._thigh_px = self._geometry(self._calib_pos)
        self._facing_given = facing is not None
        self._facing = (1 if facing >= 0 else -1) if self._facing_given else int(g_facing)
        self._frame_count = 0
        self._revolutions = 0

        # 圈数: 膝盖Y峰谷检测
        self._ky_prev = None
        self._ky_dir = 0       # -1=上移(y减), +1=下移(y增)
        self._last_bottom = None  # 上次谷底(踏板下死点, y最大)
        self._last_top = None    # 上次峰顶(踏板上死点, y最小)
        self._amp_thresh = 0.13 * self._thigh_px if self._thigh_px else 15   # 最小振幅(px): 随画面比例缩放

        # 指标统计 {name: {cur, max, min, sum, cnt}}
        self._stats = {}
        for name in METRIC_CN:
            self._stats[name] = {'cur': None, 'max': None,
                                 'min': None, 'sum': 0.0, 'cnt': 0}

        # 滚动窗口 (有限长度, 不再无限增长): X 方向位置 / 踝角度
        self._x_hist = {5: deque(maxlen=SWAY_WINDOW), 3: deque(maxlen=SWAY_WINDOW),
                        4: deque(maxlen=SWAY_WINDOW)}
        self._ank_hist = deque(maxlen=SWAY_WINDOW)
        self._last_key = None

        # 曲柄 (踏轴点轨迹) → KOPS: 曲柄水平朝前时测量
        self._cr_ring = deque(maxlen=200)
        self._cr_hist = deque(maxlen=3)            # (e, 实时足膝偏移), e = 曲柄角 - 朝前水平角

    def update(self, pixels, angles):
        """采集一帧数据, 返回显示用 dict"""
        if not pixels or len(pixels) < 3:
            return self.get_display()

        self._frame_count += 1
        p = {jid: (float(x), float(y)) for jid, (x, y) in pixels.items()}
        if self._thigh_px is None:                  # 未经标定直接使用: 用首帧推算比例 (和朝向, 若未指定)
            g_facing, self.k, self._thigh_px = self._geometry(p)
            if not self._facing_given:
                self._facing = int(g_facing)
            if self._thigh_px:
                self._amp_thresh = 0.13 * self._thigh_px
        m = compute_all_metrics(pixels, angles, self._calib_pos, self._facing, self.k)

        # 1. 圈数
        if 5 in p:
            self._update_revolutions(p[5][1])
        m['revolutions'] = self._revolutions

        # 2. 追踪 max/min/avg
        for name in TRACK_STATS:
            v = m.get(name)
            if v is not None:
                self._track(name, v)

        # 3. 偏摆量: 相对滚动均值的偏移 (实时 / 窗口最大 / 窗口最小)
        #    旧版取"从开始到现在的累计极差", 只增不减, 踩一圈后就不再变化
        new_sample = self._is_new_sample(p)
        # 偏摆量: X 方向运动范围, 存入朝向校正后的 x (正=向前)
        for jid, metric in [(5, 'knee_lateral'), (3, 'shoulder_lateral'),
                             (4, 'hip_lateral')]:
            if jid not in p:
                continue
            h = self._x_hist[jid]
            xv = self._facing * p[jid][0] * self.k
            if new_sample:
                h.append(xv)
            if len(h) > 1:
                mean = sum(h) / len(h)
                st = self._stats[metric]
                cur = xv - mean
                st['cur'] = cur
                st['max'] = max(h) - mean
                st['min'] = min(h) - mean
                st['sum'] += cur
                st['cnt'] += 1

        # 4. 踝角度区间: 滚动窗口内 最大-最小 (实时), 并给出窗口内最大/最小角
        av = m.get('ankle_angle')
        if av is not None and new_sample:
            self._ank_hist.append(av)
        if len(self._ank_hist) > 1:
            hi, lo = max(self._ank_hist), min(self._ank_hist)
            ar = self._stats['ankle_range']
            rng = hi - lo
            ar['cur'] = rng
            ar['max'] = hi
            ar['min'] = lo
            ar['sum'] += rng
            ar['cnt'] += 1

        # 5. 髋闭角 = 髋角 (追踪 max/min/avg: min=最大屈曲, max=最大伸展)
        hv = m.get('hip_closed')
        if hv is not None:
            self._track('hip_closed', hv)

        # 6. KOPS: 曲柄水平朝前 (3 点钟) 时, 膝相对踏轴的水平偏移
        if new_sample:
            self._update_kops(p, m.get('foot_knee_offset'))

        return self.get_display()

    @staticmethod
    def _geometry(pos):
        """由一组关节位置得到 (朝向, cm/px, 大腿像素长度); 位置不足时 (1.0, 1.0, None)"""
        pos = {int(j): (float(v[0]), float(v[1])) for j, v in (pos or {}).items()}
        facing = 1.0
        if 7 in pos and 8 in pos:                       # 与 app 一致: 趾骨在脚跟前方 → 朝右
            facing = 1.0 if pos[8][0] > pos[7][0] else -1.0
        else:
            ref = pos.get(1) or pos.get(3)
            if ref is not None and 4 in pos:
                facing = 1.0 if ref[0] >= pos[4][0] else -1.0
        thigh = None
        if 4 in pos and 5 in pos:
            d = math.hypot(pos[4][0] - pos[5][0], pos[4][1] - pos[5][1])
            thigh = d if d > 10 else None
        k = (THIGH_CM / thigh) if (thigh and THIGH_CM > 0) else 1.0
        return facing, k, thigh

    def _crank_center(self):
        """曲柄中心: 对踏轴点轨迹做最小二乘圆拟合 (不要求覆盖整数圈); 点太集中时退回均值"""
        pts = np.asarray(self._cr_ring, float)
        mx, my = pts.mean(axis=0)
        if np.ptp(pts[:, 0]) < 8 and np.ptp(pts[:, 1]) < 8:
            return float(mx), float(my)
        x, y = pts[:, 0] - mx, pts[:, 1] - my
        A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
        sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
        return float(mx + sol[0]), float(my + sol[1])

    def _update_kops(self, p, live_offset):
        """踏轴点绕曲柄中心转: 曲柄角经过"水平朝前"时, 用相邻 3 个采样做二次插值得到该时刻的足膝偏移"""
        sp = SPINDLE if SPINDLE in p else (6 if 6 in p else None)
        if sp is None or live_offset is None:
            return
        self._cr_ring.append(p[sp])
        if len(self._cr_ring) < 8:
            return
        cx, cy = self._crank_center()
        theta = math.atan2(p[sp][1] - cy, p[sp][0] - cx)
        phi_f = 0.0 if self._facing > 0 else math.pi             # 曲柄指向前方水平
        e = (theta - phi_f + math.pi) % (2 * math.pi) - math.pi
        if self._cr_hist:
            e0, v0 = self._cr_hist[-1]
            # 过零且方向为向前蹬踏 (朝右时角度增大, 朝左时减小), 排除跨 ±π 的跳变
            if e0 * e < 0 and abs(e - e0) < math.pi and (e - e0) * self._facing > 0:
                pts = list(self._cr_hist) + [(e, live_offset)]
                if len(pts) >= 3:
                    (a, va), (b, vb), (c, vc) = pts[-3:]
                    v = (va * (0 - b) * (0 - c) / ((a - b) * (a - c)) +
                         vb * (0 - a) * (0 - c) / ((b - a) * (b - c)) +
                         vc * (0 - a) * (0 - b) / ((c - a) * (c - b)))
                else:
                    v = v0 + (live_offset - v0) * (0 - e0) / (e - e0)
                self._track('kops', v)
        self._cr_hist.append((e, live_offset))

    def _is_new_sample(self, p):
        """与上一采样相比关节位置有变化才算新采样 (视频重复帧 / 相机重复读取不计入窗口)"""
        key = tuple(round(p[j][k], 1) if j in p else 0.0 for j in (3, 4, 5, 6) for k in (0, 1))
        if self._last_key is not None and max(
                abs(a - b) for a, b in zip(key, self._last_key)) < 0.6:
            return False
        self._last_key = key
        return True

    def _track(self, name, v):
        s = self._stats[name]
        s['cur'] = v
        s['sum'] += v
        s['cnt'] += 1
        if s['max'] is None or v > s['max']:
            s['max'] = v
        if s['min'] is None or v < s['min']:
            s['min'] = v

    def _track_min(self, name, v):
        s = self._stats[name]
        s['cur'] = v
        if s['min'] is None or v < s['min']:
            s['min'] = v

    def _update_revolutions(self, knee_y):
        """圈数检测: 膝盖Y的峰谷检测, 一个完整周期(谷→峰→谷)=1圈"""
        if self._ky_prev is None:
            self._ky_prev = knee_y
            return

        dy = knee_y - self._ky_prev
        if abs(dy) < 1.0:
            return

        cur_dir = 1 if dy > 0 else -1  # y增=膝下降, y减=膝上升

        if self._ky_dir != 0 and cur_dir != self._ky_dir:
            # 方向反转 → 经过极值点
            if cur_dir == 1 and self._ky_dir == -1:
                # 膝从上升→下降, 刚经过峰顶 (y最小 = 踏板上死点)
                if self._last_bottom is not None:
                    amp = self._last_bottom - self._ky_prev
                    if amp >= self._amp_thresh:
                        self._revolutions += 1
                self._last_top = self._ky_prev
            elif cur_dir == -1 and self._ky_dir == 1:
                # 膝从下降→上升, 刚经过谷底 (y最大 = 踏板下死点)
                self._last_bottom = self._ky_prev

            self._ky_dir = cur_dir

        if self._ky_dir == 0:
            self._ky_dir = cur_dir

        self._ky_prev = knee_y

    def get_display(self):
        """返回格式化的显示数据 {metric: {cur, max, min, avg}}"""
        result = {}
        for name in METRIC_CN:
            s = self._stats[name]
            entry = {'cur': s.get('cur')}
            if name in TRACK_STATS:
                entry['max'] = s.get('max')
                entry['min'] = s.get('min')
                entry['avg'] = (s['sum'] / s['cnt']) if s['cnt'] > 0 else None
            elif name in RANGE_METRICS:
                entry['max'] = s.get('max')
                entry['min'] = s.get('min')
                entry['avg'] = (s['sum'] / s['cnt']) if s['cnt'] > 0 else None
            elif name == 'revolutions':
                entry['cur'] = self._revolutions
            elif name in MIN_METRICS:
                entry['min'] = s.get('min')
                entry['cur'] = s.get('cur')
            result[name] = entry
        return result


class BilateralMetrics:
    """双相机 (左 / 右侧各一台) 汇总:
        · 每侧一个 MetricsCollector, 各自的朝向、cm/px 比例、窗口和曲柄角, 互不混用
        · 圈数只取"主侧"一个, 不会因为两台相机而重复计数
        · 单相机时只有一侧有数据, 另一侧的估计值 (正面图里的虚线腿) 不进入任何统计
        · asymmetry(): 左 - 右 的差值, 两侧都有数据时才有
    """
    # 左右差只比较"稳定的统计量", 不比较瞬时值 (两条腿相位差半圈, 瞬时值必然不同)
    ASYM_COL = {'knee_angle': 'avg', 'ankle_angle': 'avg', 'elbow_angle': 'avg',
                'shoulder_angle': 'avg', 'back_angle': 'avg',
                'hip_closed': 'min',            # 髋闭角: 各侧的最小值
                'kops': 'avg'}                  # 3 点钟足膝偏移: 各侧的平均

    def __init__(self, primary='R'):
        self.sides = {'L': MetricsCollector(), 'R': MetricsCollector()}
        self.primary = primary
        self._frames = {'L': 0, 'R': 0}

    def reset(self, calib_by_side=None, facing_by_side=None):
        calib_by_side = calib_by_side or {}
        facing_by_side = facing_by_side or {}
        for s in ('L', 'R'):
            self.sides[s].reset(calib_by_side.get(s), facing_by_side.get(s))
            self._frames[s] = 0

    def update(self, side, pixels, angles):
        self._frames[side] += 1
        return self.sides[side].update(pixels, angles)

    def active_sides(self):
        return [s for s in ('L', 'R') if self._frames[s] > 0]

    def primary_side(self):
        act = self.active_sides()
        if len(act) == 1:
            return act[0]
        return self.primary if self.primary in act or not act else act[0]

    def revolutions(self):
        return self.sides[self.primary_side()]._revolutions

    def get_display(self, side=None):
        side = side or self.primary_side()
        d = self.sides[side].get_display()
        d['revolutions'] = {'cur': self.revolutions()}
        return d

    def asymmetry(self):
        """{指标: 左 - 右}; 各指标取 ASYM_COL 指定的统计列, 缺一侧返回 {}"""
        if len(self.active_sides()) < 2:
            return {}
        L, R = (self.sides[s].get_display() for s in ('L', 'R'))
        out = {}
        for key, col in self.ASYM_COL.items():
            a, b = L.get(key, {}).get(col), R.get(key, {}).get(col)
            if a is not None and b is not None:
                out[key] = a - b
        return out