"""骑行姿态实时指标计算引擎

计算16项生物力学指标:
  圈数、膝关节角度(实时/最大/最小)、踝关节角度、肩部角度、手肘角度、
  水平背角度、足膝偏移量、膝关节偏摆量、肩关节偏摆量、
  髋关节水平位移、髋关节垂直位移、髋关节偏摆量、踝角度变化区间、髋闭角角度

关节编号: 1腕 2肘 3肩 4髋 5膝 6踝 7脚跟 8趾骨
"""
from collections import deque

import numpy as np

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
    'hip_closed',
]

# 滚动窗口型指标: 实时=相对窗口均值的偏移 (踝角度区间: 实时=窗口内最大-最小), 最大/最小=窗口内极值
RANGE_METRICS = ['knee_lateral', 'shoulder_lateral', 'hip_lateral', 'ankle_range']

# 最小值型指标 (只追踪最小) — hip_closed 已并入 TRACK_STATS, 保留接口以备扩展
MIN_METRICS = []


def compute_all_metrics(pixels, angles, calib_pos=None, facing=1):
    """计算当前帧的所有指标

    pixels: {jid_int: (x, y)}  8个关节像素坐标
    angles: {'elbow':v, 'shoulder':v, 'hip':v, 'knee':v, 'ankle':v}
    calib_pos: {jid_int: (x, y)}  标定位置
    facing: +1 骑行者朝右 / -1 朝左; 水平类指标乘以此符号, 使"正=向前"与画面左右朝向无关
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
    # 膝(5)与踝(6)的水平距离 (px), 正=膝在前 (乘朝向符号统一方向)
    if 5 in p and 6 in p:
        m['foot_knee_offset'] = facing * (p[5][0] - p[6][0])

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
        m['hip_h_disp'] = facing * (p[4][0] - cx)
        m['hip_v_disp'] = p[4][1] - cy

    # --- 髋闭角角度 (髋关节最小角度 = 最大屈曲) ---
    m['hip_closed'] = angles.get('hip')

    return m


class MetricsCollector:
    """实时指标采集器: 追踪各项指标的实时/最大/最小/平均/圈数"""

    def __init__(self):
        self.reset()

    def reset(self, calib_pos=None, facing=1):
        self._calib_pos = calib_pos or {}
        self._facing = 1 if facing >= 0 else -1
        self._frame_count = 0
        self._revolutions = 0

        # 圈数: 膝盖Y峰谷检测
        self._ky_prev = None
        self._ky_dir = 0       # -1=上移(y减), +1=下移(y增)
        self._last_bottom = None  # 上次谷底(踏板下死点, y最大)
        self._last_top = None    # 上次峰顶(踏板上死点, y最小)
        self._amp_thresh = 15   # 最小振幅(px), 过滤噪声

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

    def update(self, pixels, angles):
        """采集一帧数据, 返回显示用 dict"""
        if not pixels or len(pixels) < 3:
            return self.get_display()

        self._frame_count += 1
        m = compute_all_metrics(pixels, angles, self._calib_pos, self._facing)
        p = {jid: (float(x), float(y)) for jid, (x, y) in pixels.items()}

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
            xv = self._facing * p[jid][0]
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

        return self.get_display()

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
