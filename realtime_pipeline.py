# -*- coding: utf-8 -*-
"""
实时流水线 — 让相机/视频捕捉丝滑流畅

原来的瓶颈 (全部串在 GUI 线程的 33ms QTimer 里):
    取帧(Kinect update, 阻塞) → 红外转换 → MediaPipe 推理 → 反光点精化
    → 角度/指标 → 画 HUD → QImage → QPixmap → 缩放 → 显示
任何一步超过 33ms, 界面就卡顿丢帧; 而且 Windows 上 QTimer 默认是粗粒度定时器
(约 15.6ms 分辨率), 33ms 的节拍实际会在 31/47ms 之间抖动 → 肉眼可见的"不匀速"。

新架构 (三段解耦):

    [采集线程]  阻塞读相机/解码视频 ──► LatestSlot(单槽, 只留最新帧) ──►
    [处理线程]  追踪 + 角度 + 指标 + 画 HUD + 缩放到显示尺寸 + BGRA 转换 ──►
    [GUI 线程]  只做一件事: 把现成的 QImage 贴到屏幕上

    · 相机: 永远处理"最新一帧", 处理慢了就丢旧帧, 延迟不会累积
    · 视频: 按原生 fps 打节拍, 背压不丢帧 (指标统计精确), 慢了就整体降速但节奏均匀
    · GUI 线程不再做任何重活, 拖参数/点按钮不会让画面卡住

对外接口:
    PipelineDriver   鸭子类型兼容 QTimer 的 start()/stop(), app.py 里原有调用无需改
    VideoLabel       自绘视频控件 (替代原 QLabel+QPixmap.scaled), 保留 clicked / set_frame_size
    to_display_bgra  BGR → 显示尺寸 BGRA (给处理线程用)
    fit_size         等比缩放尺寸计算
"""

from __future__ import annotations

import threading
import time
import traceback
from collections import deque

import cv2
import numpy as np
from PyQt6.QtCore import (QObject, QPointF, QRectF, Qt, QThread,
                          pyqtSignal, pyqtSlot)
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import QLabel, QSizePolicy


# ==================== 小工具 ====================

def fit_size(sw: int, sh: int, bw: int, bh: int):
    """把 (sw, sh) 等比缩放进 (bw, bh); 不放大 (放大交给 Qt 绘制)"""
    if bw <= 0 or bh <= 0 or sw <= 0 or sh <= 0:
        return sw, sh
    s = min(1.0, bw / sw, bh / sh)
    return max(1, int(sw * s)), max(1, int(sh * s))


def to_display_bgra(img: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    """BGR/灰度 → 显示尺寸的 BGRA (Qt 的 Format_RGB32 与之内存布局一致, 贴图最快)"""
    h, w = img.shape[:2]
    dw, dh = fit_size(w, h, box_w, box_h)
    if (dw, dh) != (w, h):
        img = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_LINEAR)
    code = cv2.COLOR_GRAY2BGRA if img.ndim == 2 else cv2.COLOR_BGR2BGRA
    return cv2.cvtColor(img, code)


class RateMeter:
    """滑动窗口 FPS 统计 (单写多读, 无需加锁)"""

    def __init__(self, window: int = 30):
        self._ts = deque(maxlen=window)

    def tick(self):
        self._ts.append(time.perf_counter())

    @property
    def fps(self) -> float:
        try:
            dt = self._ts[-1] - self._ts[0]
            return (len(self._ts) - 1) / dt if dt > 0 else 0.0
        except IndexError:
            return 0.0


class LatestSlot:
    """
    单槽邮箱: put 覆盖旧帧 (latest-wins), get 取走最新帧。
    wait_free=True 时 put 会等槽被取走 (背压, 用于视频文件, 不丢帧)。
    """

    def __init__(self):
        self._cv = threading.Condition()
        self._item = None
        self._closed = False

    def put(self, item, wait_free: bool = False) -> bool:
        with self._cv:
            if wait_free:
                while self._item is not None and not self._closed:
                    self._cv.wait(0.05)
            if self._closed:
                return False
            self._item = item
            self._cv.notify_all()
            return True

    def get(self, timeout: float = 0.1):
        with self._cv:
            if self._item is None and not self._closed:
                self._cv.wait(timeout)
            item, self._item = self._item, None
            if item is not None:
                self._cv.notify_all()
            return item

    def close(self):
        with self._cv:
            self._closed = True
            self._item = None
            self._cv.notify_all()


# ==================== 采集线程 ====================

class CaptureThread(QThread):
    """
    循环调用 read_fn() 取帧并投入 LatestSlot。

    read_fn() -> (frame_bgr, gray[, idx]) | None
        返回 None 视为暂时无帧 (持续超过 no_frame_timeout 秒才判定故障);
        抛异常视为设备故障。
    frame_interval > 0: 按该间隔打节拍 (视频文件, 按原生 fps 播放)
    frame_interval = 0: 不打节拍, 由设备阻塞决定 (相机); min_period 防空转
    """
    failed = pyqtSignal(str)

    def __init__(self, read_fn, slot: LatestSlot, *, frame_interval: float = 0.0,
                 block_when_full: bool = False, min_period: float = 0.004,
                 no_frame_timeout: float = 1.5, parent=None):
        super().__init__(parent)
        self.setObjectName("pose-capture")
        self._read_fn = read_fn
        self._slot = slot
        self._interval = frame_interval
        self._block = block_when_full
        self._min_period = min_period
        self._timeout = no_frame_timeout
        self._quit = threading.Event()
        self.meter = RateMeter()

    def request_stop(self):
        self._quit.set()

    def _sleep(self, sec: float):
        """可中断睡眠, 分段 sleep 避开 Event.wait 在 Windows 上的粗粒度"""
        end = time.perf_counter() + sec
        while not self._quit.is_set():
            rest = end - time.perf_counter()
            if rest <= 0:
                return
            time.sleep(min(rest, 0.004))

    def run(self):
        next_t = time.perf_counter()
        last_ok = next_t
        idx = 0
        while not self._quit.is_set():
            t0 = time.perf_counter()
            try:
                item = self._read_fn()
            except Exception as e:  # 设备故障
                traceback.print_exc()
                self.failed.emit(f"采集异常: {e}")
                return
            if item is None:
                if time.perf_counter() - last_ok > self._timeout:
                    self.failed.emit("相机无信号, 请检查连接")
                    return
                self._sleep(0.002)
                continue
            last_ok = time.perf_counter()
            frame, gray = item[0], item[1]
            idx = item[2] if len(item) > 2 and item[2] is not None else idx + 1
            if not self._slot.put((frame, gray, idx), wait_free=self._block):
                return
            self.meter.tick()

            if self._interval > 0:                      # 视频: 匀速节拍
                next_t += self._interval
                delay = next_t - time.perf_counter()
                if delay > 0:
                    self._sleep(delay)
                elif delay < -3 * self._interval:       # 落后太多, 重新对表
                    next_t = time.perf_counter()
            else:                                       # 相机: 只防空转
                rest = self._min_period - (time.perf_counter() - t0)
                if rest > 0:
                    self._sleep(rest)


# ==================== 处理线程 ====================

class ProcessThread(QThread):
    """
    从 LatestSlot 取最新帧 → process_fn(frame, gray, idx) → 结果放入输出槽。
    输出做了信号合并: GUI 没来得及取走时不会堆积信号。
    """
    result_ready = pyqtSignal()

    def __init__(self, process_fn, slot: LatestSlot, parent=None):
        super().__init__(parent)
        self.setObjectName("pose-process")
        self._fn = process_fn
        self._slot = slot
        self._quit = threading.Event()
        self._out_lock = threading.Lock()
        self._out = None
        self._notified = False
        self.meter = RateMeter()
        self._last_err = 0.0

    def request_stop(self):
        self._quit.set()

    def take(self):
        with self._out_lock:
            res, self._out = self._out, None
            self._notified = False
        return res

    def run(self):
        while not self._quit.is_set():
            item = self._slot.get(0.1)
            if item is None:
                continue
            try:
                res = self._fn(*item)
            except Exception:
                now = time.perf_counter()
                if now - self._last_err > 5.0:      # 限频打印, 避免刷屏
                    try:
                        traceback.print_exc()
                    except OSError:
                        pass          # 句柄无效 (WinError 6) 时静默跳过
                    self._last_err = now
                continue
            if res is None:
                continue
            self.meter.tick()
            with self._out_lock:
                self._out = res
                need_emit = not self._notified
                self._notified = True
            if need_emit:
                self.result_ready.emit()


# ==================== 驱动器 (兼容 QTimer 的 start/stop) ====================

class PipelineDriver(QObject):
    """
    用法:
        self._timer = PipelineDriver(self._make_source, self._process_core, parent=self)
        self._timer.result_ready.connect(self._on_pipeline_result)   # GUI 线程
        self._timer.failed.connect(self._on_pipeline_failed)         # GUI 线程
        self._timer.start() / self._timer.stop()                     # 同 QTimer

    source_factory() -> (read_fn, frame_interval, block_when_full) | None
    process_fn(frame, gray, idx) -> dict | None      (在处理线程执行)
    display_box: (w, h) 由 GUI 维护, 处理线程据此把画面缩放到显示尺寸
    """
    result_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, source_factory, process_fn, parent=None):
        super().__init__(parent)
        self._source_factory = source_factory
        self._process_fn = process_fn
        self.display_box = (0, 0)
        self._slot = None
        self._cap = None
        self._proc = None
        self._zombies = []

    # ── QTimer 兼容接口 ──
    def isActive(self) -> bool:
        return self._cap is not None

    def setInterval(self, ms):  # 兼容旧调用, 无意义
        pass

    def start(self):
        if self.isActive():
            return
        src = self._source_factory()
        if src is None:
            return
        read_fn, interval, block = src
        self._zombies = [t for t in self._zombies if t.isRunning()]
        slot = LatestSlot()
        cap = CaptureThread(read_fn, slot, frame_interval=interval,
                            block_when_full=block)
        proc = ProcessThread(self._process_fn, slot)
        cap.failed.connect(self._on_failed)
        proc.result_ready.connect(self._on_result)
        self._slot, self._cap, self._proc = slot, cap, proc
        proc.start()
        cap.start(QThread.Priority.HighPriority)

    def stop(self):
        cap, proc, slot = self._cap, self._proc, self._slot
        self._cap = self._proc = self._slot = None      # 先摘掉, 过期信号会被忽略
        if slot is not None:
            slot.close()
        for t in (cap, proc):
            if t is not None:
                t.request_stop()
        for t in (cap, proc):
            if t is not None and not t.wait(2000):      # 卡在设备调用里: 留存引用防止崩溃
                self._zombies.append(t)

    # ── 统计 ──
    @property
    def cap_fps(self) -> float:
        return self._cap.meter.fps if self._cap is not None else 0.0

    @property
    def proc_fps(self) -> float:
        return self._proc.meter.fps if self._proc is not None else 0.0

    # ── 内部槽 (GUI 线程) ──
    @pyqtSlot()
    def _on_result(self):
        proc = self._proc
        if proc is None or self.sender() is not proc:
            return
        res = proc.take()
        if res is not None:
            self.result_ready.emit(res)

    @pyqtSlot(str)
    def _on_failed(self, msg: str):
        if self._cap is None or self.sender() is not self._cap:
            return
        self.stop()
        self.failed.emit(msg)


# ==================== 自绘视频控件 ====================

class VideoLabel(QLabel):
    """
    视频画面控件 (替代原 QLabel + QPixmap.fromImage + scaled):
        · 直接把 QImage 画到屏幕, 省掉 QPixmap 转换和一次全帧缩放
        · 处理线程已缩放到显示尺寸时 1:1 贴图, 无任何重采样
        · 保留 clicked(x, y) (原始帧坐标) 与 set_frame_size() 接口
    """
    clicked = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            "background-color:#070b14; color:#4a5a6a;"
            "border:1px solid #1a3a5c;"
            "font-family:Consolas,monospace; font-size:12px;")
        self.setText("◈  等待信号  ◈\n\n请加载视频或连接相机")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self._img = None        # QImage
        self._keep = None       # 持有 numpy 缓冲区, 防止被回收
        self._frame_w = 0
        self._frame_h = 0
        self._rect = QRectF()

    def set_frame_size(self, w, h):
        self._frame_w = w
        self._frame_h = h

    def show_array(self, arr: np.ndarray, src_w: int = 0, src_h: int = 0):
        """arr: HxWx4 BGRA / HxWx3 BGR / HxW 灰度, uint8。src_*: 原始帧尺寸(点击换算用)"""
        if arr is None:
            return
        arr = np.ascontiguousarray(arr)
        h, w = arr.shape[:2]
        if arr.ndim == 2:
            fmt = QImage.Format.Format_Grayscale8
        elif arr.shape[2] == 4:
            fmt = QImage.Format.Format_RGB32
        else:
            fmt = QImage.Format.Format_BGR888
        self._img = QImage(arr.data, w, h, arr.strides[0], fmt)
        self._keep = arr
        self._frame_w = src_w or w
        self._frame_h = src_h or h
        if self.text():
            self.setText("")
        self.update()

    def paintEvent(self, ev):
        img = self._img
        if img is None:
            super().paintEvent(ev)
            return
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#070b14"))
        iw, ih = img.width(), img.height()
        lw, lh = self.width(), self.height()
        s = min(lw / iw, lh / ih)
        dw, dh = iw * s, ih * s
        x, y = (lw - dw) / 2.0, (lh - dh) / 2.0
        self._rect = QRectF(x, y, dw, dh)
        if abs(s - 1.0) < 0.01:
            p.drawImage(QPointF(round(x), round(y)), img)
        else:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            p.drawImage(self._rect, img)
        p.setPen(QColor("#1a3a5c"))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        p.end()

    def mousePressEvent(self, ev):
        if (ev.button() == Qt.MouseButton.LeftButton and self._img is not None
                and self._frame_w > 0 and self._rect.width() > 0):
            r = self._rect
            px = ev.position().x() - r.x()
            py = ev.position().y() - r.y()
            if 0 <= px < r.width() and 0 <= py < r.height():
                self.clicked.emit(px * self._frame_w / r.width(),
                                  py * self._frame_h / r.height())
        super().mousePressEvent(ev)