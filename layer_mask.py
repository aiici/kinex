# -*- coding: utf-8 -*-
"""图层蒙版: 把一张图作为视频画面的蒙版/叠加层 (HUD 与骨架画在它上面)

环境变量:
    KINEX_MASK_MODE     off(默认) / mask / screen
        mask   : 亮度蒙版 — 蒙版亮处显示视频, 暗处压暗 (像 Photoshop 的图层蒙版)
        screen : 滤色叠加 — 蒙版图的发光部分叠加到视频上
    KINEX_MASK_OPACITY  强度 0~1, 默认 0.6
    KINEX_MASK_INVERT   1 = 反相蒙版 (只对 mask 模式有意义)
    KINEX_MASK_FIT      cover(默认, 铺满并裁切) / contain(完整显示, 四周补黑)
    KINEX_MASK          蒙版图路径, 默认 pose_analysis/assets/mask.jpg
"""
import os
import numpy as np
import cv2

_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "assets", "mask.jpg")
MODE = os.environ.get("KINEX_MASK_MODE", "off").strip().lower()
OPACITY = float(os.environ.get("KINEX_MASK_OPACITY", "0.6"))
INVERT = os.environ.get("KINEX_MASK_INVERT", "0").strip() == "1"
PATH = os.environ.get("KINEX_MASK", _DEFAULT)
FIT = os.environ.get("KINEX_MASK_FIT", "cover").strip().lower()

_cache = {}


def _build(h, w, mode, opacity, invert, path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    ih, iw = img.shape[:2]
    if FIT == "contain":                           # 完整显示, 居中, 四周补黑
        s = min(w / iw, h / ih)
        img = cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))),
                         interpolation=cv2.INTER_AREA)
        canvas = np.zeros((h, w, 3), np.uint8)
        y0, x0 = (h - img.shape[0]) // 2, (w - img.shape[1]) // 2
        canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
        img = canvas
    else:                                          # cover: 等比放大后居中裁切
        s = max(w / iw, h / ih)
        img = cv2.resize(img, (int(iw * s) + 1, int(ih * s) + 1),
                         interpolation=cv2.INTER_AREA)
        y0, x0 = (img.shape[0] - h) // 2, (img.shape[1] - w) // 2
        img = np.ascontiguousarray(img[y0:y0 + h, x0:x0 + w])
    if mode == "screen":
        return (img.astype(np.float32) * opacity).clip(0, 255).astype(np.uint8)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g = np.clip(g * 2.2, 0.0, 1.0)                 # 拉亮, 让暗部细节也能透出
    if invert:
        g = 1.0 - g
    alpha = (1.0 - opacity) + opacity * g
    return cv2.cvtColor((alpha * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def apply_layer_mask(frame, mode=None, opacity=None, invert=None, path=None):
    """返回应用蒙版后的新画面 (不修改入参); 模式为 off 或蒙版图缺失时原样返回"""
    mode = (mode or MODE)
    if mode not in ("mask", "screen"):
        return frame
    opacity = OPACITY if opacity is None else opacity
    invert = INVERT if invert is None else invert
    path = path or PATH
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = frame.shape[:2]
    key = (mode, opacity, invert, path, h, w)
    layer = _cache.get(key, 0)
    if layer == 0:
        layer = _build(h, w, mode, opacity, invert, path)
        _cache.clear()                             # 只缓存当前分辨率的一份
        _cache[key] = layer
    if layer is None:
        return frame
    if mode == "screen":
        return cv2.add(frame, layer)
    return cv2.multiply(frame, layer, scale=1.0 / 255.0)
