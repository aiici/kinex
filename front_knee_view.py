# -*- coding: utf-8 -*-
"""
正面膝盖骨骼图 (纯 OpenCV 渲染, 返回 BGR 图像)

    蓝 = 内翻 (膝盖向外)    绿 = 中立    红 = 外翻 (膝盖向内)
    实线 = 该侧有相机实测    虚线 = 由另一侧相机按曲柄相位估计
    左右偏摆为模型模拟时, 画面标注 SIMULATED

直接预览:  python front_knee_view.py          (合成踩踏动画, 按 q 退出)
           python front_knee_view.py --one    (只接一侧相机, 另一腿估计)
"""
import math
import sys

import numpy as np
import cv2

FRONT_W, FRONT_H = 340, 460

C_OK = (0, 240, 128)         # 绿
C_HIGH = (0, 51, 255)        # 红 (外翻)
C_LOW = (255, 140, 0)        # 蓝 (内翻)
C_NODE = (255, 212, 0)
C_DIM = (110, 90, 40)
C_BG = (20, 11, 7)
STATUS_COLOR = {"neutral": C_OK, "valgus": C_HIGH, "varus": C_LOW}
STATUS_TEXT = {"neutral": "NEUTRAL", "valgus": "VALGUS", "varus": "VARUS"}


def _dashed(img, p, q, col, th=2, seg=7):
    p, q = np.asarray(p, float), np.asarray(q, float)
    n = max(1, int(np.hypot(*(q - p)) / seg))
    for i in range(0, n, 2):
        a = p + (q - p) * i / n
        b = p + (q - p) * min(i + 1, n) / n
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), col, th)


def render_front_knee(state, size=(FRONT_W, FRONT_H)):
    W, H = size
    img = np.empty((H, W, 3), np.uint8)
    img[:] = C_BG
    for gy in range(0, H, 40):
        cv2.line(img, (0, gy), (W, gy), (30, 22, 14), 1)
    for gx in range(0, W, 40):
        cv2.line(img, (gx, 0), (gx, H), (30, 22, 14), 1)
    cv2.rectangle(img, (0, 0), (W - 1, H - 1), (92, 58, 26), 1)
    cv2.putText(img, "FRONT KNEE", (12, 22), cv2.FONT_HERSHEY_DUPLEX, 0.5, C_NODE, 1)
    legs = {s: state.get(s) for s in ("L", "R")} if state else {}
    modes = {l["lateral_mode"] for l in legs.values() if l}
    sim = (not modes) or "simulated" in modes          # 只要有模拟成分就标 SIMULATED
    cv2.putText(img, "SIMULATED" if sim else "MEASURED", (W - 100, 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.4, (53, 107, 255) if sim else C_OK, 1)
    if not any(legs.values()):
        cv2.putText(img, "AWAITING DATA", (W // 2 - 70, H // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (120, 100, 0), 1)
        return img

    g = state["geom"]
    s = (H * 0.72) / 100.0                       # px / cm
    cx, y0 = W // 2, int(H * 0.16)

    def P(pt):
        return (int(cx + pt[0] * s), int(y0 + pt[1] * s))

    cv2.line(img, (cx, y0 - 30), (cx, H - 30), (60, 45, 25), 1)       # 中线
    hw = g["hip_w"] / 2.0
    cv2.line(img, P((-hw, 0)), P((hw, 0)), C_NODE, 3)                  # 骨盆
    cv2.line(img, P((0, 0)), P((0, -20)), C_DIM, 2)                    # 躯干示意

    for side in ("R", "L"):
        leg = legs.get(side)
        if leg is None:
            continue
        est = leg["source"] == "estimated"
        col = STATUS_COLOR[leg["status"]]
        line_col = tuple(int(c * 0.6) for c in col) if est else col
        hip, knee, ankle = P(leg["hip"]), P(leg["knee"]), P(leg["ankle"])
        draw = (lambda a, b, c_: _dashed(img, a, b, c_, 3)) if est else \
               (lambda a, b, c_: cv2.line(img, a, b, c_, 3))
        draw(hip, knee, line_col)
        draw(knee, ankle, line_col)
        cv2.line(img, (ankle[0] - int(4 * s), ankle[1] + int(2 * s)),
                 (ankle[0] + int(4 * s), ankle[1] + int(2 * s)), line_col, 3)
        cv2.circle(img, hip, 6, C_NODE, -1)
        cv2.circle(img, ankle, 5, C_NODE, -1)
        # 膝盖: 半透明光晕 + 状态环
        glow = img.copy()
        cv2.circle(glow, knee, 15, col, -1)
        cv2.addWeighted(glow, 0.3, img, 0.7, 0, img)
        cv2.circle(img, knee, 9, col, 2)
        cv2.circle(img, knee, 3, C_NODE, -1)
        # 读数: 画在膝盖外侧
        out_dir = -1 if side == "R" else 1
        tx = knee[0] + out_dir * 22 - (62 if out_dir < 0 else 0)
        cv2.putText(img, f"{side} {leg['dev_deg']:+.1f}", (tx, knee[1] - 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42, col, 1)
        cv2.putText(img, STATUS_TEXT[leg["status"]], (tx, knee[1] + 13),
                    cv2.FONT_HERSHEY_DUPLEX, 0.32, col, 1)
        cv2.putText(img, side + ("-EST" if est else "-MEAS"),
                    (ankle[0] - 22, min(H - 28, ankle[1] + int(14 * s))),
                    cv2.FONT_HERSHEY_DUPLEX, 0.38, line_col, 1)

    for i, (txt, col) in enumerate((("VARUS", C_LOW), ("NEUTRAL", C_OK),
                                    ("VALGUS", C_HIGH))):               # 图例
        x0 = 12 + i * 80
        cv2.rectangle(img, (x0, H - 18), (x0 + 9, H - 9), col, -1)
        cv2.putText(img, txt, (x0 + 13, H - 9), cv2.FONT_HERSHEY_DUPLEX, 0.32, col, 1)
    return img


# ───────────────────────── 预览 (合成踩踏) ─────────────────────────

def _synthetic_pixels(phase, ox=300.0, oy=250.0, thigh=114.0, shank=116.0, crank=60.0):
    """侧视图合成数据: 曲柄中心在髋下方, 踝点绕曲柄转, 膝由两段骨长解出"""
    hip = np.array([ox, oy])
    cc = hip + np.array([-20.0, 220.0])
    ankle = cc + crank * np.array([math.cos(phase), math.sin(phase)])
    d = np.linalg.norm(ankle - hip)
    a = (thigh ** 2 - shank ** 2 + d ** 2) / (2 * d)
    h = math.sqrt(max(thigh ** 2 - a ** 2, 0.0))
    u = (ankle - hip) / d
    knee = hip + a * u + h * np.array([u[1], -u[0]])
    heel, toe = ankle + [-12, 22], ankle + [41, 36]
    return {1: (hip[0] + 245, hip[1] + 30), 4: tuple(hip), 5: tuple(knee),
            6: tuple(ankle), 7: tuple(heel), 8: tuple(toe)}


if __name__ == "__main__":
    from rider_model import RiderModel
    one = "--one" in sys.argv
    m = RiderModel()
    t = 0.0
    while True:
        ph = t * 2 * math.pi * 1.0
        m.push(_synthetic_pixels(ph), t, "R")
        if not one:
            m.push(_synthetic_pixels(ph + math.pi), t, "L")
        cv2.imshow("front knee", render_front_knee(m.front_state(t)))
        t += 1 / 30.0
        if cv2.waitKey(33) & 0xFF == ord("q"):
            break
