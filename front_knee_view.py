# -*- coding: utf-8 -*-
"""
正面骨骼图 (纯 OpenCV 渲染, 返回 BGR 图像): 肩 / 髋 / 膝 全部是活动的, 并带数据标注框。

    · 肩线、骨盆随实测的上下起伏 + 模拟的左右平移/倾斜一起动
    · 膝盖: 蓝 = 内翻 (向外)   绿 = 中立   红 = 外翻 (向内)
    · 实线 = 该侧有相机实测   虚线 = 由另一侧相机按曲柄相位估计
    · 竖直量 (起伏/高度) 为实测, 左右量 (平移/倾斜/偏摆) 为模型模拟 —— 标注框里逐项标明

直接预览:  python front_knee_view.py          (合成踩踏动画, 按 q 退出)
           python front_knee_view.py --one    (只接一侧相机, 另一腿估计)
"""
import math
import sys

import numpy as np
import cv2

FRONT_W, FRONT_H = 440, 460

C_OK = (0, 240, 128)         # 绿
C_HIGH = (0, 51, 255)        # 红 (外翻)
C_LOW = (255, 140, 0)        # 蓝 (内翻)
C_NODE = (255, 212, 0)
C_DIM = (110, 90, 40)
C_BG = (20, 11, 7)
C_BOX = (30, 22, 14)
C_BOX_EDGE = (92, 58, 26)
C_SIM = (53, 107, 255)       # 橙: 模拟
STATUS_COLOR = {"neutral": C_OK, "valgus": C_HIGH, "varus": C_LOW}
STATUS_TEXT = {"neutral": "NEUTRAL", "valgus": "VALGUS", "varus": "VARUS"}
FONT = cv2.FONT_HERSHEY_DUPLEX


def _dashed(img, p, q, col, th=2, seg=7):
    p, q = np.asarray(p, float), np.asarray(q, float)
    n = max(1, int(np.hypot(*(q - p)) / seg))
    for i in range(0, n, 2):
        a = p + (q - p) * i / n
        b = p + (q - p) * min(i + 1, n) / n
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), col, th)


def _tag(mode):
    return ("MEAS", C_OK) if mode == "measured" else ("SIM", C_SIM)


def _callout(img, anchor, x0, ycenter, lines, W):
    """白点 + 引线 + 圆角深色标注框; lines: [(text, color, scale)]"""
    h = 8 + sum(int(15 * sc) for _, _, sc in lines)
    top = int(np.clip(ycenter - h // 2, 4, img.shape[0] - h - 4))
    box = (x0, top, W - 6, top + h)
    cv2.line(img, anchor, (box[0], top + h // 2), (170, 150, 90), 1, cv2.LINE_AA)
    cv2.circle(img, anchor, 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, anchor, 6, (255, 255, 255), 1, cv2.LINE_AA)
    ov = img.copy()
    cv2.rectangle(ov, (box[0], box[1]), (box[2], box[3]), C_BOX, -1)
    cv2.addWeighted(ov, 0.85, img, 0.15, 0, img)
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), C_BOX_EDGE, 1)
    y = top + 4
    for text, col, sc in lines:
        y += int(15 * sc)
        cv2.putText(img, text, (box[0] + 6, y - 3), FONT, 0.36 * sc + 0.0, col, 1, cv2.LINE_AA)


def render_front_knee(state, size=(FRONT_W, FRONT_H)):
    W, H = size
    img = np.empty((H, W, 3), np.uint8)
    img[:] = C_BG
    for gy in range(0, H, 40):
        cv2.line(img, (0, gy), (W, gy), (30, 22, 14), 1)
    for gx in range(0, W, 40):
        cv2.line(img, (gx, 0), (gx, H), (30, 22, 14), 1)
    cv2.rectangle(img, (0, 0), (W - 1, H - 1), (92, 58, 26), 1)
    cv2.putText(img, "FRONT VIEW", (12, 22), FONT, 0.5, C_NODE, 1)

    legs = {s: state.get(s) for s in ("L", "R")} if state else {}
    if not any(legs.values()):
        cv2.putText(img, "AWAITING DATA", (W // 2 - 70, H // 2), FONT, 0.45, (120, 100, 0), 1)
        return img
    pel, sho, hands = state.get("pelvis"), state.get("shoulder"), state.get("hands")
    modes = {l["lateral_mode"] for l in legs.values() if l} | ({pel["mode"]} if pel else set())
    sim = "simulated" in modes
    cv2.putText(img, "LATERAL: SIM" if sim else "LATERAL: MEAS", (W - 150, 22), FONT, 0.38,
                C_SIM if sim else C_OK, 1)
    cv2.putText(img, "VERTICAL: MEAS", (W - 150, 38), FONT, 0.34, C_OK, 1)

    s = (H * 0.62) / 100.0                       # px / cm
    cx, y0 = 150, int(H * 0.30)

    def P(pt):
        return (int(cx + pt[0] * s), int(y0 + pt[1] * s))

    cv2.line(img, (cx, 6), (cx, H - 26), (60, 45, 25), 1)

    # ── 躯干 / 肩 / 手臂 / 头 ──
    if sho and pel:
        pc, sc = P(pel["center"]), P(sho["center"])
        cv2.line(img, pc, sc, C_DIM, 3, cv2.LINE_AA)                          # 脊柱
        cv2.circle(img, (sc[0], sc[1] - int(14 * s)), int(7.5 * s), C_DIM, 1, cv2.LINE_AA)   # 头
        cv2.line(img, sc, (sc[0], sc[1] - int(6 * s)), C_DIM, 2, cv2.LINE_AA)
        pl, pr = P(sho["pts"]["L"]), P(sho["pts"]["R"])
        if hands:
            for side, pt in (("L", pl), ("R", pr)):
                hx = int(cx + (1 if side == "L" else -1) * hands["w"] / 2 * s)
                hy = int(y0 + hands["y"] * s)
                el = ((pt[0] + hx) // 2 + (6 if side == "L" else -6), (pt[1] + hy) // 2)
                cv2.line(img, pt, el, C_DIM, 2, cv2.LINE_AA)
                cv2.line(img, el, (hx, hy), C_DIM, 2, cv2.LINE_AA)
                cv2.circle(img, (hx, hy), 4, C_DIM, -1, cv2.LINE_AA)
        cv2.line(img, pl, pr, C_NODE, 3, cv2.LINE_AA)                         # 肩线
        for q in (pl, pr):
            cv2.circle(img, q, 5, C_NODE, -1, cv2.LINE_AA)

    # ── 骨盆 ──
    hl, hr = P(pel["hips"]["L"]), P(pel["hips"]["R"])
    cv2.line(img, hl, hr, C_NODE, 4, cv2.LINE_AA)

    # ── 腿 ──
    knees = []
    for side in ("R", "L"):
        leg = legs.get(side)
        if leg is None:
            continue
        est = leg["source"] == "estimated"
        col = STATUS_COLOR[leg["status"]]
        line_col = tuple(int(c * 0.6) for c in col) if est else col
        hip, knee, ankle = P(leg["hip"]), P(leg["knee"]), P(leg["ankle"])
        draw = (lambda a, b, c_: _dashed(img, a, b, c_, 3)) if est else \
               (lambda a, b, c_: cv2.line(img, a, b, c_, 3, cv2.LINE_AA))
        draw(hip, knee, line_col)
        draw(knee, ankle, line_col)
        cv2.line(img, (ankle[0] - int(4 * s), ankle[1] + int(2 * s)),
                 (ankle[0] + int(4 * s), ankle[1] + int(2 * s)), line_col, 3)
        cv2.circle(img, hip, 6, C_NODE, -1, cv2.LINE_AA)
        cv2.circle(img, ankle, 5, C_NODE, -1, cv2.LINE_AA)
        glow = img.copy()
        cv2.circle(glow, knee, 15, col, -1, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.3, img, 0.7, 0, img)
        cv2.circle(img, knee, 9, col, 2, cv2.LINE_AA)
        cv2.circle(img, knee, 3, C_NODE, -1, cv2.LINE_AA)
        cv2.putText(img, side + ("-EST" if est else "-MEAS"),
                    (ankle[0] - 22, min(H - 28, ankle[1] + int(14 * s))), FONT, 0.38, line_col, 1)
        knees.append((side, leg, knee))

    # ── 数据标注框 (白点 + 引线, 与参考图一致) ──
    bx = 262
    if sho:
        tg, tc = _tag(sho["mode"])
        _callout(img, P(sho["center"]), bx, P(sho["center"])[1], [
            ("SHOULDER SWAY  [" + tg + "]", tc, 1.0),
            (f"{sho['shift_cm']:+.1f} cm   tilt {sho['tilt_deg']:+.1f} deg", (235, 235, 200), 1.0),
        ], W)
    tg, tc = _tag(pel["mode"])
    _callout(img, P(pel["center"]), bx, P(pel["center"])[1] + 20, [
        ("HIP SWAY  [" + tg + "]", tc, 1.0),
        (f"{pel['shift_cm']:+.1f} cm   tilt {pel['tilt_deg']:+.1f} deg", (235, 235, 200), 1.0),
        (f"bob {-pel['bob_cm']:+.1f} cm  [MEAS]", C_OK, 1.0),
    ], W)
    if knees:
        ky = int(np.mean([k[2][1] for k in knees]))
        lines = [("KNEE SWAY  [" + _tag(knees[0][1]["lateral_mode"])[0] + "]",
                  _tag(knees[0][1]["lateral_mode"])[1], 1.0)]
        for side, leg, _ in sorted(knees, key=lambda k: k[0]):
            lines.append((f"{side} {leg['dev_deg']:+.1f} deg {STATUS_TEXT[leg['status']]}"
                          + (" est" if leg["source"] == "estimated" else ""),
                          STATUS_COLOR[leg["status"]], 1.0))
        _callout(img, knees[0][2], bx, ky, lines, W)

    for i, (txt, col) in enumerate((("VARUS", C_LOW), ("NEUTRAL", C_OK), ("VALGUS", C_HIGH))):
        x0 = 12 + i * 80
        cv2.rectangle(img, (x0, H - 18), (x0 + 9, H - 9), col, -1)
        cv2.putText(img, txt, (x0 + 13, H - 9), FONT, 0.32, col, 1)
    return img


# ───────────────────────── 预览 (合成踩踏) ─────────────────────────

def _synthetic_pixels(phase, ox=300.0, oy=250.0, thigh=114.0, shank=116.0, crank=45.0):
    """侧视图合成数据: 曲柄中心在髋下方, 踝点绕曲柄转, 膝由两段骨长解出"""
    oy = oy + 3.0 * math.sin(2 * phase)                  # 髋随蹬踏上下起伏
    hip = np.array([ox, oy])
    cc = np.array([ox - 10.0, oy + 175.0])        # 曲柄中心: 脚在最低点时腿未完全伸直
    ankle = cc + crank * np.array([math.cos(phase), math.sin(phase)])
    d = np.linalg.norm(ankle - hip)
    a = (thigh ** 2 - shank ** 2 + d ** 2) / (2 * d)
    h = math.sqrt(max(thigh ** 2 - a ** 2, 0.0))
    u = (ankle - hip) / d
    knee = hip + a * u + h * np.array([u[1], -u[0]])
    heel, toe = ankle + [-12, 22], ankle + [41, 36]
    return {1: (hip[0] + 245, 280.0), 3: (hip[0] + 140, 175.0 + 2 * math.sin(2 * phase)),
            4: tuple(hip), 5: tuple(knee), 6: tuple(ankle), 7: tuple(heel), 8: tuple(toe)}


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
        cv2.imshow("front view", render_front_knee(m.front_state(t)))
        t += 1 / 30.0
        if cv2.waitKey(33) & 0xFF == ord("q"):
            break
