# -*- coding: utf-8 -*-
"""离线评测 (无需 Qt/MediaPipe): 用 demo.mp4 逐帧跑 MarkerAssociator, 检查
每个输出点是否落在"强反光点"上 (对比度≈标定值、半径≈标定值)。
用法 (仓库根目录):  python eval_tracking.py pose_analysis/tracking_engine.py videos/demo.mp4
手动标定点 APPROX 是 demo.mp4 首帧 (960 宽坐标) 的 8 个点; 换视频需改它。
"""
import sys, types, importlib.util, numpy as np, cv2
CFG = types.SimpleNamespace(min_radius=3, max_radius=20, roi_radius=60, track_max_move=35.0,
    track_min_circularity=0.3, lower_gate_scale=3.0, use_tophat=True, tophat_floor=20, bone_weight=1.0)
APPROX = {1:(631,279),2:(575,234),3:(526,175),4:(382,250),5:(398,363),6:(357,472),7:(345,494),8:(397,508)}
def load(p):
    s = importlib.util.spec_from_file_location("te", p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def prep(f):
    if f.shape[1] > 960: f = cv2.resize(f, (960, int(f.shape[0]*960/f.shape[1])), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
def main(te_path, video):
    te = load(te_path); cap = cv2.VideoCapture(video); ok, f = cap.read(); g0 = prep(f)
    H, W = g0.shape; a = te.MarkerAssociator(CFG); bl = a.extract_blobs(g0, [(0,0,W,H)])
    cp = {}
    for j,(x,y) in APPROX.items():
        c = min(bl, key=lambda b: np.hypot(b.x-x, b.y-y)); cp[j] = (c.x, c.y)
    a.set_calibration(cp, 0.0, g0); mo = te.MotionPredictor(); mo.seed(cp)
    lost = {j:0 for j in cp}; last = dict(cp); bad = 0; n = 0; hits = []; i = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        i += 1; g = prep(f)
        m = a.associate(g, mo, list(cp), last, lost, 1.0)
        safe = {j:p for j,p in m.items() if np.hypot(p[0]-cp[j][0], p[1]-cp[j][1]) <= a._calib_limit(j)}
        mo.update(safe, 1.0); px = {}
        for j in cp:
            if j in safe: px[j] = safe[j]; lost[j] = 0
            else:
                lost[j] += 1
                if mo.lost_age(j) > 8 or lost[j] > 5: mo.reseed(j, *cp[j]); lost[j] = 4
                px[j] = mo.predict(j, 1.0)
        last = px; hits.append(len(safe))
        if i % 5 == 0:
            strong = [b for b in a.extract_blobs(g, [(0,0,W,H)]) if b.contrast >= 0.85*a.contrast_ref and b.r >= 0.6*a.r_ref]
            for j,(x,y) in px.items():
                n += 1; bad += min(np.hypot(b.x-x, b.y-y) for b in strong) > 6
    print("frames", i, "mean hits %.2f/8" % np.mean(hits), "| off-marker outputs %d/%d" % (bad, n))
if __name__ == "__main__": main(sys.argv[1], sys.argv[2])
