# -*- coding: utf-8 -*-
"""
第二个补丁: 给已打过 apply_patch.py 的 app.py 增加
  · KINEX_TRACK_MODE   mp(默认, 原流程) / marker(纯反光点) / hybrid(反光点+MediaPipe 救场)
  · KINEX_DEBUG=1      画面叠加调试信息

用法:
    python apply_patch2.py pose_analysis/app.py
运行:
    set KINEX_TRACK_MODE=mp        (Windows cmd; PowerShell 用 $env:KINEX_TRACK_MODE="mp")
    set KINEX_DEBUG=1
    python pose_analysis/app.py
"""
import re
import sys
from apply_patch import apply

EDITS = []


def edit(name, pattern, repl, flags=0):
    EDITS.append((name, pattern, repl, flags))


edit("模式/调试常量",
     r'(DEFAULT_VIDEO = resource_path\(\s*"pose_analysis", "videos", "demo.mp4"\)\n)',
     r'''\1
# 追踪模式(默认 marker): mp = MediaPipe + 反光点精化 (原流程) / marker = 纯反光点追踪
#           / hybrid = 反光点为主, 命中不足时 MediaPipe 救场
TRACK_MODE = os.environ.get("KINEX_TRACK_MODE", "marker").strip().lower()
DEBUG_TRACK = os.environ.get("KINEX_DEBUG", "0").strip() == "1"
''')

edit("主路径按模式分支",
     r"                # 反光点追踪为主 \(快, 像素级精度\)\n"
     r"                pixels = self\._fallback_track\(gray, dt\)\n",
     '''                if TRACK_MODE == "mp":
                    # 原流程: MediaPipe + 反光点精化, 失败才用反光点追踪
                    pixels, _src = self._mp_tracker.track_with_refine(frame, gray)
                    if not pixels:
                        pixels = self._fallback_track(gray, dt)
                    self._last_matched = {int(j) for j, s in _src.items()
                                          if s == 'marker'}
                else:
                    pixels = self._fallback_track(gray, dt)
''')

edit("救场仅 hybrid",
     r"if self\._weak_cnt >= 3:",
     'if TRACK_MODE == "hybrid" and self._weak_cnt >= 3:')

edit("调用调试叠加",
     r"(vis = draw_visual\(frame, pixels, angles, ranges,\n"
     r"\s+missing=missing, frame_idx=idx\)\n)",
     r'''\1        if DEBUG_TRACK:
            self._draw_debug(vis, pixels)
''')

edit("调试叠加方法",
     r"(    def _on_pipeline_result\(self, r\):)",
     r'''    def _draw_debug(self, vis, pixels):
        """调试叠加: 灰圈=候选反光点 黄十字+细圆=预测位置与搜索门
        绿圈=本帧真正命中 红圈=外推/MediaPipe 位置"""
        d = self._associator.debug or {}
        for (x, y, r) in d.get("blobs", []):
            cv2.circle(vis, (int(x), int(y)), max(2, int(r)), (160, 160, 160), 1)
        gates = d.get("gate", {})
        for j, (px, py) in d.get("pred", {}).items():
            p = (int(px), int(py))
            cv2.drawMarker(vis, p, (0, 255, 255), cv2.MARKER_CROSS, 10, 1)
            cv2.circle(vis, p, int(gates.get(j, 0)), (0, 120, 120), 1)
            cv2.putText(vis, str(j), (p[0] + 6, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        for j, (x, y) in pixels.items():
            hit = int(j) in self._last_matched
            cv2.circle(vis, (int(x), int(y)), 7,
                       (0, 255, 0) if hit else (0, 0, 255), 2)
        cv2.putText(vis, "MODE:%s HIT:%d/8 BLOBS:%d" % (
            TRACK_MODE, len(self._last_matched), len(d.get("blobs", []))),
            (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

\1''')


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    p = sys.argv[1]
    print(f"patch {p}")
    new = apply(p, EDITS)
    with open(p, encoding="utf-8") as f:
        old = f.read()
    with open(p + ".bak2", "w", encoding="utf-8") as f:
        f.write(old)
    with open(p, "w", encoding="utf-8") as f:
        f.write(new)
    print("完成, 备份为 .bak2")


if __name__ == "__main__":
    main()
