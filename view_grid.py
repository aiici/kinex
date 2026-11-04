# -*- coding: utf-8 -*-
"""
三联画面容器:   [ 左侧相机 | 正面膝盖 | 右侧相机 ]

· 正面膝盖图始终在中间, 不随相机数量增减
· 某一侧没接相机时, 该位自动收缩成一条窄占位栏 (点击可触发"连接相机"),
  已连接的相机自动获得大部分宽度; 第二台相机接入时占位栏展开, 无需重新插入控件
用法:
    grid = ViewGrid(left_widget, front_widget, right_widget)
    grid.set_active("R", True)          # 右侧相机接入
    grid.slot_clicked.connect(lambda side: ...)
"""
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QStackedLayout, QWidget


class _Slot(QWidget):
    clicked = pyqtSignal(str)

    def __init__(self, side, video_widget, parent=None):
        super().__init__(parent)
        self.side = side
        self.stack = QStackedLayout(self)
        self.stack.setContentsMargins(0, 0, 0, 0)
        name = "左" if side == "L" else "右"
        self.hint = QLabel(f"◈ {name}侧相机\n未连接\n\n点击连接")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint.setStyleSheet(
            "background:#0d1520;color:#4a5a6a;border:1px dashed #1a3a5c;"
            "font-family:Consolas,monospace;font-size:12px;")
        self.stack.addWidget(video_widget)     # 0: 画面
        self.stack.addWidget(self.hint)        # 1: 占位
        self.stack.setCurrentIndex(1)

    def mousePressEvent(self, ev):
        if self.stack.currentIndex() == 1:
            self.clicked.emit(self.side)
        super().mousePressEvent(ev)

    def set_active(self, on):
        self.stack.setCurrentIndex(0 if on else 1)


class ViewGrid(QWidget):
    slot_clicked = pyqtSignal(str)

    def __init__(self, left, front, right, parent=None):
        super().__init__(parent)
        self._lay = QHBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(2)
        self._slots = {"L": _Slot("L", left), "R": _Slot("R", right)}
        for s in self._slots.values():
            s.clicked.connect(self.slot_clicked)
        self._front = front
        self._lay.addWidget(self._slots["L"])
        self._lay.addWidget(front)
        self._lay.addWidget(self._slots["R"])
        self._active = {"L": False, "R": False}
        self._relayout()

    def set_active(self, side, on):
        self._active[side] = bool(on)
        self._slots[side].set_active(on)
        self._relayout()

    def _relayout(self):
        """已连接 → 占主要宽度; 未连接 → 窄占位栏; 全部未连接时占位栏平分"""
        none = not any(self._active.values())
        for i, side in ((0, "L"), (2, "R")):
            on = self._active[side]
            self._lay.setStretch(i, 6 if on else (3 if none else 1))
            self._slots[side].setMinimumWidth(640 if on else 110)
        self._lay.setStretch(1, 0)               # 正面图固定宽度
