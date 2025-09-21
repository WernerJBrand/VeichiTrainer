import os, json
import numpy as np
from PIL import Image
import cv2
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsLineItem, QGraphicsSimpleTextItem,
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QDoubleSpinBox, QMessageBox, QToolButton, QButtonGroup
)
from lib.aruco_utils import detect_aruco_scale  # script is inside app/, so this works

APP_TITLE = "Veichi Annotator (M1)"

def qimage_from_pil(img):
    return QtGui.QImage(img.tobytes(), img.size[0], img.size[1], img.size[0]*3,
                        QtGui.QImage.Format_RGB888)

class Canvas(QGraphicsView):
    def __init__(self, labels):
        super().__init__()
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.scene = QGraphicsScene(self); self.setScene(self.scene)
        self.pix=None; self.img_path=None; self.img_np=None

        self.labels = labels; self.current_label = labels[0] if labels else ""
        self.tool='box'; self.px_per_mm=None

        self.shapes=[]              # serialized data
        self._item=None             # the temp rect/line while dragging
        self._start=None
        self._measure_text=None     # live mm overlay while dragging a line
        self._graphics_stack=[]     # for Undo: list of [items...] just created

    # --- IO ---
    def load(self, path):
        self.scene.clear(); self.pix=None
        self.shapes=[]; self._graphics_stack.clear()
        self.px_per_mm=None; self._item=None; self._start=None; self._measure_text=None

        self.img_path = path
        img = Image.open(path).convert("RGB")
        self.img_np = np.array(img)[:, :, ::-1]  # BGR for cv2
        self.pix = QGraphicsPixmapItem(QtGui.QPixmap.fromImage(qimage_from_pil(img)))
        self.scene.addItem(self.pix)
        self.fitInView(self.pix, Qt.KeepAspectRatio)

    def to_json(self):
        return {"image_path": self.img_path, "px_per_mm": self.px_per_mm, "shapes": self.shapes}

    # --- View helpers ---
    def wheelEvent(self,e): self.scale(1.2,1.2) if e.angleDelta().y()>0 else self.scale(1/1.2,1/1.2)

    # --- Mouse drawing ---
    def mousePressEvent(self,e):
        if e.button()==Qt.LeftButton and self.pix:
            p=self.mapToScene(e.pos())
            if self.tool=='box':
                self._item=QGraphicsRectItem(QRectF(p,p)); self._item.setPen(QtGui.QPen(Qt.green,2))
            else:
                self._item=QGraphicsLineItem(QtCore.QLineF(p,p)); self._item.setPen(QtGui.QPen(Qt.yellow,2))
                self._measure_text = QGraphicsSimpleTextItem("")  # live mm label
                self._measure_text.setBrush(QtGui.QBrush(Qt.black))
                self.scene.addItem(self._measure_text)
            self.scene.addItem(self._item)
            self._start=p
        else:
            super().mousePressEvent(e)

    def mouseMoveEvent(self,e):
        if self._item and self._start:
            p=self.mapToScene(e.pos())
            if isinstance(self._item,QGraphicsRectItem):
                self._item.setRect(QRectF(self._start,p).normalized())
            else:
                self._item.setLine(QtCore.QLineF(self._start,p))
                # live length label
                l=self._item.line()
                mid=QPointF((l.x1()+l.x2())/2.0,(l.y1()+l.y2())/2.0)
                if self.px_per_mm: txt=f"{l.length()/self.px_per_mm:.1f} mm"
                else: txt=f"{l.length():.0f} px"
                if self._measure_text:
                    self._measure_text.setText(txt)
                    self._measure_text.setPos(mid)
        else:
            super().mouseMoveEvent(e)

    def mouseReleaseEvent(self,e):
        if e.button()==Qt.LeftButton and self._item:
            created=[self._item]  # items to keep for undo
            if isinstance(self._item,QGraphicsRectItem):
                r=self._item.rect()
                if r.width()>5 and r.height()>5:
                    self.shapes.append({"type":"box","label":self.current_label,
                                        "points":[r.x(),r.y(),r.width(),r.height()]})
                    self._item.setPen(QtGui.QPen(Qt.cyan,2))
                else:
                    self.scene.removeItem(self._item); created=[]
            else:
