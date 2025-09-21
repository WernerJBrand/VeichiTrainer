import os, json
import numpy as np
from PIL import Image
import cv2
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsLineItem,
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QDoubleSpinBox, QMessageBox
)
from lib.aruco_utils import detect_aruco_scale  # runs because this script lives in app/

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
        self.shapes=[]; self._item=None; self._start=None

    def load(self, path):
        self.scene.clear(); self.pix=None; self.shapes=[]; self.px_per_mm=None
        self.img_path = path
        img = Image.open(path).convert("RGB")
        self.img_np = np.array(img)[:, :, ::-1]  # BGR
        self.pix = QGraphicsPixmapItem(QtGui.QPixmap.fromImage(qimage_from_pil(img)))
        self.scene.addItem(self.pix); self.fitInView(self.pix, Qt.KeepAspectRatio)

    def wheelEvent(self,e): self.scale(1.2,1.2) if e.angleDelta().y()>0 else self.scale(1/1.2,1/1.2)

    def mousePressEvent(self,e):
        if e.button()==Qt.LeftButton and self.pix:
            p=self.mapToScene(e.pos())
            if self.tool=='box':
                self._item=QGraphicsRectItem(QRectF(p,p)); self._item.setPen(QtGui.QPen(Qt.green,2))
            else:
                self._item=QGraphicsLineItem(QtCore.QLineF(p,p)); self._item.setPen(QtGui.QPen(Qt.yellow,2))
            self.scene.addItem(self._item); self._start=p
        else: super().mousePressEvent(e)

    def mouseMoveEvent(self,e):
        if self._item and self._start:
            p=self.mapToScene(e.pos())
            if isinstance(self._item,QGraphicsRectItem):
                self._item.setRect(QRectF(self._start,p).normalized())
            else:
                self._item.setLine(QtCore.QLineF(self._start,p))
        else: super().mouseMoveEvent(e)

    def mouseReleaseEvent(self,e):
        if e.button()==Qt.LeftButton and self._item:
            if isinstance(self._item,QGraphicsRectItem):
                r=self._item.rect()
                if r.width()>5 and r.height()>5:
                    self.shapes.append({"type":"box","label":self.current_label,
                                        "points":[r.x(),r.y(),r.width(),r.height()]})
                    self._item.setPen(QtGui.QPen(Qt.cyan,2))
                else: self.scene.removeItem(self._item)
            else:
                l=self._item.line()
                if l.length()>5:
                    mm = (l.length()/self.px_per_mm) if self.px_per_mm else None
                    self.shapes.append({"type":"line","label":self.current_label,
                                        "points":[l.x1(),l.y1(),l.x2(),l.y2()],
                                        "mm_value":mm})
                    self._item.setPen(QtGui.QPen(Qt.magenta,2))
                else: self.scene.removeItem(self._item)
            self._item=None; self._start=None
        else: super().mouseReleaseEvent(e)

    def detect_scale(self, marker_mm):
        if self.img_np is None: return None
        v, corners, ids = detect_aruco_scale(self.img_np, marker_mm)
        if not v: return None
        self.px_per_mm = v
        # visual feedback
        for c in corners:
            pts=[QPointF(float(x),float(y)) for (x,y) in c[0]]
            for i in range(4):
                li=QGraphicsLineItem(QtCore.QLineF(pts[i],pts[(i+1)%4]))
                li.setPen(QtGui.QPen(Qt.red,2)); self.scene.addItem(li)
        return v

    def to_json(self):
        return {"image_path": self.img_path, "px_per_mm": self.px_per_mm, "shapes": self.shapes}

class Main(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle(APP_TITLE); self.resize(1200,800)
        labels_cfg = json.load(open(os.path.join(os.path.dirname(__file__),"config","labels.json")))
        labels = labels_cfg["classes"]

        self.canvas = Canvas(labels)
        self.cmb = QComboBox(); [self.cmb.addItem(x) for x in labels]
        self.cmb.currentTextChanged.connect(lambda s: setattr(self.canvas,"current_label",s))
        self.mm = QDoubleSpinBox(); self.mm.setRange(1,500); self.mm.setValue(labels_cfg.get("default_marker_size_mm",60.0))
        self.lbl = QLabel("px/mm: —")

        bOpen=QPushButton("Open image");    bOpen.clicked.connect(self._open)
        bBox =QPushButton("Tool: Box");     bBox.clicked.connect(lambda: setattr(self.canvas,"tool","box"))
        bLine=QPushButton("Tool: Measure"); bLine.clicked.connect(lambda: setattr(self.canvas,"tool","line"))
        bScale=QPushButton("Detect Scale"); bScale.clicked.connect(self._scale)
        bSave=QPushButton("Save JSON");     bSave.clicked.connect(self._save)

        left=QVBoxLayout()
        left.addWidget(bOpen); left.addWidget(QLabel("Label")); left.addWidget(self.cmb)
        left.addWidget(bBox); left.addWidget(bLine)
        left.addWidget(QLabel("Marker side (mm)")); left.addWidget(self.mm)
        left.addWidget(bScale); left.addWidget(self.lbl); left.addStretch()
        wrapL=QWidget(); wrapL.setLayout(left)

        layout=QHBoxLayout(); layout.addWidget(wrapL); layout.addWidget(self.canvas,1)
        central=QWidget(); central.setLayout(layout); self.setCentralWidget(central)

    def _open(self):
        p,_=QFileDialog.getOpenFileName(self,"Open image","","Images (*.jpg *.jpeg *.png)")
        if p: self.canvas.load(p)

    def _scale(self):
        v=self.canvas.detect_scale(float(self.mm.value()))
        self.lbl.setText(f"px/mm: {v:.3f}" if v else "px/mm: —")

    def _save(self):
        if not self.canvas.img_path: return
        out=self.canvas.img_path + ".annotations.json"
        with open(out,"w") as f: json.dump(self.canvas.to_json(),f,indent=2)
        QMessageBox.information(self,"Saved",out)

if __name__=="__main__":
    app = QApplication([])
    w = Main(); w.show()
    app.exec()
