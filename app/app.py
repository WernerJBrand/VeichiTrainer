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
                l=self._item.line()
                if l.length()>5:
                    mm = (l.length()/self.px_per_mm) if self.px_per_mm else None
                    self.shapes.append({"type":"line","label":self.current_label,
                                        "points":[l.x1(),l.y1(),l.x2(),l.y2()],
                                        "mm_value":mm})
                    self._item.setPen(QtGui.QPen(Qt.magenta,2))
                    if self._measure_text:
                        self._measure_text.setBrush(QtGui.QBrush(Qt.darkMagenta))
                        created.append(self._measure_text)
                        self._measure_text=None
                else:
                    self.scene.removeItem(self._item)
                    if self._measure_text: self.scene.removeItem(self._measure_text)
                    created=[]
                    self._measure_text=None
            if created: self._graphics_stack.append(created)
            self._item=None; self._start=None
        else:
            super().mouseReleaseEvent(e)

    # --- Tools / features ---
    def set_tool(self, name:str): self.tool=name

    def undo(self):
        if not self._graphics_stack or not self.shapes: return
        items = self._graphics_stack.pop()
        for it in items: self.scene.removeItem(it)
        self.shapes.pop()

    def detect_scale(self, marker_mm):
        if self.img_np is None: return None
        v, corners, ids = detect_aruco_scale(self.img_np, marker_mm)
        if not v: return None
        self.px_per_mm = v
        # draw marker borders for feedback
        for c in corners:
            pts=[QPointF(float(x),float(y)) for (x,y) in c[0]]
            for i in range(4):
                li=QGraphicsLineItem(QtCore.QLineF(pts[i],pts[(i+1)%4]))
                li.setPen(QtGui.QPen(Qt.red,2)); self.scene.addItem(li)
        return v

class Main(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle(APP_TITLE); self.resize(1200,800)

        labels_cfg = json.load(open(os.path.join(os.path.dirname(__file__),"config","labels.json")))
        labels = labels_cfg["classes"]

        self.canvas = Canvas(labels)

        # --- Left panel ---
        self.cmb = QComboBox(); [self.cmb.addItem(x) for x in labels]
        self.cmb.currentTextChanged.connect(lambda s: setattr(self.canvas,"current_label",s))

        self.mm = QDoubleSpinBox(); self.mm.setRange(1,500); self.mm.setDecimals(2)
        self.mm.setValue(labels_cfg.get("default_marker_size_mm",60.0))

        self.lbl_pxmm = QLabel("px/mm: —")
        self.lbl_tool = QLabel("Tool: box")

        # Tool buttons (toggle look)
        self.btn_box = QToolButton(text="Tool: Box"); self.btn_box.setCheckable(True); self.btn_box.setChecked(True)
        self.btn_line= QToolButton(text="Tool: Measure"); self.btn_line.setCheckable(True)
        group = QButtonGroup(self); group.setExclusive(True); group.addButton(self.btn_box); group.addButton(self.btn_line)
        self.btn_box.toggled.connect(lambda on: (on and self._set_tool('box')))
        self.btn_line.toggled.connect(lambda on: (on and self._set_tool('line')))

        bOpen = QPushButton("Open image");      bOpen.clicked.connect(self._open)
        bScale= QPushButton("Detect Scale");    bScale.clicked.connect(self._scale)
        bSave = QPushButton("Save JSON");       bSave.clicked.connect(self._save)
        bUndo = QPushButton("Undo (Z)");        bUndo.clicked.connect(self.canvas.undo)

        left=QVBoxLayout()
        left.addWidget(bOpen)
        left.addWidget(QLabel("Label")); left.addWidget(self.cmb)
        left.addWidget(self.btn_box); left.addWidget(self.btn_line)
        left.addWidget(QLabel("Marker side (mm)")); left.addWidget(self.mm)
        left.addWidget(bScale); left.addWidget(self.lbl_pxmm); left.addWidget(self.lbl_tool)
        left.addWidget(bUndo); left.addWidget(bSave); left.addStretch()
        wrapL=QWidget(); wrapL.setLayout(left)

        layout=QHBoxLayout(); layout.addWidget(wrapL); layout.addWidget(self.canvas,1)
        central=QWidget(); central.setLayout(layout); self.setCentralWidget(central)

        # Shortcuts
        QtGui.QShortcut(QtGui.QKeySequence("B"), self, activated=lambda: self.btn_box.setChecked(True))
        QtGui.QShortcut(QtGui.QKeySequence("M"), self, activated=lambda: self.btn_line.setChecked(True))
        QtGui.QShortcut(QtGui.QKeySequence("Z"), self, activated=self.canvas.undo)
        QtGui.QShortcut(QtGui.QKeySequence("S"), self, activated=self._save)
        QtGui.QShortcut(QtGui.QKeySequence("D"), self, activated=self._scale)

    def _set_tool(self,name):
        self.canvas.set_tool(name)
        self.lbl_tool.setText(f"Tool: {name}")

    def _open(self):
        p,_=QFileDialog.getOpenFileName(self,"Open image","","Images (*.jpg *.jpeg *.png)")
        if p: self.canvas.load(p)

    def _scale(self):
        v=self.canvas.detect_scale(float(self.mm.value()))
        self.lbl_pxmm.setText(f"px/mm: {v:.3f}" if v else "px/mm: —")

    def _save(self):
        if not self.canvas.img_path: return
        out=self.canvas.img_path + ".annotations.json"
        with open(out,"w") as f: json.dump(self.canvas.to_json(),f,indent=2)
        QMessageBox.information(self,"Saved",out)

if __name__=="__main__":
    app = QApplication([])
    w = Main(); w.show()
    app.exec()
