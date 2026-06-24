import os
import sys

import onnxruntime as ort 

import cv2
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                            QWidget, QLabel, QHBoxLayout, QFrame, QFileDialog, 
                            QMessageBox, QProgressBar, QSlider, QTabWidget, QCheckBox, QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPainterPath

# ==============================================================================
# AI 深度学习引擎
# ==============================================================================

class ONNXAIEngine:
    def __init__(self):
        self.model_path = "Zero_DCE.onnx"
        self.ort_session = None
        self.use_mock_ai = False
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                available_providers = ort.get_available_providers()
                providers = []
                if "CUDAExecutionProvider" in available_providers:
                    providers.append("CUDAExecutionProvider")
                if "CoreMLExecutionProvider" in available_providers:
                    providers.append("CoreMLExecutionProvider")
                providers.append("CPUExecutionProvider")

                self.ort_session = ort.InferenceSession(self.model_path, providers=providers)
                print(f"ONNXRuntime: 成功加载 Zero-DCE 模型，运行设备驱动: {self.ort_session.get_providers()}")
            except Exception as e:
                print(f"ONNX 模型加载失败: {e}，将使用模拟模式")
                self.use_mock_ai = True
        else:
            print("未找到 Zero_DCE.onnx，开启模拟效果。")
            self.use_mock_ai = True

    def enhance_ai(self, img_u8):
        if self.use_mock_ai or self.ort_session is None: 
            return img_u8
        
        img_float = img_u8.astype(np.float32) / 255.0
        average_brightness = np.mean(img_float)
        BRIGHTNESS_THRESHOLD = 0.4 
        
        if average_brightness > BRIGHTNESS_THRESHOLD:
            return img_u8

        intensity = np.clip(1.0 - (average_brightness / BRIGHTNESS_THRESHOLD), 0.1, 1.0)

        h, w = img_float.shape[:2]
        scale = 1.0
        max_dim = 1200
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            ai_input = cv2.resize(img_float, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else: 
            ai_input = img_float

        blob = ai_input.transpose(2, 0, 1).astype(np.float32)
        blob = np.expand_dims(blob, axis=0)

        input_name = self.ort_session.get_inputs()[0].name
        out_tensor = self.ort_session.run(None, {input_name: blob})[0]
        out_img = out_tensor[0].transpose(1, 2, 0)

        if scale != 1.0:
            out_img = cv2.resize(out_img, (w, h), interpolation=cv2.INTER_CUBIC)
        out_img = np.clip(out_img, 0.0, 1.0)

        luma = 0.2126 * img_float[:,:,0] + 0.7152 * img_float[:,:,1] + 0.0722 * img_float[:,:,2]
        mask = np.power(1.0 - luma, 2.0) 
        mask = np.expand_dims(mask, axis=2)
        
        effective_mask = mask * intensity
        smart_img = img_float * (1.0 - effective_mask) + out_img * effective_mask
        
        return (np.clip(smart_img, 0.0, 1.0) * 255).astype(np.uint8)

ai_engine = ONNXAIEngine()

# ==============================================================================
# 核心渲染管线（色彩域隔离版，大幅提升滑块响应速度）
# ==============================================================================

def process_color_and_detail(img_u8, params):
    """纯色彩计算逻辑（剥离耗时的磨皮），保证拖动滑块时具备 60FPS 的渲染速度"""
    img_f = img_u8.astype(np.float32) / 255.0
    
    # 1. 基础光影与色温色调
    exp = params.get('exposure', 0) / 50.0
    temp = params.get('temperature', 0) / 100.0
    tint = params.get('tint', 0) / 100.0
    
    if exp != 0 or temp != 0 or tint != 0:
        r_mult = (2.0 ** exp) * (1.0 + temp * 0.2)
        g_mult = (2.0 ** exp) * (1.0 + tint * 0.2)
        b_mult = (2.0 ** exp) * (1.0 - temp * 0.2)
        img_f[:,:,0] = np.clip(img_f[:,:,0] * r_mult, 0.0, 1.0)
        img_f[:,:,1] = np.clip(img_f[:,:,1] * g_mult, 0.0, 1.0)
        img_f[:,:,2] = np.clip(img_f[:,:,2] * b_mult, 0.0, 1.0)

    # 2. 动态范围: 高光/阴影/对比度
    cont = params.get('contrast', 0) / 100.0
    hl = params.get('highlights', 0) / 100.0
    shad = params.get('shadows', 0) / 100.0
    if hl != 0 or shad != 0 or cont != 0:
        luma = 0.299 * img_f[:,:,0] + 0.587 * img_f[:,:,1] + 0.114 * img_f[:,:,2]
        luma_new = luma.copy()
        if shad > 0: luma_new += shad * ((1.0 - luma)**2) * 0.5
        elif shad < 0: luma_new += shad * luma * 0.5
        if hl > 0: luma_new -= hl * (luma**2) * 0.5
        elif hl < 0: luma_new -= hl * (1.0 - luma) * 0.5
        if cont != 0: luma_new = (luma_new - 0.5) * (1.0 + cont) + 0.5
        
        luma_new = np.clip(luma_new, 1e-6, 1.0)
        img_f = np.clip(img_f * np.expand_dims(luma_new / (luma + 1e-6), axis=2), 0.0, 1.0)

    # 3. 黑白场与胶片曲线
    whites = params.get('whites', 0) / 100.0
    blacks = params.get('blacks', 0) / 100.0
    if whites != 0 or blacks != 0:
        bp = 0.0 - (blacks * 0.05)
        wp = 1.0 - (whites * 0.05)
        img_f = np.clip((img_f - bp) / (wp - bp + 1e-6), 0.0, 1.0)

    film_curve = params.get('film_curve', 0) / 100.0
    if film_curve > 0:
        s_curve = img_f * img_f * (3.0 - 2.0 * img_f)
        img_f = img_f * (1.0 - film_curve) + s_curve * film_curve

    # 4. 色彩鲜艳度与饱和度
    vib = params.get('vibrance', 0) / 100.0
    sat = params.get('saturation', 0) / 100.0
    if vib != 0 or sat != 0:
        luma = 0.299 * img_f[:,:,0] + 0.587 * img_f[:,:,1] + 0.114 * img_f[:,:,2]
        l_map = np.expand_dims(luma, axis=2)
        sat_mult = 1.0 + sat
        if vib != 0:
            delta = np.max(img_f, axis=2) - np.min(img_f, axis=2)
            sat_mult = sat_mult + vib * (1.0 - delta)
        
        if isinstance(sat_mult, np.ndarray):
            sat_mult = np.expand_dims(np.maximum(0.0, sat_mult), axis=2)
        else:
            sat_mult = max(0.0, sat_mult)
        img_f = np.clip(l_map + (img_f - l_map) * sat_mult, 0.0, 1.0)

    # 5. 清晰度与锐化
    clarity = params.get('clarity', 0) / 100.0
    sharpen = params.get('sharpen', 0) / 100.0
    if clarity != 0:
        blur_c = cv2.GaussianBlur(img_f, (0, 0), 3.0)
        img_f = np.clip(img_f + (img_f - blur_c) * clarity, 0.0, 1.0)
    if sharpen > 0:
        blur_s = cv2.GaussianBlur(img_f, (0, 0), 1.0)
        img_f = np.clip(img_f + (img_f - blur_s) * sharpen, 0.0, 1.0)

    return (img_f * 255).astype(np.uint8)

# ==============================================================================
# UI 核心组件
# ==============================================================================
class HistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(85)
        self.hist_data = None

    def update_hist(self, img_u8):
        h, w = img_u8.shape[:2]
        max_dim = 256
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img_small = cv2.resize(img_u8, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
        else:
            img_small = img_u8

        r = cv2.calcHist([img_small], [0], None, [256], [0, 256]).flatten()
        g = cv2.calcHist([img_small], [1], None, [256], [0, 256]).flatten()
        b = cv2.calcHist([img_small], [2], None, [256], [0, 256]).flatten()
        self.hist_data = [r, g, b]
        self.update() 

    def paintEvent(self, event):
        if not self.hist_data: 
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        
        max_val = max(self.hist_data[0].max(), self.hist_data[1].max(), self.hist_data[2].max())
        if max_val == 0: 
            max_val = 1
        
        colors = [QColor(255, 60, 60, 110), QColor(60, 255, 60, 110), QColor(60, 160, 255, 110)]
        
        for i in range(3):
            path = QPainterPath()
            path.moveTo(0, h)
            for x in range(256):
                x_pos = x * w / 255.0
                y_pos = h - (self.hist_data[i][x] / max_val * h * 0.85) 
                path.lineTo(x_pos, y_pos)
            path.lineTo(w, h)
            p.fillPath(path, colors[i])

class ModernToggle(QCheckBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(48, 26)
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRect(0, 0, self.width(), self.height())
        if self.isChecked(): 
            p.setBrush(QColor("#ffffff")) 
        else: 
            p.setBrush(QColor("#3f3f46"))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(rect, 13, 13)
        
        p.setBrush(QColor("#000000") if self.isChecked() else QColor("#ffffff"))
        if self.isChecked(): 
            p.drawEllipse(self.width() - 24, 2, 22, 22)
        else: 
            p.drawEllipse(2, 2, 22, 22)

class ResetSlider(QSlider):
    def mouseDoubleClickEvent(self, event):
        self.setValue(0)
        super().mouseDoubleClickEvent(event)

class SleekCanvas(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setAcceptDrops(True)
        self.setStyleSheet("QFrame { background-color: #09090b; border-radius: 12px; }")
        
        self.layout = QVBoxLayout(self)
        self.center_info = QLabel("拖入照片\n\n或点击此处选择")
        self.center_info.setStyleSheet("color: #71717a; font-size: 16px; font-weight: bold; letter-spacing: 2px;")
        self.center_info.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.center_info)

        self.preview_lbl = QLabel(self)
        self.preview_lbl.setAlignment(Qt.AlignCenter)
        self.preview_lbl.hide()
        self.layout.addWidget(self.preview_lbl, stretch=1)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): 
            e.acceptProposedAction()

    def dropEvent(self, e):
        files = [u.toLocalFile() for u in e.mimeData().urls() if os.path.isfile(u.toLocalFile())]
        if files: 
            self.parent.add_files(files)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and not self.parent.files:
            files, _ = QFileDialog.getOpenFileNames(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
            if files:
                self.parent.add_files(files)

# ==============================================================================
# 多线程与管线
# ==============================================================================
class BatchWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(int)

    def __init__(self, files, params_dict):
        super().__init__()
        self.files = files
        self.params_dict = params_dict
    
    def run(self):
        success = 0
        for i, path in enumerate(self.files):
            try:
                p = self.params_dict[path]
                out_path = f"{os.path.splitext(path)[0]}_Pro{os.path.splitext(path)[1]}"
                with Image.open(path) as img:
                    if p.get("rotation", 0) != 0: 
                        img = img.rotate(-p["rotation"], expand=True)
                    img_u8 = np.array(img.convert('RGB'))
                    
                    is_ai = p.get("ai_enabled", True)
                    sliders = p["ai_sliders"] if is_ai else p["raw_sliders"]
                    
                    if is_ai: 
                        img_u8 = ai_engine.enhance_ai(img_u8)
                    
                    # 补齐：在导出时处理空间域磨皮
                    retouch = sliders.get("retouch", 0)
                    if retouch > 0:
                        smoothed = cv2.edgePreservingFilter(img_u8, flags=1, sigma_s=30, sigma_r=0.1 * (retouch / 100.0))
                        img_u8 = cv2.addWeighted(smoothed, retouch / 100.0, img_u8, 1.0 - (retouch / 100.0), 0)
                        
                    out_arr = process_color_and_detail(img_u8, sliders)
                    Image.fromarray(out_arr, 'RGB').save(out_path, quality=98)
                    success += 1
            except Exception as e: 
                print(f"Error {path}: {e}")
            self.progress.emit(int(((i + 1) / len(self.files)) * 100))
        self.finished.emit(success)

# ==============================================================================
# 主程序
# ==============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QGlow 1.0.1")
        self.setStyleSheet("QMainWindow { background-color: #000000; } QWidget { font-family: sans-serif; } QLabel { color: #d4d4d8; }")
        self.setMinimumSize(550, 480)
        self.resize(550, 480)
        
        self.files = []
        self.current_idx = 0
        self.file_params = {}
        
        self.full_source_rgb = None
        self.preview_source_rgb = None
        
        # [优化项] 性能缓存机制
        self.cached_bases = {}       # 存放开启与关闭 AI 的原底图（秒切 AI 开关用）
        self.cached_spatial_u8 = None# 存放磨皮计算后的图（拖动色彩滑块免计算用）
        self.last_retouch_val = None

        self.pixmap_orig = None
        self.pixmap_graded = None
        
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._render_graded)

        self.zero_sliders = {k: 0 for k in ["exposure", "contrast", "highlights", "shadows", "whites", "blacks", 
                                           "film_curve", "temperature", "tint", "vibrance", "saturation", 
                                           "retouch", "clarity", "sharpen"]}
        self.magic_sliders = self.zero_sliders.copy()
        self.magic_sliders.update({"retouch": 30, "film_curve": 15, "vibrance": 20})

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        left_w = QWidget()
        left_l = QVBoxLayout(left_w)
        left_l.setContentsMargins(30, 20, 30, 20)
        left_l.setSpacing(15)
        
        self.hud = QLabel("")
        self.hud.setStyleSheet("color: #71717a; font-weight: bold;")
        left_l.addWidget(self.hud, alignment=Qt.AlignCenter)
        
        self.canvas = SleekCanvas(self)
        left_l.addWidget(self.canvas, stretch=1)

        self.tools_layout = QHBoxLayout()
        self.tools_layout.setAlignment(Qt.AlignCenter)
        self.tools_layout.setSpacing(15)
        btn_style = "QPushButton { background: transparent; color: #a1a1aa; font-weight: bold; font-size: 14px; padding: 5px; } QPushButton:hover { color: #ffffff; }"
        
        self.btn_prev = QPushButton("◀ 上一张")
        self.btn_prev.setStyleSheet(btn_style)
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_prev.hide()

        self.btn_rotate = QPushButton("↺ 旋转")
        self.btn_rotate.setStyleSheet(btn_style)
        self.btn_rotate.clicked.connect(self.rotate_image)
        self.btn_rotate.hide()

        self.btn_crop = QPushButton("✂ 裁切")
        self.btn_crop.setStyleSheet(btn_style)
        self.btn_crop.clicked.connect(lambda: QMessageBox.information(self, "提示", "预留接口"))
        self.btn_crop.hide()

        self.btn_compare = QPushButton("👁 对比原图")
        self.btn_compare.setStyleSheet(btn_style)
        self.btn_compare.pressed.connect(lambda: self.show_original(True))
        self.btn_compare.released.connect(lambda: self.show_original(False))
        self.btn_compare.hide()

        self.btn_next = QPushButton("下一张 ▶")
        self.btn_next.setStyleSheet(btn_style)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_next.hide()

        self.tools_layout.addWidget(self.btn_prev)
        self.tools_layout.addWidget(self.btn_rotate)
        self.tools_layout.addWidget(self.btn_crop)
        self.tools_layout.addWidget(self.btn_compare)
        self.tools_layout.addWidget(self.btn_next)
        left_l.addLayout(self.tools_layout)

        bottom_h = QHBoxLayout()
        self.btn_export = QPushButton("✨ 批量导出")
        self.btn_export.setStyleSheet("QPushButton { background: #ffffff; color: #000; padding: 15px; font-weight: bold; border-radius: 8px; } QPushButton:disabled { background: #27272a; color: #71717a; }")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.start_export)
        bottom_h.addWidget(self.btn_export, stretch=1)

        self.btn_toggle_panel = QPushButton("⚙️ 精修微调")
        self.btn_toggle_panel.setStyleSheet("background: #27272a; color: #fff; padding: 15px; font-weight: bold; border-radius: 8px;")
        self.btn_toggle_panel.setCheckable(True)
        self.btn_toggle_panel.toggled.connect(self.toggle_panel)
        bottom_h.addWidget(self.btn_toggle_panel)
        
        left_l.addLayout(bottom_h)
        self.progress = QProgressBar()
        self.progress.hide()
        self.progress.setFixedHeight(2)
        left_l.addWidget(self.progress)
        main_layout.addWidget(left_w, stretch=1)

        self.right_panel = QFrame()
        self.right_panel.setFixedWidth(320)
        self.right_panel.setStyleSheet("background-color: #09090b; border-left: 1px solid #27272a;")
        self.right_panel.hide()
        right_l = QVBoxLayout(self.right_panel)
        right_l.setContentsMargins(15, 20, 15, 20)
        right_l.setSpacing(10)

        self.histogram = HistogramWidget()
        right_l.addWidget(self.histogram)

        preset_box = QFrame()
        preset_box.setStyleSheet("QFrame { background-color: #18181b; border: 1px solid #27272a; border-radius: 8px; }")
        preset_layout = QVBoxLayout(preset_box)
        preset_layout.setContentsMargins(10, 10, 10, 10)
        
        preset_title = QLabel("✨ 一键智能美化")
        preset_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #a1a1aa;")
        preset_layout.addWidget(preset_title)

        grid = QGridLayout()
        grid.setSpacing(6)
        
        btn_preset_style = """
            QPushButton { 
                background: #27272a; color: #f4f4f5; border: none; 
                border-radius: 6px; padding: 8px 4px; font-size: 11px; font-weight: bold;
            } 
            QPushButton:hover { background: #3f3f46; color: #ffffff; }
            QPushButton:pressed { background: #09090b; }
        """
        
        self.btn_auto = QPushButton("✨ 智能一键")
        self.btn_auto.setStyleSheet("QPushButton { background: #ffffff; color: #000; border-radius: 6px; padding: 8px 4px; font-size: 11px; font-weight: bold; } QPushButton:hover { background: #e4e4e7; }")
        self.btn_auto.clicked.connect(lambda: self.apply_preset("auto"))
        
        self.btn_film = QPushButton("🎞️ 复古胶片")
        self.btn_film.setStyleSheet(btn_preset_style)
        self.btn_film.clicked.connect(lambda: self.apply_preset("film"))

        self.btn_landscape = QPushButton("🏞️ 清透风景")
        self.btn_landscape.setStyleSheet(btn_preset_style)
        self.btn_landscape.clicked.connect(lambda: self.apply_preset("landscape"))

        self.btn_warm = QPushButton("🌸 温暖日系")
        self.btn_warm.setStyleSheet(btn_preset_style)
        self.btn_warm.clicked.connect(lambda: self.apply_preset("warm"))

        self.btn_cool = QPushButton("💎 冷调高级")
        self.btn_cool.setStyleSheet(btn_preset_style)
        self.btn_cool.clicked.connect(lambda: self.apply_preset("cool"))

        self.btn_reset = QPushButton("🔄 还原原始")
        self.btn_reset.setStyleSheet("QPushButton { background: #3f3f46; color: #d4d4d8; border-radius: 6px; padding: 8px 4px; font-size: 11px; font-weight: bold; } QPushButton:hover { background: #52525b; color: #ffffff; }")
        self.btn_reset.clicked.connect(lambda: self.apply_preset("reset"))

        grid.addWidget(self.btn_auto, 0, 0)
        grid.addWidget(self.btn_film, 0, 1)
        grid.addWidget(self.btn_landscape, 0, 2)
        grid.addWidget(self.btn_warm, 1, 0)
        grid.addWidget(self.btn_cool, 1, 1)
        grid.addWidget(self.btn_reset, 1, 2)
        preset_layout.addLayout(grid)
        right_l.addWidget(preset_box)

        ai_header = QHBoxLayout()
        lbl_ai = QLabel("🧠 AI 自适应光影修护")
        lbl_ai.setStyleSheet("font-weight: bold; font-size: 12px; color: #fff;")
        self.toggle_ai = ModernToggle()
        self.toggle_ai.setChecked(True)
        self.toggle_ai.toggled.connect(self.on_ai_toggled)
        ai_header.addWidget(lbl_ai)
        ai_header.addStretch()
        ai_header.addWidget(self.toggle_ai)
        right_l.addLayout(ai_header)
        right_l.addWidget(QFrame(frameShape=QFrame.HLine))

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabWidget::pane { border: none; } QTabBar::tab { background: transparent; color: #71717a; padding: 6px 12px; font-weight: bold; font-size: 12px; } QTabBar::tab:selected { color: #ffffff; border-bottom: 2px solid #ffffff; }")
        self.sliders = {}
        self.labels = {}
        
        tab_light = QWidget()
        lay_light = QVBoxLayout(tab_light)
        lay_light.setContentsMargins(0, 10, 0, 0)
        self._add_s(lay_light, "exposure", "曝光", -100, 100)
        self._add_s(lay_light, "contrast", "对比度", -100, 100)
        self._add_s(lay_light, "highlights", "高光", -100, 100)
        self._add_s(lay_light, "shadows", "阴影", -100, 100)
        self._add_s(lay_light, "whites", "白色色阶", -100, 100)
        self._add_s(lay_light, "blacks", "黑色色阶", -100, 100)
        self._add_s(lay_light, "film_curve", "🎞️ 胶片感曲线", 0, 100) 
        lay_light.addStretch(1)

        tab_color = QWidget()
        lay_color = QVBoxLayout(tab_color)
        lay_color.setContentsMargins(0, 10, 0, 0)
        self._add_s(lay_color, "temperature", "色温", -100, 100)
        self._add_s(lay_color, "tint", "色调", -100, 100)
        self._add_s(lay_color, "vibrance", "自然饱和度", -100, 100)
        self._add_s(lay_color, "saturation", "饱和度", -100, 100)
        lay_color.addStretch(1)

        tab_detail = QWidget()
        lay_detail = QVBoxLayout(tab_detail)
        lay_detail.setContentsMargins(0, 10, 0, 0)
        self._add_s(lay_detail, "retouch", "🌸 智能磨皮", 0, 100)
        self._add_s(lay_detail, "clarity", "清晰度", -100, 100)
        self._add_s(lay_detail, "sharpen", "锐化", 0, 100)
        lay_detail.addStretch(1)

        self.tabs.addTab(tab_light, "光影")
        self.tabs.addTab(tab_color, "色彩")
        self.tabs.addTab(tab_detail, "细节")
        right_l.addWidget(self.tabs, stretch=1)

        self.btn_sync = QPushButton("同步参数至全部图片")
        self.btn_sync.setStyleSheet("background: #27272a; color: white; padding: 10px; border-radius: 6px;")
        self.btn_sync.clicked.connect(self.sync_all)
        right_l.addWidget(self.btn_sync)

        main_layout.addWidget(self.right_panel)

    def _add_s(self, layout, key, name, min_v, max_v):
        row = QVBoxLayout()
        row.setSpacing(2)
        h = QHBoxLayout()
        c = "#ffffff" if "胶片" in name or "磨皮" in name else "#d4d4d8"
        h.addWidget(QLabel(name, styleSheet=f"color: {c}; font-size: 11px; font-weight: bold;"))
        lbl_v = QLabel("0", styleSheet="color: #fff; font-size: 10px; font-weight: bold;")
        h.addStretch()
        h.addWidget(lbl_v)
        row.addLayout(h)
        s = ResetSlider(Qt.Horizontal)
        s.setRange(min_v, max_v)
        s.setStyleSheet("QSlider::groove:horizontal{background:#3f3f46; height:3px; border-radius:1px;} QSlider::handle:horizontal{background:#ffffff; width:12px; height:12px; margin:-4px 0; border-radius:6px;}")
        s.valueChanged.connect(lambda v: self.on_slide(key, v, lbl_v))
        self.sliders[key] = s
        self.labels[key] = lbl_v
        row.addWidget(s)
        layout.addLayout(row)

    def toggle_panel(self, checked):
        # [优化项] 移除了原有的自我拉伸逻辑，现在界面从内部平滑推挤，最大化时完美适配
        self.right_panel.setVisible(checked)
        QApplication.processEvents()
        if self.pixmap_graded:
            self.update_canvas(self.pixmap_graded)

    def add_files(self, files):
        new_files = [f for f in files if f not in self.files and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        for f in new_files:
            self.file_params[f] = {"ai_enabled": True, "rotation": 0, "ai_sliders": self.magic_sliders.copy(), "raw_sliders": self.zero_sliders.copy()}
            self.files.append(f)
            
        if self.files:
            self.canvas.center_info.hide()
            self.canvas.preview_lbl.show()
            self.btn_export.setEnabled(True)
            self.btn_rotate.show()
            self.btn_crop.show()
            self.btn_compare.show()
            self.update_navigation_visibility()
            self.load_image()

    def update_navigation_visibility(self):
        has_multiple = len(self.files) > 1
        self.btn_prev.setVisible(has_multiple)
        self.btn_next.setVisible(has_multiple)

    def prev_image(self):
        if len(self.files) > 1:
            self.current_idx = (self.current_idx - 1) % len(self.files)
            self.load_image()

    def next_image(self):
        if len(self.files) > 1:
            self.current_idx = (self.current_idx + 1) % len(self.files)
            self.load_image()

    def load_image(self):
        path = self.files[self.current_idx]
        p = self.file_params[path]
        
        # 换图时，彻底清空缓存矩阵
        self.cached_bases = {}
        self.cached_spatial_u8 = None
        
        with Image.open(path) as img:
            self.hud.setText(f"{self.current_idx+1}/{len(self.files)}  •  {os.path.basename(path)}")
            if p.get("rotation", 0) != 0: 
                img = img.rotate(-p["rotation"], expand=True)
            
            self.full_source_rgb = np.array(img.convert('RGB')) 

            h, w = self.full_source_rgb.shape[:2]
            max_dim = 1000 
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                self.preview_source_rgb = cv2.resize(self.full_source_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                self.preview_source_rgb = self.full_source_rgb.copy()

            ph, pw = self.preview_source_rgb.shape[:2]
            self.pixmap_orig = QPixmap.fromImage(QImage(self.preview_source_rgb.data, pw, ph, pw*3, QImage.Format_RGB888))

        self.toggle_ai.blockSignals(True)
        self.toggle_ai.setChecked(p["ai_enabled"])
        self.toggle_ai.blockSignals(False)
        self._sync_ui_to_dict()
        self.process_pipeline()

    def _sync_ui_to_dict(self):
        p = self.file_params[self.files[self.current_idx]]
        active_sliders = p["ai_sliders"] if p["ai_enabled"] else p["raw_sliders"]
        for k, s in self.sliders.items():
            s.blockSignals(True)
            s.setValue(active_sliders[k])
            s.blockSignals(False)
            self.labels[k].setText(str(s.value()))

    def rotate_image(self):
        if not self.files: return
        p = self.file_params[self.files[self.current_idx]]
        p["rotation"] = (p.get("rotation", 0) + 90) % 360
        
        # [优化项] 告别重新载入硬盘数据，直接利用内存极速旋转 Numpy 矩阵 (k=-1 顺时针 90 度)
        self.full_source_rgb = np.ascontiguousarray(np.rot90(self.full_source_rgb, k=-1))
        self.preview_source_rgb = np.ascontiguousarray(np.rot90(self.preview_source_rgb, k=-1))
        
        self.cached_bases = {}
        self.cached_spatial_u8 = None
        
        ph, pw = self.preview_source_rgb.shape[:2]
        self.pixmap_orig = QPixmap.fromImage(QImage(self.preview_source_rgb.data, pw, ph, pw*3, QImage.Format_RGB888))
        self.process_pipeline()

    def on_ai_toggled(self, checked):
        if not self.files: return
        self.file_params[self.files[self.current_idx]]["ai_enabled"] = checked
        self._sync_ui_to_dict() 
        self.process_pipeline()

    def on_slide(self, key, val, lbl):
        if not self.files: return
        lbl.setText(str(val))
        p = self.file_params[self.files[self.current_idx]]
        if p["ai_enabled"]: p["ai_sliders"][key] = val
        else: p["raw_sliders"][key] = val
        self.render_grading(False)

    def calculate_auto_analysis(self):
        if self.preview_source_rgb is None:
            return self.zero_sliders.copy()
            
        img_f = self.preview_source_rgb.astype(np.float32) / 255.0
        luma = 0.2126 * img_f[:,:,0] + 0.7152 * img_f[:,:,1] + 0.0722 * img_f[:,:,2]
        avg_luma = np.mean(luma)
        std_luma = np.std(luma)
        
        exposure_val = int(np.clip((0.45 - avg_luma) * 110, -40, 50))
        contrast_val = int(np.clip((0.22 - std_luma) * 160, -20, 25))
        
        p5 = np.percentile(luma, 5)
        p95 = np.percentile(luma, 95)
        shadows_val = int(np.clip((0.15 - p5) * 180, 0, 45)) if p5 < 0.15 else 0
        highlights_val = int(np.clip((p95 - 0.85) * 120, 0, 35)) if p95 > 0.85 else 0
        
        auto_preset = self.zero_sliders.copy()
        auto_preset.update({
            "exposure": exposure_val, "contrast": contrast_val, "shadows": shadows_val,
            "highlights": highlights_val, "film_curve": 10, "vibrance": 15,
            "retouch": 25 if avg_luma > 0.2 else 10, "clarity": 10, "sharpen": 15
        })
        return auto_preset

    def apply_preset(self, preset_name):
        if not self.files: return
            
        presets = {
            "reset": self.zero_sliders.copy(),
            "film": {**self.zero_sliders, "exposure": 5, "contrast": 15, "film_curve": 40, "temperature": 15, "vibrance": 25, "retouch": 30, "clarity": -10},
            "landscape": {**self.zero_sliders, "exposure": 8, "contrast": 20, "highlights": 15, "shadows": -5, "temperature": -10, "tint": 5, "vibrance": 35, "clarity": 20, "sharpen": 25},
            "warm": {**self.zero_sliders, "exposure": 12, "contrast": -10, "shadows": 25, "temperature": 25, "vibrance": 20, "retouch": 40, "clarity": -15},
            "cool": {**self.zero_sliders, "exposure": 6, "contrast": 25, "whites": 10, "blacks": -5, "temperature": -20, "tint": -5, "vibrance": -10, "saturation": -5, "sharpen": 30}
        }
        
        target_params = self.calculate_auto_analysis() if preset_name == "auto" else presets.get(preset_name, self.zero_sliders).copy()
            
        p = self.file_params[self.files[self.current_idx]]
        if p["ai_enabled"]: p["ai_sliders"] = target_params
        else: p["raw_sliders"] = target_params
            
        self._sync_ui_to_dict()
        self.render_grading(True)

    def process_pipeline(self):
        if not self.files: return
        p = self.file_params[self.files[self.current_idx]]
        is_ai = p["ai_enabled"]
        
        # [优化项] 智能状态缓存。Toggled AI 时将不会再卡死，直接从内存中调取图层
        if is_ai not in self.cached_bases:
            self.hud.setText("引擎渲染中...")
            QApplication.processEvents()
            self.cached_bases[is_ai] = ai_engine.enhance_ai(self.preview_source_rgb) if is_ai else self.preview_source_rgb.copy()
            
        self.cached_spatial_u8 = None 
        self.last_retouch_val = None
            
        self.hud.setText(f"{self.current_idx+1}/{len(self.files)}  •  {os.path.basename(self.files[self.current_idx])}")
        self.render_grading(True)

    def render_grading(self, immediate=False):
        if immediate: 
            self.timer.stop()
            self._render_graded()
        else: 
            self.timer.start(16)

    def _render_graded(self):
        p = self.file_params[self.files[self.current_idx]]
        is_ai = p["ai_enabled"]
        active_sliders = p["ai_sliders"] if is_ai else p["raw_sliders"]
        
        # [优化项] 隔离磨皮渲染：仅在"智能磨皮"数值发生改变时，才重新计算最慢的保边滤波
        retouch = active_sliders.get('retouch', 0)
        base_u8 = self.cached_bases.get(is_ai)
        
        if base_u8 is None:
            return
            
        if self.cached_spatial_u8 is None or self.last_retouch_val != retouch:
            if retouch > 0:
                smoothed = cv2.edgePreservingFilter(base_u8, flags=1, sigma_s=30, sigma_r=0.1 * (retouch / 100.0))
                self.cached_spatial_u8 = cv2.addWeighted(smoothed, retouch / 100.0, base_u8, 1.0 - (retouch / 100.0), 0)
            else:
                self.cached_spatial_u8 = base_u8.copy()
            self.last_retouch_val = retouch
        
        # 因为跳过了耗时的磨皮，拖动其余色彩滑块将保持如丝般顺滑
        out_arr = np.ascontiguousarray(process_color_and_detail(self.cached_spatial_u8, active_sliders))
        self.histogram.update_hist(out_arr)

        h, w = out_arr.shape[:2]
        self.pixmap_graded = QPixmap.fromImage(QImage(out_arr.data, w, h, w*3, QImage.Format_RGB888))
        self.update_canvas(self.pixmap_graded)

    def update_canvas(self, pm):
        if pm and not pm.isNull():
            r = self.canvas.preview_lbl.devicePixelRatioF() or 1.0
            lbl_w = max(self.canvas.preview_lbl.width(), 10)
            lbl_h = max(self.canvas.preview_lbl.height(), 10)
            s = pm.scaled(int(lbl_w * r), int(lbl_h * r), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            s.setDevicePixelRatio(r)
            self.canvas.preview_lbl.setPixmap(s)

    def show_original(self, orig):
        self.update_canvas(self.pixmap_orig if orig else self.pixmap_graded)

    def sync_all(self):
        if not self.files: return
        cur = self.file_params[self.files[self.current_idx]]
        for f in self.files: 
            self.file_params[f] = {"ai_enabled": cur["ai_enabled"], "rotation": cur["rotation"], "ai_sliders": cur["ai_sliders"].copy(), "raw_sliders": cur["raw_sliders"].copy()}
        self.hud.setText("已同步当前全套参数至全部图片")

    def start_export(self):
        if not self.files: return
        self.btn_export.setEnabled(False)
        self.btn_toggle_panel.setEnabled(False)
        self.progress.show()
        self.progress.setValue(0)
        
        self.worker = BatchWorker(self.files, self.file_params)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.export_done)
        self.worker.start()

    def export_done(self, count):
        self.btn_export.setEnabled(True)
        self.btn_toggle_panel.setEnabled(True)
        self.progress.hide()
        QMessageBox.information(self, "批量完成", f"成功输出 {count} 张精修图片！")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.pixmap_graded: self.update_canvas(self.pixmap_graded)
        
if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
