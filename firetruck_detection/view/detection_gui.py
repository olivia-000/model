import os
import cv2
import numpy as np
from time import perf_counter
from typing import List

import json

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, QPoint
from PyQt5.QtGui import QImage, QFontDatabase

from model.vlm_worker import PROVIDER_MODELS

from utils import (
    open_crop_image,
    yaml_load,
    DEFAULT_CFG,
    NUM_VIDEO_STREAMS, 
    MAX_CONFIDENCE,
    MIN_CONFIDENCE
)
from .roi_selector import ROIDialog

class ToggleSwitch(QtWidgets.QWidget):
    """Left-right sliding toggle switch widget."""
    toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._checked = False
        self.setFixedSize(52, 26)
        self.setCursor(QtCore.Qt.PointingHandCursor)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool) -> None:
        self._checked = checked
        self.update()

    def mousePressEvent(self, event):
        self._checked = not self._checked
        self.toggled.emit(self._checked)
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        track = QtGui.QColor('#5E81AC') if self._checked else QtGui.QColor('#4C566A')
        painter.setBrush(track)
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRoundedRect(0, 3, 52, 20, 10, 10)
        painter.setBrush(QtGui.QColor('#ECEFF4'))
        x = 26 if self._checked else 2
        painter.drawEllipse(x, 1, 24, 24)
        painter.end()


class DetectionGUI(QtWidgets.QWidget):
    """
    The view class of the firetruck detection system.
    It is responsible for handling the GUI components and displaying the video streams. 
    """

    parameters_changed_signal = pyqtSignal()
    """ 
    Signal to adjust the confidence level and time interval.
    """

    window_closed_signal = pyqtSignal()
    """
    Signal to indicate the window is closed.
    """

    set_roi_signal = pyqtSignal(int, tuple)
    """
    Signal to set the ROI.

    Args:
        video_stream_idx (int): The index of the video stream.
        roi (tuple): The ROI of the video stream.
    """

    clear_roi_signal = pyqtSignal(int)
    """
    Signal to clear the ROI for a specific video stream.

    Args:
        video_stream_idx (int): The index of the video stream.
    """

    def __init__(self, cfg_path=DEFAULT_CFG) -> None:
        """ Initialize the DetectionMonitorView class """
        super().__init__()
        self.is_single_channel = False
        self.cfg = yaml_load(cfg_path)
        dm = self.cfg.get('detection_modules', {})
        self.bridge_enabled = dm.get('bridge', {}).get('enable', True)
        self.road_enabled = dm.get('road_damage', {}).get('enable', True)
        self.stage2_status = dm.get('bridge', {}).get('stage2', {}).get('enable', False)
        stage2 = dm.get('bridge', {}).get('stage2', {})
        self.vlm_enabled  = stage2.get('method', 'yolo_cls') == 'vlm'
        vlm_cfg = self.cfg.get('vlm', {})
        self.vlm_provider  = vlm_cfg.get('provider', 'gemini')
        self.vlm_model     = vlm_cfg.get('model', 'gemini-2.0-flash-lite')
        self.vlm_cooldown  = vlm_cfg.get('cooldown_sec', 30)
        self.font = self.cfg['font']
        self.font_path = f"font/{self.cfg['font']}.ttf"
        self.stream_locations = []
        for i in range(NUM_VIDEO_STREAMS):
            self.stream_locations.append(self.cfg[f'ch{i+1}_location'])

        # self.last_update_times = [0] * NUM_VIDEO_STREAMS
        # self.update_counters = [0] * NUM_VIDEO_STREAMS
        self.last_frames = [None] * NUM_VIDEO_STREAMS
        # self.mv_avg_fps = [0] * NUM_VIDEO_STREAMS # Exponential moving average of the FPS
        # self.alpha = 0.15  # Smoothing factor for exponential moving average
        self.rois = [None] * NUM_VIDEO_STREAMS  # Store ROIs for each video stream
        self._load_roi()

    def setupUi(self) -> None:
        """ Setup the GUI components """
        print('Setting up GUI...')

        self.setObjectName("Form")
        self.resize(1700, 820)
        self.setStyleSheet("""
            QWidget {
                background-color: #2E3440;
                color: #D8DEE9;
                font-family: Arial, sans-serif;
            }
            QLabel {
                color: #D8DEE9;
            }
            QPushButton {
                background-color: #4C566A;
                color: #D8DEE9;
                border: none;
                padding: 7px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5E81AC;
            }
            QSlider::groove:horizontal {
                height: 5px;
                background: #4C566A;
                margin: 2px 0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #88C0D0;
                border: none;
                width: 13px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSpinBox {
                background-color: #3B4252;
                color: #D8DEE9;
                border: 1px solid #4C566A;
                border-radius: 4px;
                padding: 4px;
            }
            QListWidget {
                background-color: #3B4252;
                color: #D8DEE9;
                border: none;
                border-radius: 4px;
                font-size: 12px;
            }
            QListWidget::item:hover {
                background-color: #4C566A;
            }
            #rightPanel {
                background-color: #272C38;
                border-left: 1px solid #3B4252;
            }
        """)

        # Main layout: horizontal (left = video, right = controls)
        self.mainLayout = QtWidgets.QHBoxLayout(self)
        self.mainLayout.setContentsMargins(4, 4, 0, 4)
        self.mainLayout.setSpacing(0)

        # ── LEFT: video grid + log list ──
        self.leftLayout = QtWidgets.QVBoxLayout()
        self.leftLayout.setContentsMargins(0, 0, 0, 0)
        self.leftLayout.setSpacing(4)

        self.videoStreamsGridWidget = QtWidgets.QWidget(self)
        self.videoStreamsGridLayout = QtWidgets.QGridLayout()
        self.videoStreamsGridLayout.setSpacing(2)
        self.videoStreamsGridLayout.setContentsMargins(0, 0, 0, 0)

        self.videoStreamLayouts = []
        self.videoStreamLabels = []
        self.videoStreamWidgets: List[QtWidgets.QWidget] = []

        for i in range(NUM_VIDEO_STREAMS):
            videoStreamLayout = QtWidgets.QVBoxLayout()
            videoStreamLayout.setSpacing(1)

            locationLabel = QtWidgets.QLabel(self.videoStreamsGridWidget)
            locationLabel.setObjectName(f"location_label_{i + 1}")
            locationLabel.setAlignment(QtCore.Qt.AlignCenter)
            locationLabel.setStyleSheet("font-size: 12px; color: #88C0D0; padding: 2px 0;")

            videoStreamWidget = QtWidgets.QLabel(self.videoStreamsGridWidget)
            videoStreamWidget.setObjectName(f"stream_{i + 1}")
            videoStreamWidget.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            videoStreamWidget.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Expanding)
            videoStreamWidget.setMinimumWidth(360)
            videoStreamWidget.setMinimumHeight(220)
            videoStreamWidget.setScaledContents(True)
            videoStreamWidget.setStyleSheet("background-color: #1E2330;")

            videoStreamLayout.addWidget(locationLabel)
            videoStreamLayout.addWidget(videoStreamWidget)

            self.videoStreamsGridLayout.addLayout(videoStreamLayout, i // 3, i % 3)

            self.videoStreamLayouts.append(videoStreamLayout)
            self.videoStreamLabels.append(locationLabel)
            self.videoStreamWidgets.append(videoStreamWidget)

        self.videoStreamsGridWidget.setLayout(self.videoStreamsGridLayout)
        self.leftLayout.addWidget(self.videoStreamsGridWidget, stretch=1)

        self.cropInfoListWidget = QtWidgets.QListWidget(self)
        self.cropInfoListWidget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        self.cropInfoListWidget.setMaximumHeight(120)
        self.cropInfoListWidget.setObjectName("cropInfoListWidget")
        self.leftLayout.addWidget(self.cropInfoListWidget)
        self.cropInfoListWidget.itemClicked.connect(self.crop_info_clicked)

        self.mainLayout.addLayout(self.leftLayout, stretch=1)

        # ── RIGHT: control panel ──
        self.controlPanelHorizontalWidget = QtWidgets.QWidget(self)
        self.controlPanelHorizontalWidget.setObjectName("rightPanel")
        self.controlPanelHorizontalWidget.setFixedWidth(260)
        self.controlPanelHorizontalLayout = QtWidgets.QVBoxLayout(self.controlPanelHorizontalWidget)
        self.controlPanelHorizontalLayout.setContentsMargins(14, 14, 14, 14)
        self.controlPanelHorizontalLayout.setSpacing(10)

        # Module toggles
        self.stageLayout = QtWidgets.QVBoxLayout()
        self.stageLayout.setSpacing(6)

        def _make_toggle_row(label_text, slot):
            w = QtWidgets.QWidget(self)
            layout = QtWidgets.QHBoxLayout(w)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(8)
            btn = QtWidgets.QPushButton(label_text)
            btn.clicked.connect(slot)
            layout.addWidget(btn)
            indicator = QtWidgets.QLabel()
            indicator.setFixedSize(16, 16)
            indicator.setStyleSheet("border-radius: 8px; background-color: green;")
            layout.addWidget(indicator)
            return w, btn, indicator

        self.bridgeWidget, self.setBridgeButton, self.bridgeIndicator = _make_toggle_row(
            "橋樑偵測 Stage1", lambda: self.toggle_stage_status(1))
        self.stageLayout.addWidget(self.bridgeWidget)

        self.roadWidget, self.setRoadButton, self.roadIndicator = _make_toggle_row(
            "AC 路面偵測", lambda: self.toggle_stage_status(2))
        self.stageLayout.addWidget(self.roadWidget)

        self.stage2Widget, self.setStage2Button, self.stage2Indicator = _make_toggle_row(
            "橋樑 Stage2", lambda: self.toggle_stage_status(3))
        self.stage2Indicator.setObjectName("stage2Indicator")
        self.stageLayout.addWidget(self.stage2Widget)

        self.controlPanelHorizontalLayout.addLayout(self.stageLayout)

        # ── VLM 區塊 ──────────────────────────────────────────────
        vlm_sep = QtWidgets.QFrame()
        vlm_sep.setFrameShape(QtWidgets.QFrame.HLine)
        vlm_sep.setStyleSheet("color: #3B4252;")
        self.controlPanelHorizontalLayout.addWidget(vlm_sep)

        # VLM toggle row
        vlm_toggle_row = QtWidgets.QWidget()
        vlm_toggle_layout = QtWidgets.QHBoxLayout(vlm_toggle_row)
        vlm_toggle_layout.setContentsMargins(0, 0, 0, 0)
        vlm_toggle_layout.setSpacing(8)
        vlm_label = QtWidgets.QLabel("VLM 模式")
        vlm_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #88C0D0;")
        self.vlmToggle = ToggleSwitch()
        self.vlmToggle.setChecked(self.vlm_enabled)
        self.vlmToggle.toggled.connect(self._on_vlm_toggle)
        vlm_toggle_layout.addWidget(vlm_label)
        vlm_toggle_layout.addStretch()
        vlm_toggle_layout.addWidget(self.vlmToggle)
        self.controlPanelHorizontalLayout.addWidget(vlm_toggle_row)

        # VLM provider dropdown
        vlm_provider_row = QtWidgets.QHBoxLayout()
        vlm_provider_lbl = QtWidgets.QLabel("Provider:")
        vlm_provider_lbl.setStyleSheet("font-size: 11px;")
        vlm_provider_row.addWidget(vlm_provider_lbl)
        self.vlmProviderCombo = QtWidgets.QComboBox()
        self.vlmProviderCombo.addItems(list(PROVIDER_MODELS.keys()))
        self.vlmProviderCombo.setCurrentText(self.vlm_provider)
        self.vlmProviderCombo.currentTextChanged.connect(self._on_vlm_provider_changed)
        self.vlmProviderCombo.setStyleSheet(
            "QComboBox { background-color: #3B4252; color: #D8DEE9; border: 1px solid #4C566A; "
            "border-radius: 4px; padding: 3px; font-size: 11px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { background-color: #3B4252; color: #D8DEE9; }"
        )
        vlm_provider_row.addWidget(self.vlmProviderCombo)
        self.controlPanelHorizontalLayout.addLayout(vlm_provider_row)

        # VLM model dropdown
        vlm_model_row = QtWidgets.QHBoxLayout()
        vlm_model_lbl = QtWidgets.QLabel("Model:")
        vlm_model_lbl.setStyleSheet("font-size: 11px;")
        vlm_model_row.addWidget(vlm_model_lbl)
        self.vlmModelCombo = QtWidgets.QComboBox()
        self._refresh_model_combo(self.vlm_provider)
        self.vlmModelCombo.setCurrentText(self.vlm_model)
        self.vlmModelCombo.setStyleSheet(
            "QComboBox { background-color: #3B4252; color: #D8DEE9; border: 1px solid #4C566A; "
            "border-radius: 4px; padding: 3px; font-size: 11px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { background-color: #3B4252; color: #D8DEE9; }"
        )
        vlm_model_row.addWidget(self.vlmModelCombo)
        self.controlPanelHorizontalLayout.addLayout(vlm_model_row)

        # LLM cooldown spinbox
        vlm_cooldown_row = QtWidgets.QHBoxLayout()
        vlm_cooldown_lbl = QtWidgets.QLabel("間隔(秒):")
        vlm_cooldown_lbl.setStyleSheet("font-size: 11px;")
        vlm_cooldown_row.addWidget(vlm_cooldown_lbl)
        self.vlmCooldownSpinBox = QtWidgets.QSpinBox()
        self.vlmCooldownSpinBox.setRange(0, 300)
        self.vlmCooldownSpinBox.setValue(self.vlm_cooldown)
        self.vlmCooldownSpinBox.setSuffix(" s")
        self.vlmCooldownSpinBox.setToolTip("同一相機兩次送件最短間隔（0 = 不限制）")
        self.vlmCooldownSpinBox.setStyleSheet(
            "QSpinBox { background-color: #3B4252; color: #D8DEE9; border: 1px solid #4C566A; padding: 2px; }"
        )
        vlm_cooldown_row.addWidget(self.vlmCooldownSpinBox)
        self.controlPanelHorizontalLayout.addLayout(vlm_cooldown_row)

        # VLM status label
        self.vlmStatusLabel = QtWidgets.QLabel()
        self.vlmStatusLabel.setStyleSheet("font-size: 10px; color: #A0AABB;")
        self.vlmStatusLabel.setWordWrap(True)
        self._update_vlm_status_label()
        self.controlPanelHorizontalLayout.addWidget(self.vlmStatusLabel)
        # ──────────────────────────────────────────────────────────

        # Separator
        sep1 = QtWidgets.QFrame()
        sep1.setFrameShape(QtWidgets.QFrame.HLine)
        sep1.setStyleSheet("color: #3B4252;")
        self.controlPanelHorizontalLayout.addWidget(sep1)

        # Current settings label
        self.currentSettingsLabel = QtWidgets.QLabel()
        self.currentSettingsLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.currentSettingsLabel.setObjectName("currentSettingsLabel")
        self.currentSettingsLabel.setWordWrap(True)
        self.currentSettingsLabel.setStyleSheet("font-size: 11px; color: #A0AABB;")
        self.controlPanelHorizontalLayout.addWidget(self.currentSettingsLabel)

        # Separator
        sep2 = QtWidgets.QFrame()
        sep2.setFrameShape(QtWidgets.QFrame.HLine)
        sep2.setStyleSheet("color: #3B4252;")
        self.controlPanelHorizontalLayout.addWidget(sep2)

        # Confidence sliders
        self.confidenceVerticalLayout = QtWidgets.QVBoxLayout()
        self.confidenceVerticalLayout.setSpacing(3)

        def _make_slider_group(name_text, val_attr, slider_attr, lo, hi):
            row = QtWidgets.QHBoxLayout()
            name_lbl = QtWidgets.QLabel(name_text)
            name_lbl.setStyleSheet("font-size: 11px;")
            row.addWidget(name_lbl)
            val_lbl = QtWidgets.QLabel()
            val_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            val_lbl.setStyleSheet("color: #88C0D0; font-weight: bold; font-size: 11px;")
            setattr(self, val_attr, val_lbl)
            row.addWidget(val_lbl)
            self.confidenceVerticalLayout.addLayout(row)

            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            slider.setMinimum(lo)
            slider.setMaximum(hi)
            slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
            slider.valueChanged.connect(self._update_confidence_text)
            setattr(self, slider_attr, slider)
            self.confidenceVerticalLayout.addWidget(slider)

            range_row = QtWidgets.QHBoxLayout()
            lo_lbl = QtWidgets.QLabel(str(lo))
            lo_lbl.setStyleSheet("font-size: 10px; color: #6B7588;")
            range_row.addWidget(lo_lbl)
            hi_lbl = QtWidgets.QLabel(str(hi))
            hi_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            hi_lbl.setStyleSheet("font-size: 10px; color: #6B7588;")
            range_row.addWidget(hi_lbl)
            self.confidenceVerticalLayout.addLayout(range_row)

        _make_slider_group("橋樑信心值:", "confValLabel_bridge", "slider_bridge_conf",
                           MIN_CONFIDENCE, MAX_CONFIDENCE)
        _make_slider_group("路面信心值:", "confValLabel_road", "slider_road_conf",
                           MIN_CONFIDENCE, MAX_CONFIDENCE)
        _make_slider_group("Stage2 高低差:", "confValLabel_height", "slider_stage2_height_conf",
                           5, MAX_CONFIDENCE)
        _make_slider_group("Stage2 阻塞:", "confValLabel_gap", "slider_stage2_gap_conf",
                           5, MAX_CONFIDENCE)

        # objectNames for controller's findChild lookup
        self.slider_bridge_conf.setObjectName('confidenceSlider')
        self.slider_road_conf.setObjectName('confidencePostSlider')

        # Keep legacy aliases so controller code still compiles without change
        self.confidenceLable_stage1 = QtWidgets.QLabel()
        self.confidenceThresholdLabel_stage1 = self.confValLabel_bridge
        self.confidenceHorizontalSlider_stage1 = self.slider_bridge_conf
        self.confidenceLable_stage2 = QtWidgets.QLabel()
        self.confidenceThresholdLabel_stage2 = self.confValLabel_road
        self.confidenceHorizontalSlider_stage2 = self.slider_road_conf

        self.controlPanelHorizontalLayout.addLayout(self.confidenceVerticalLayout)

        # Separator
        sep3 = QtWidgets.QFrame()
        sep3.setFrameShape(QtWidgets.QFrame.HLine)
        sep3.setStyleSheet("color: #3B4252;")
        self.controlPanelHorizontalLayout.addWidget(sep3)


        # ROI buttons (set + clear)
        self.setRoiButton = QtWidgets.QPushButton(self)
        self.setRoiButton.setFont(QtGui.QFont(self.font))
        self.setRoiButton.clicked.connect(self.set_roi_clicked)
        self.controlPanelHorizontalLayout.addWidget(self.setRoiButton)

        self.clearRoiButton = QtWidgets.QPushButton(self)
        self.clearRoiButton.setFont(QtGui.QFont(self.font))
        self.clearRoiButton.clicked.connect(self.clear_roi_clicked)
        self.controlPanelHorizontalLayout.addWidget(self.clearRoiButton)

        # Apply button
        self.setPushButton = QtWidgets.QPushButton(self)
        self.setPushButton.setObjectName("setPushButton")
        self.setPushButton.setStyleSheet("background-color: #5E81AC; padding: 8px;")
        self.controlPanelHorizontalLayout.addWidget(self.setPushButton)

        self.controlPanelHorizontalLayout.addStretch()

        # Exit button (bottom)
        self.exitButton = QtWidgets.QPushButton(self)
        self.exitButton.setStyleSheet("background-color: #BF616A; color: white; padding: 8px;")
        self.exitButton.clicked.connect(self._confirm_exit)
        self.controlPanelHorizontalLayout.addWidget(self.exitButton)
        self.mainLayout.addWidget(self.controlPanelHorizontalWidget)


        self.setPushButton.clicked.connect(self._update_current_settings)

        self._retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

        self.crop_info_popup_holder = QtWidgets.QWidget()
        self.post_processing_popup_holder = [QtWidgets.QWidget() for _ in range(NUM_VIDEO_STREAMS)]

        self.update_stage_indicator(1, self.bridge_enabled)
        self.update_stage_indicator(2, self.road_enabled)
        self.update_stage_indicator(3, self.stage2_status)


    def update_stage_indicator(self, stage: int, status: bool):
        color = "green" if status else "red"
        style = f"border-radius: 8px; background-color: {color};"
        if stage == 1:
            self.bridgeIndicator.setStyleSheet(style)
        elif stage == 2:
            self.roadIndicator.setStyleSheet(style)
        elif stage == 3:
            self.stage2Indicator.setStyleSheet(style)


    def _retranslateUi(self) -> None:
        self._translate = QtCore.QCoreApplication.translate
        QFontDatabase.addApplicationFont(f"font/{self.font}.ttf")

        self.setWindowTitle(self._translate("Form", "橋樑 & 路面異常偵測系統"))
        
        icon = os.listdir("icon")
        icon = os.path.join("icon", icon[0])
        self.setWindowIcon(QtGui.QIcon(icon))

        self.setRoiButton.setText(self._translate("Form", "設定ROI"))
        self.clearRoiButton.setText(self._translate("Form", "清除ROI"))
        self.exitButton.setText(self._translate("Form", "結束程式"))


        self.setPushButton.setText(self._translate("Form", "套用設定"))
        self.setPushButton.setFont(QtGui.QFont(self.font))

        for i, location in enumerate(self.stream_locations):
            self.videoStreamLabels[i].setText(self._translate("Form", location))
            self.videoStreamLabels[i].setFont(QtGui.QFont(self.font))

        self.cropInfoListWidget.setSortingEnabled(False)

    # show the image with a pop-up window
    def show_popup(self, pixmap: QtGui.QPixmap, window_title: str, pos: QPoint=None, channel_id: int=-1) -> None:
        """ Show the video stream in a new pop-up window not using FullScreenWindow """
        if channel_id >= 0:
            if self.post_processing_popup_holder[channel_id].isVisible():
                self.post_processing_popup_holder[channel_id].resize(pixmap.width(), pixmap.height())
                self.post_processing_popup_holder[channel_id].move(pos)
                label = self.post_processing_popup_holder[channel_id].findChild(QtWidgets.QLabel)
                label.setPixmap(pixmap)
                self.post_processing_popup_holder[channel_id].activateWindow()
            else:
                self.post_processing_popup_holder[channel_id] = QtWidgets.QWidget()
                popup_holder = self.post_processing_popup_holder[channel_id]
                popup_holder.setWindowTitle(window_title)
                # popup_holder.setWindowFlags(QtCore.Qt.FramelessWindowHint) # Remove title bar
                popup_holder.resize(pixmap.width(), pixmap.height())
                label = QtWidgets.QLabel(popup_holder)
                label.setPixmap(pixmap)
                layout = QtWidgets.QVBoxLayout()
                layout.addWidget(label)
                popup_holder.setLayout(layout)
                label.setAlignment(QtCore.Qt.AlignCenter)
                popup_holder.move(pos)
                popup_holder.show()
                popup_holder.activateWindow()
        else:
            self.crop_info_popup_holder = QtWidgets.QWidget()
            popup_holder = self.crop_info_popup_holder
            popup_holder.setWindowTitle(window_title)
            popup_holder.resize(800, 600)
            label = QtWidgets.QLabel(popup_holder)
            label.setPixmap(pixmap)
            layout = QtWidgets.QVBoxLayout()
            layout.addWidget(label)
            popup_holder.setLayout(layout)
            label.setAlignment(QtCore.Qt.AlignCenter)
            self.center()
            popup_holder.show()
            popup_holder.activateWindow()
    
    def center(self):
        """ Center the window """
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # ── VLM helpers ───────────────────────────────────────────────────

    def _refresh_model_combo(self, provider: str) -> None:
        models = PROVIDER_MODELS.get(provider, [])
        self.vlmModelCombo.blockSignals(True)
        self.vlmModelCombo.clear()
        self.vlmModelCombo.addItems(models)
        self.vlmModelCombo.blockSignals(False)

    def _on_vlm_provider_changed(self, provider: str) -> None:
        self.vlm_provider = provider
        self._refresh_model_combo(provider)
        self.vlm_model = self.vlmModelCombo.currentText()
        self._update_vlm_status_label()

    def _on_vlm_toggle(self, checked: bool) -> None:
        self.vlm_enabled = checked
        if checked:
            # VLM ON → Stage2 CLS OFF
            self.stage2_status = False
            self.update_stage_indicator(3, False)
        self._update_vlm_status_label()

    def _update_vlm_status_label(self) -> None:
        if self.vlm_enabled:
            provider = self.vlmProviderCombo.currentText() if hasattr(self, 'vlmProviderCombo') else self.vlm_provider
            model = self.vlmModelCombo.currentText() if hasattr(self, 'vlmModelCombo') else self.vlm_model
            self.vlmStatusLabel.setText(f"● VLM 啟用\n{provider} / {model}")
            self.vlmStatusLabel.setStyleSheet("font-size: 10px; color: #A3BE8C;")
        else:
            self.vlmStatusLabel.setText("○ VLM 關閉（使用 Stage2 CLS）")
            self.vlmStatusLabel.setStyleSheet("font-size: 10px; color: #A0AABB;")

    # ─────────────────────────────────────────────────────────────────

    def toggle_stage_status(self, stage: int):
        if stage == 1:
            self.bridge_enabled = not self.bridge_enabled
            self.update_stage_indicator(1, self.bridge_enabled)
        elif stage == 2:
            self.road_enabled = not self.road_enabled
            self.update_stage_indicator(2, self.road_enabled)
        elif stage == 3:
            self.stage2_status = not self.stage2_status
            self.update_stage_indicator(3, self.stage2_status)

    def crop_info_clicked(self, item:QtWidgets.QListWidgetItem) -> None:
        """ Show annotated frame saved to disk when a log entry is clicked.
        Log format: '{timestamp} | {ch_key} | {location} | {detail}' """
        parts = item.text().split(' | ')
        if len(parts) < 3:
            return
        timestamp_str = parts[0]
        ch_key = parts[1]
        filename = f"{timestamp_str}_{ch_key}.jpeg"

        image = open_crop_image(
            image_name=filename,
            location=ch_key,
            dir=os.path.join(self.cfg['result_img_path'], 'annotated'),
        )
        if image is None:
            return

        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        pixmap = pixmap.scaled(
            1280,
            720,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.FastTransformation
        )
        self.show_popup(pixmap, filename)
    
    def show_post_processing_result(self, cropped_frame: np.ndarray, title: str, channel_idx: int) -> None:
        """ Show the post-processing result in a new window """
        cropped_frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        image = QImage(cropped_frame_rgb, cropped_frame_rgb.shape[1], cropped_frame_rgb.shape[0], cropped_frame_rgb.strides[0], QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        pixmap = pixmap.scaled(
            self.videoStreamWidgets[channel_idx].width() * 0.8,
            self.videoStreamWidgets[channel_idx].height() * 0.8,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.FastTransformation
        )
        # Set the position of the pop-up window to be next to video stream but not overlapping with it
        pos = self.videoStreamWidgets[channel_idx].mapToGlobal(QPoint(0, 0))
        if channel_idx % 2 == 0:
            pos.setX(pos.x() - pixmap.width() - 30)
        else:
            pos.setX(pos.x() + self.videoStreamWidgets[channel_idx].width() + 15)
        
        # print("channel_idx: ", channel_idx)
        # print(f"len(self.post_processing_popup_holder): {len(self.post_processing_popup_holder)}")
 
        self.show_popup(pixmap, title, pos, channel_idx)

    def set_roi_clicked(self) -> None:
        """ Set the ROI for the model """
        stream_location, ok = QtWidgets.QInputDialog.getItem(self, "選擇影像串流", "請選擇影像串流地點:", self.stream_locations, 0, False)
        if ok:
            stream_idx = self.stream_locations.index(stream_location)
            frame = self.last_frames[stream_idx]
            roi_dialog = ROIDialog(frame)
            if roi_dialog.exec_():
                roi = roi_dialog.get_roi()
                self.rois[stream_idx] = roi  # Store the ROI
                self.set_roi_signal.emit(stream_idx, roi)
    
    def clear_roi_clicked(self) -> None:
        """ Clear the ROI for a selected video stream """
        stream_location, ok = QtWidgets.QInputDialog.getItem(
            self, "清除ROI", "請選擇要清除ROI的影像串流:", self.stream_locations, 0, False)
        if ok:
            stream_idx = self.stream_locations.index(stream_location)
            self.rois[stream_idx] = None
            self.clear_roi_signal.emit(stream_idx)

    def _confirm_exit(self) -> None:
        """ Show confirmation dialog before exiting """
        reply = QtWidgets.QMessageBox.question(
            self, "確認結束", "確定要結束程式嗎？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.close()

    def draw_roi(self, frame: np.ndarray, roi: tuple) -> np.ndarray:
        """ Draw the ROI rectangle on the frame """
        if roi:
            x, y, w, h = roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 10)  
        return frame

    def init_values(self, conf: int, conf_post: int,
                    stage2_status: bool, conf_road: int = 30,
                    conf_height: int = 30, conf_gap: int = 10,
                    bridge_enabled: bool = True, road_enabled: bool = True,
                    vlm_enabled: bool = False, vlm_provider: str = 'gemini',
                    vlm_model: str = 'gemini-2.0-flash-lite',
                    vlm_cooldown: int = 30) -> None:
        """ Initialize the confidence values and module states """
        self.slider_bridge_conf.setValue(conf)
        self.slider_road_conf.setValue(conf_road)
        self.slider_stage2_height_conf.setValue(conf_height)
        self.slider_stage2_gap_conf.setValue(conf_gap)
        self.bridge_enabled = bridge_enabled
        self.road_enabled = road_enabled
        self.stage2_status = stage2_status
        self.update_stage_indicator(1, bridge_enabled)
        self.update_stage_indicator(2, road_enabled)
        self.update_stage_indicator(3, stage2_status)
        # VLM initial state
        self.vlm_enabled = vlm_enabled
        self.vlm_provider = vlm_provider
        self.vlm_model = vlm_model
        self.vlm_cooldown = vlm_cooldown
        self.vlmToggle.setChecked(vlm_enabled)
        self.vlmProviderCombo.setCurrentText(vlm_provider)
        self._refresh_model_combo(vlm_provider)
        self.vlmModelCombo.setCurrentText(vlm_model)
        self.vlmCooldownSpinBox.setValue(vlm_cooldown)
        self._update_vlm_status_label()
        self._update_current_settings()

    def _update_confidence_text(self) -> None:
        """ Update the confidence threshold shown on the GUI """
        self.confValLabel_bridge.setText(f"{self.slider_bridge_conf.value()}%")
        self.confValLabel_road.setText(f"{self.slider_road_conf.value()}%")
        self.confValLabel_height.setText(f"{self.slider_stage2_height_conf.value()}%")
        self.confValLabel_gap.setText(f"{self.slider_stage2_gap_conf.value()}%")

    def _update_current_settings(self) -> None:
        """ Change the current settings of the model """
        self.parameters_changed_signal.emit()
        vlm_model_txt = self.vlmModelCombo.currentText() if hasattr(self, 'vlmModelCombo') else self.vlm_model
        vlm_str = f"VLM:{vlm_model_txt}" if self.vlm_enabled else "Stage2 CLS"
        lines = [
            f"橋樑: {'開' if self.bridge_enabled else '關'}  路面: {'開' if self.road_enabled else '關'}",
            f"Stage2: {'開' if self.stage2_status else '關'}  {vlm_str}",
            f"橋樑信心: {self.slider_bridge_conf.value()}%  路面: {self.slider_road_conf.value()}%",
            f"高低差: {self.slider_stage2_height_conf.value()}%  阻塞: {self.slider_stage2_gap_conf.value()}%",
        ]
        self.currentSettingsLabel.setText("\n".join(lines))
        self.currentSettingsLabel.setFont(QtGui.QFont(self.font))
        self._update_vlm_status_label()
        
    # def _update_toggle_stage(self) -> None:

    def update_frame(self, frame: np.ndarray, ch_idx:int) -> None:
        """
            This function will take image input from 1 channels and resize it 
            only for display purpose and convert it to QImage
            to set at the label.
        """
        self.last_frames[ch_idx] = frame

        height = self.videoStreamWidgets[ch_idx].height()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (height * frame.shape[1] // frame.shape[0], height))
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        frame = pixmap
        self.videoStreamWidgets[ch_idx].setPixmap(frame)

    
    def update_fps(self, ch_idx: int, fps: float) -> None:
        self.videoStreamLabels[ch_idx].setText(f"{self.stream_locations[ch_idx]} - 預測速率: {fps:.2f} FPS")
        self.currentSettingsLabel.setFont(QtGui.QFont(self.font))


    def append_crop_info(self, info: str) -> None:
        """ Append the crop info to the list widget """

        # Activate pop-up window to show the cropped image
        # [popup_holder.activateWindow() for popup_holder in self.post_processing_popup_holder]

        # TODO: This code is not so clean, but it works. To be improved.
        
        # If there's already an item with the same info, update it
        for i in reversed(range(self.cropInfoListWidget.count())):
            item = self.cropInfoListWidget.item(i)
            info_list = info.split("偵")
            if info_list[0] in item.text():
                if len(info_list) == 2: 
                    # Detection confirmed by post-processing, so update the 'to be confirmed' log to 'confirmed'
                    item.setText(info)
                else: 
                    # Detection rejected by post-processing, so remove the 'to be confirmed' log
                    self.cropInfoListWidget.takeItem(i)
                return

        # Detection under post-processing, so add the 'to be confirmed' log
        self.cropInfoListWidget.addItem(info)
    
    
    def mouseReleaseEvent(self, event) -> None:
        """ Show single video stream when clicked """
        
        for i, w in enumerate(self.videoStreamWidgets):
            if w.pixmap() is None: # No image is displayed yet
                return
            if w.underMouse(): # Show single video stream when clicked
                # hide others
                if not self.is_single_channel:
                    frame = self.last_frames[i]
                    frame = cv2.resize(frame, (w.width() * 2 , w.height() * 2 + self.videoStreamLabels[i].height()))
                    # frame = cv2.resize(frame, (self.videoStreamWidgets[i].height() * frame.shape[1] // frame.shape[0], self.videoStreamWidgets[i].height()))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
                    pixmap = QtGui.QPixmap.fromImage(image)
                    w.setPixmap(pixmap)
                    for j, w in enumerate(self.videoStreamWidgets):
                        if i != j:
                            w.hide()
                            self.videoStreamLabels[j].hide()
                    self.is_single_channel = True
                else: # Show all video streams when clicked again
                    frame = self.last_frames[i]
                    frame = cv2.resize(frame, (w.width() // 2, (w.height() - self.videoStreamLabels[i].height()) // 2))
                    # frame = cv2.resize(frame, (self.videoStreamWidgets[i].height() * frame.shape[1] // frame.shape[0], self.videoStreamWidgets[i].height()))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
                    pixmap = QtGui.QPixmap.fromImage(image)
                    w.setPixmap(pixmap)
                    # w.hide()
                    for j, w in enumerate(self.videoStreamWidgets):
                        w.show()
                        self.videoStreamLabels[j].show()
                    self.is_single_channel = False
                break
        
        event.accept()
            

    def closeEvent(self, event) -> None:
        """ Emit the windowClosed signal when the window is closed """
        self.window_closed_signal.emit()
        self.crop_info_popup_holder.close()
        [popup_holder.close() for popup_holder in self.post_processing_popup_holder]
        event.accept()
    

    def _load_roi(self):
        """Load ROI from the json configuration file."""
        cfg_path = 'cfg/'
        for i in range(NUM_VIDEO_STREAMS):
            self.roi_cfg_file = os.path.join(cfg_path, f'roi_ch{i+1}.json')
            if os.path.exists(self.roi_cfg_file):
                with open(self.roi_cfg_file, 'r') as f:
                    self.rois[i] = json.load(f).get('roi', None)
