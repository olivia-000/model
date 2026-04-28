import torch
import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject
from ultralytics import YOLO

from .video_stream import VideoStream


class PostProcessingWorker(QObject):
    """ Stage 2 classification worker for bridge anomaly detection. """
    add_crop_info_signal = pyqtSignal(str)
    show_post_processing_result_signal = pyqtSignal(np.ndarray, str)

    def __init__(self, cfg_dict: dict, video_stream: VideoStream, video_stream_idx: int = -1) -> None:
        super().__init__()
        self.video_stream_idx = video_stream_idx
        self.ch_key = f'ch{video_stream_idx + 1}'
        self._cfg_setup(cfg_dict)
        self.video_stream = video_stream


    def _cfg_setup(self, cfg_dict: dict):
        self.cfg = cfg_dict
        stage2 = self.cfg.get('detection_modules', {}).get('bridge', {}).get('stage2', {})
        self.bridge_stage2_enabled = stage2.get('enable', False)
        self.bridge_stage2_cfg = stage2.get('per_channel', {}).get(self.ch_key, None)
        self.stage2_conf = self.bridge_stage2_cfg.get('conf', 0.3) if self.bridge_stage2_cfg else 0.3

    def set_conf(self, conf: float) -> None:
        self.stage2_conf = conf


    def setup_model(self) -> None:
        self.bridge_stage2_model = None
        if self.bridge_stage2_cfg is not None:
            weight_name = self.bridge_stage2_cfg['weight']
            s2_model = YOLO(f"weights/{weight_name}.pt")
            s2_model.predict(source=torch.zeros(1, 3, 224, 224).to(s2_model.device), verbose=False)
            self.bridge_stage2_model = s2_model
            task = self.bridge_stage2_cfg.get('task', '?')
            print(f"  {self.ch_key} Stage2 CLS ready: {weight_name} (task={task}) | classes: {s2_model.names}")


    def classify_crop(self, crop: np.ndarray):
        """Run Stage2 CLS on a cropped bbox. Returns (label, conf) or None."""
        if self.bridge_stage2_model is None or crop is None or crop.size == 0:
            return None
        conf_thresh = self.stage2_conf
        result = self.bridge_stage2_model(crop, verbose=False)[0]
        probs = result.probs
        if probs is None:
            return None
        top1 = int(probs.top1)
        top1_conf = float(probs.top1conf)
        if top1_conf < conf_thresh:
            return None
        return result.names[top1], top1_conf
