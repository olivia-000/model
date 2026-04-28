from datetime import datetime
from time import perf_counter, sleep
from typing import Tuple
from threading import Thread

import cv2
import torch
from ultralytics import YOLO, RTDETR

# My modules
from .detector import Detector
from .vlm_worker import VLMWorker
from utils import yaml_load, yaml_save
from utils import (
    DEFAULT_CFG,
    COSTUMIZED_CFG,
    NUM_VIDEO_STREAMS,
    MULTI_MODEL,
    LIVE_VIDEO_STREAM,
    VIDEO_DEMO
)



class Detection():
    """
    The model class of the firetruck detection system 
    It is responsible for handling threads detecting firetruck in the live video streams. 
    """    
    def __init__(
            self, 
            cfg_path: str=DEFAULT_CFG,
        ) -> None:
        """
        Initialize the firetruck detection model class.

        Args:
            cfg_path (str): The path of the configuration file.
        """
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        print(f'Loading configuration file {cfg_path}... ')
        self._cfg_setup(yaml_load(cfg_path))

        # Multiple models for multiple video streams
        self.workers = [
            Detector(
                cfg_dict=self.cfg,
                worker_idx=i
            )
            for i in range(NUM_VIDEO_STREAMS)
        ]

        # VLM worker (shared across channels) — always started; idle when VLM method is off
        self.vlm_worker = VLMWorker(self.cfg)
        vlm_channels = self.cfg.get('vlm', {}).get('vlm_channels', ['ch1', 'ch3', 'ch4'])
        for w in self.workers:
            if w.ch_key in vlm_channels:
                w.vlm_worker = self.vlm_worker
        self.vlm_worker.start()

        if not MULTI_MODEL: # Batch inference mode
            self.processed_frames_num = 0
            self.detection_t = Thread(target=self._inference_worker, name='detection')


    def _cfg_setup(self, cfg_dict: dict=None) -> None:
        """ Import settings with the specified configuration file path """
        self.cfg = cfg_dict

        # Model settings — read all three module enable states from YAML
        dm = self.cfg.get('detection_modules', {})
        self.bridge_enabled = dm.get('bridge', {}).get('enable', True)
        self.road_enabled = dm.get('road_damage', {}).get('enable', True)
        self.stage2_status = dm.get('bridge', {}).get('stage2', {}).get('enable', False)
        self.conf = self.cfg['confidence']
        self.conf_post = self.cfg['confidence_post']
        self.iou = self.cfg['iou']

        self.weight = self.cfg['weight']

        # Per-module runtime conf (updated by GUI sliders)
        dm = self.cfg.get('detection_modules', {})
        self.bridge_conf = dm.get('bridge', {}).get('conf', 0.3)
        self.road_conf = dm.get('road_damage', {}).get('conf', 0.3)
        s2_per_ch = dm.get('bridge', {}).get('stage2', {}).get('per_channel', {})
        height_ch = next((v for v in s2_per_ch.values() if v.get('task') == 'height'), {})
        gap_ch = next((v for v in s2_per_ch.values() if v.get('task') == 'gap'), {})
        self.stage2_height_conf = height_ch.get('conf', 0.3)
        self.stage2_gap_conf = gap_ch.get('conf', 0.1)

        # ── M2: 讀取 detection_modules 開關 ──────────────────────────
        self._load_detection_modules()

    def _load_detection_modules(self) -> None:
        """ 讀取並驗證 detection_modules 設定，打印啟用狀態 + per-channel 分配 """
        dm = self.cfg.get('detection_modules', {})

        rd = dm.get('road_damage', {})
        self.road_damage_enabled = rd.get('enable', False)
        self.road_damage_cfg = rd

        br = dm.get('bridge', {})
        self.bridge_enabled = br.get('enable', False)
        self.bridge_cfg = br
        stage2 = br.get('stage2', {})
        self.bridge_stage2_enabled = stage2.get('enable', False)

        self.channel_rules = self.cfg.get('channel_rules', {})

        def _fmt(mod_cfg, label):
            on = 'ON ' if mod_cfg.get('enable', False) else 'OFF'
            chs = mod_cfg.get('channels', [])
            desc = mod_cfg.get('description', '')
            return (f"  {label:<12}: {on} | channels: {chs}\n"
                    f"               weight: {mod_cfg.get('weight', 'N/A')} | conf: {mod_cfg.get('conf', 'N/A')}\n"
                    f"               {desc}")

        print("─── Detection Modules ───────────────────────────────")
        print(_fmt(rd, "Road Damage"))
        print(_fmt(br, "Bridge Stg1"))
        s2_chs = list(stage2.get('per_channel', {}).keys())
        print(f"  Bridge Stg2 : {'ON ' if self.bridge_stage2_enabled else 'OFF'} | channels: {s2_chs}")
        print(f"               method: {stage2.get('method', 'N/A')} | trigger cls: {stage2.get('trigger_class_ids', [])}")
        print(f"  Channel rules: {list(self.channel_rules.keys())}")
        print("─────────────────────────────────────────────────────")


    # ── Per-module conf getters / setters ────────────────────────────
    def get_bridge_conf(self) -> float:
        return self.bridge_conf

    def get_road_conf(self) -> float:
        return self.road_conf

    def get_stage2_height_conf(self) -> float:
        return self.stage2_height_conf

    def get_stage2_gap_conf(self) -> float:
        return self.stage2_gap_conf

    def set_bridge_conf(self, conf: float) -> None:
        self.bridge_conf = conf
        self.cfg['detection_modules']['bridge']['conf'] = conf
        yaml_save(COSTUMIZED_CFG, self.cfg)

    def set_road_conf(self, conf: float) -> None:
        self.road_conf = conf
        self.cfg['detection_modules']['road_damage']['conf'] = conf
        yaml_save(COSTUMIZED_CFG, self.cfg)

    def set_stage2_height_conf(self, conf: float) -> None:
        self.stage2_height_conf = conf
        for ch_cfg in self.cfg['detection_modules']['bridge']['stage2']['per_channel'].values():
            if ch_cfg.get('task') == 'height':
                ch_cfg['conf'] = conf
        yaml_save(COSTUMIZED_CFG, self.cfg)
        for w in self.workers:
            if w.stage2_cfg and w.stage2_cfg.get('task') == 'height':
                w.post_processing_worker.set_conf(conf)

    def set_stage2_gap_conf(self, conf: float) -> None:
        self.stage2_gap_conf = conf
        for ch_cfg in self.cfg['detection_modules']['bridge']['stage2']['per_channel'].values():
            if ch_cfg.get('task') == 'gap':
                ch_cfg['conf'] = conf
        yaml_save(COSTUMIZED_CFG, self.cfg)
        for w in self.workers:
            if w.stage2_cfg and w.stage2_cfg.get('task') == 'gap':
                w.post_processing_worker.set_conf(conf)
    # ─────────────────────────────────────────────────────────────────

    def get_bridge_enabled(self) -> bool:
        return self.bridge_enabled

    def get_road_enabled(self) -> bool:
        return self.road_enabled

    def set_bridge_enabled(self, enabled: bool) -> None:
        self.bridge_enabled = enabled
        self.cfg['detection_modules']['bridge']['enable'] = enabled
        yaml_save(COSTUMIZED_CFG, self.cfg)
        br_channels = self.cfg.get('detection_modules', {}).get('bridge', {}).get('channels', [])
        for w in self.workers:
            w.bridge_enabled = enabled
            w.run_bridge = enabled and w.ch_key in br_channels

    def set_road_enabled(self, enabled: bool) -> None:
        self.road_enabled = enabled
        self.cfg['detection_modules']['road_damage']['enable'] = enabled
        yaml_save(COSTUMIZED_CFG, self.cfg)
        rd_channels = self.cfg.get('detection_modules', {}).get('road_damage', {}).get('channels', [])
        for w in self.workers:
            w.run_road_damage = enabled and w.ch_key in rd_channels

    def set_stage2_status(self, stage2_status: bool) -> None:
        self.stage2_status = stage2_status
        self.cfg['detection_modules']['bridge']['stage2']['enable'] = stage2_status
        yaml_save(COSTUMIZED_CFG, self.cfg)
        for w in self.workers:
            w.stage2_enabled = stage2_status
        if MULTI_MODEL:
            for worker in self.workers:
                worker.set_stage2_status(self.stage2_status)

    # ── VLM methods ──────────────────────────────────────────────────

    def get_vlm_enabled(self) -> bool:
        return self.cfg.get('detection_modules', {}).get('bridge', {}).get('stage2', {}).get('method', 'yolo_cls') == 'vlm'

    def get_vlm_provider(self) -> str:
        return self.cfg.get('vlm', {}).get('provider', 'gemini')

    def get_vlm_model(self) -> str:
        return self.cfg.get('vlm', {}).get('model', 'gemini-2.0-flash-lite')

    def set_vlm_enabled(self, enabled: bool) -> None:
        """Switch Stage2 method between 'vlm' and 'yolo_cls'."""
        method = 'vlm' if enabled else 'yolo_cls'
        self.cfg['detection_modules']['bridge']['stage2']['method'] = method
        yaml_save(COSTUMIZED_CFG, self.cfg)
        for w in self.workers:
            w.stage2_method = method
        print(f"[VLM] set_vlm_enabled={enabled} → method={method}")

    def set_vlm_provider_model(self, provider: str, model_name: str) -> None:
        self.cfg['vlm']['provider'] = provider
        self.cfg['vlm']['model'] = model_name
        yaml_save(COSTUMIZED_CFG, self.cfg)
        self.vlm_worker.update_provider_model(provider, model_name)

    def get_vlm_cooldown(self) -> int:
        return self.cfg.get('vlm', {}).get('cooldown_sec', 30)

    def set_vlm_cooldown(self, cooldown_sec: int) -> None:
        self.cfg['vlm']['cooldown_sec'] = cooldown_sec
        yaml_save(COSTUMIZED_CFG, self.cfg)
        self.vlm_worker.update_cooldown(cooldown_sec)
    # ─────────────────────────────────────────────────────────────────


    def set_confidence(self, conf: float) -> None:
        """ Set the confidence threshold """
        self.conf = conf
        self.cfg['confidence'] = conf
        yaml_save(COSTUMIZED_CFG, self.cfg)
        if MULTI_MODEL:
            for worker in self.workers:
                worker.set_confidence(conf)

    def set_confidencePost(self, conf_post: float) -> None:
        """ Set the confidence threshold """
        self.conf_post = conf_post
        self.cfg['confidence_post'] = conf_post
        yaml_save(COSTUMIZED_CFG, self.cfg)
        if MULTI_MODEL:
            for worker in self.workers:
                worker.set_confidence(conf_post)


    def get_location(self, video_stream_idx:int) -> str:
        """ Get the location of the video stream """
        return self.workers[video_stream_idx].video_stream.location

    def get_stage2_status(self) -> bool:
        """ Get the stage 2 status """
        return self.stage2_status


    def get_confidence(self) -> float:
        """ Get the confidence threshold """
        return self.cfg['confidence']
    
    def get_confidence_post(self) -> float:
        """ Get the confidence post threshold """
        return self.cfg['confidence_post']
    
    def start(self) -> None:
        """ Run the firetruck detection worker threads """
        self.is_running = True
        for video_stream_idx in range(NUM_VIDEO_STREAMS):
            self.workers[video_stream_idx].start()
        if not MULTI_MODEL:
            self.detection_t.start()        


    def stop(self) -> None:
        """ Stop the running thread of the firetruck detection """
        self.is_running = False
        for i in range(NUM_VIDEO_STREAMS):
            self.workers[i].stop()
        self.vlm_worker.stop()


    def set_roi(self, video_stream_idx:int, roi:Tuple[int, int, int, int]) -> None:
        """ Set the region of interest """
        self.workers[video_stream_idx].set_roi(roi)

    def clear_roi(self, video_stream_idx: int) -> None:
        """ Clear the ROI and restore full-frame detection """
        self.workers[video_stream_idx].clear_roi()


    #============================== Methods used in batch inference mode ==============================#
    def _setup_model(self, weight) -> torch.nn.Module:
        """ 根據權重名稱自動判斷模型類別 """
        # 移除副檔名並轉小寫進行判斷
        weight_name = weight.lower()
    
        if 'rtdetr' in weight_name:
            print(f"检测到 RT-DETR 權重，使用 RTDETR 類別載入: {weight}")
            model = RTDETR(f'weights/{weight}.pt')
        else:
            print(f"使用 YOLO 類別載入: {weight}")
            model = YOLO(f'weights/{weight}.pt')
            model.fuse() # RT-DETR 通常不需手動執行 fuse()
        dummy_input = torch.zeros(1, 3, 640, 640).to(model.device)
        model.predict(source=dummy_input, verbose=False)
        return model


    def _inference_worker(self) -> None:
        """
        Batch inference worker: road damage on ch2/ch5, bridge model on all channels (M4+).
        """
        print('Inference worker is loading models... ')

        # Always load models if weight is specified; enable flag only controls frame routing
        self.road_model = None
        rd_cfg = self.cfg.get('detection_modules', {}).get('road_damage', {})
        if rd_cfg.get('weight'):
            self.road_model = self._setup_model(rd_cfg['weight'])
            print(f"Road damage model ready: {rd_cfg['weight']}")

        self.bridge_model = None
        br_cfg = self.cfg.get('detection_modules', {}).get('bridge', {})
        if br_cfg.get('weight'):
            self.bridge_model = self._setup_model(br_cfg['weight'])
            print(f"Bridge model ready: {br_cfg['weight']}")
            print(f"Bridge model classes: {self.bridge_model.names}")

        for i in range(NUM_VIDEO_STREAMS):
            self.workers[i].post_processing_worker.setup_model()
        print('Models are ready.')

        frames = [None] * NUM_VIDEO_STREAMS
        for i in range(NUM_VIDEO_STREAMS):
            frames[i] = self.workers[i].video_stream.loading_screen
            self.workers[i].video_stream.capture_next()
            self.workers[i].fetch_thread.start()

        # FPS cap: use source video FPS (already read by VideoStream), fallback 30
        source_fps = min(
            (w.video_stream.fps for w in self.workers if w.video_stream.fps > 0),
            default=30.0
        )
        frame_interval = 1.0 / source_fps
        print(f'Start detecting... (target FPS cap: {source_fps:.1f})')
        while self.is_running:
            t_start = perf_counter()
            for i in range(NUM_VIDEO_STREAMS):
                if self.workers[i].last_frame is not None:
                    frames[i] = self.workers[i].last_frame
                    self.workers[i].last_frame = None

            inference_timestamp = datetime.now()
            self.processed_frames_num += 1

            # Bridge inference — only on channels flagged run_bridge, batched
            main_results = [None] * NUM_VIDEO_STREAMS
            if self.bridge_model is not None:
                bridge_idxs = [i for i in range(NUM_VIDEO_STREAMS) if self.workers[i].run_bridge]
                if bridge_idxs:
                    br = self.bridge_model([frames[i] for i in bridge_idxs],
                                           conf=self.bridge_conf, iou=self.iou, verbose=False)
                    for j, i in enumerate(bridge_idxs):
                        main_results[i] = br[j]

            # Road damage inference — only on channels flagged run_road_damage
            road_results = [None] * NUM_VIDEO_STREAMS
            if self.road_model is not None:
                for i in range(NUM_VIDEO_STREAMS):
                    if self.workers[i].run_road_damage:
                        r = self.road_model([frames[i]], conf=self.road_conf, iou=self.iou, verbose=False)
                        road_results[i] = r[0]

            for i, frame in enumerate(frames):
                self.workers[i].result_queue.put((main_results[i], road_results[i], frame, inference_timestamp))

            elapsed = perf_counter() - t_start
            if elapsed < frame_interval:
                sleep(frame_interval - elapsed)
