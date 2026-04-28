import os
from datetime import datetime
from time import sleep
from typing import Tuple, List
from queue import Queue, Full
from threading import Thread
from collections import deque

import json

import cv2
import torch
from ultralytics import YOLO, RTDETR
from ultralytics.engine.results import Results
from ultralytics.utils.plotting import Annotator, colors
import numpy as np

# My modules
from .video_stream import VideoStream
from .post_processing import PostProcessingWorker
from .vlm_worker import VLMWorker
from utils import (
    DEFAULT_CFG,
    VIEW_DOWNSCALE_RATIO,
    CROPPING_THRESHOLD,
    MULTI_MODEL,
    CAMERA_DIRECTIONS,
    save_image,
)



class Detector():
    """ 
        The worker class to handle the video stream detection.
        
        Args:
            frame (np.ndarray): The frame to be updated.
            worker_idx (int): The index of the worker.
            post_processing_worker (PostProcessingWorker): The shared post-processing worker instance.
            post_processing_lock (Lock): The shared lock instance for the post-processing worker.

    """
    def __init__(
            self, 
            cfg_dict: dict=DEFAULT_CFG, 
            worker_idx: int=-1
        ) -> None:

        self.video_stream_idx = worker_idx

        # This flag is used to stop the running threads when the program is closed.
        self._is_running = False

        # Setup configuration from the configuration file
        self._cfg_setup(cfg_dict) 
        self.roi_cfg_file = f'cfg/roi_ch{self.video_stream_idx+1}.json'
        self.roi = None
        self._load_roi()

        
        # Now initialize video stream to handle the video source
        self.video_stream = VideoStream(self.video_stream_idx, self.cfg)
        self.post_processing_worker = PostProcessingWorker(self.cfg, self.video_stream, self.video_stream_idx)
        self.vlm_worker: VLMWorker | None = None  # injected by Detection after construction

        # Buffer holding the frame thats fetched most recently, processed by the inference worker.
        self.last_frame = None
        # self.last_frame_buffer = Queue(maxsize=1)

        # Queue holds the inference results that need to be processed by result worker.
        # self.result_buffer = None 
        self.result_queue = Queue(maxsize=5)

        # Initialize result, fetch and inference worker threads as needed
        self.result_thread = Thread(target=self._results_worker, name=f'ch{self.video_stream_idx}_result', daemon=True)
        self.fetch_thread = Thread(target=self._fetch_frame_worker, name=f'ch{self.video_stream_idx}_fetch', daemon=True)
        if MULTI_MODEL:
            self.inference_thread = Thread(target=self._inference_worker, name=f'ch{self.video_stream_idx}_inference', daemon=True)


    def _cfg_setup(self, cfg_dict=None) -> None:
        """ Import settings with the specified configuration file path """
        self.cfg = cfg_dict
        self.stage2_status = self.cfg.get('detection_modules', {}).get('bridge', {}).get('stage2', {}).get('enable', False)

        # ── Declarative per-channel module setup (driven by YAML) ─────
        self.direction = CAMERA_DIRECTIONS[self.video_stream_idx] if self.video_stream_idx < len(CAMERA_DIRECTIONS) else 'unknown'
        self.ch_key = f'ch{self.video_stream_idx + 1}'
        channel_rules = self.cfg.get('channel_rules', {})
        self.ch_rule = channel_rules.get(self.ch_key, {})

        dm = self.cfg.get('detection_modules', {})

        rd = dm.get('road_damage', {})
        self.road_damage_enabled = rd.get('enable', False)
        self.run_road_damage = self.road_damage_enabled and self.ch_key in rd.get('channels', [])

        br = dm.get('bridge', {})
        self.bridge_enabled = br.get('enable', False)
        self.run_bridge = self.bridge_enabled and self.ch_key in br.get('channels', [])

        # Stage 2 config for this specific channel (None if this channel has no Stage 2)
        stage2 = br.get('stage2', {})
        self.stage2_enabled = stage2.get('enable', False)
        self.stage2_method = stage2.get('method', 'yolo_cls')   # 'yolo_cls' | 'vlm' | 'none'
        self.stage2_cfg = stage2.get('per_channel', {}).get(self.ch_key, None)
        self.stage2_trigger_class_ids = stage2.get('trigger_class_ids', [])
        # ─────────────────────────────────────────────────────────────

        # Setup parameters for the model for each video stream
        if MULTI_MODEL:
            self.weight = self.cfg['weight']
            self.conf = self.cfg['confidence']
            self.conf_post = self.cfg['confidence_post']
            self.iou = self.cfg['iou']
            self.processed_frames_num = 0
    

    def _load_roi(self):
        """Load ROI from the json configuration file."""
        if os.path.exists(self.roi_cfg_file):
            with open(self.roi_cfg_file, 'r') as f:
                self.roi = json.load(f).get('roi', None)


    def _save_roi(self):
        """Save ROI to the json configuration file."""
        with open(self.roi_cfg_file, 'w') as f:
            json.dump({'roi': self.roi}, f)
            

    def set_roi(self, roi:Tuple[int, int, int, int]) -> None:
        """ Set the region of interest """
        self.roi = roi
        self._save_roi()

    def clear_roi(self) -> None:
        """Clear ROI and delete the JSON file to restore full-frame detection."""
        self.roi = None
        if os.path.exists(self.roi_cfg_file):
            os.remove(self.roi_cfg_file)

    def _bbox_center_in_roi(self, box_xyxy) -> bool:
        """Return True if bbox center falls within self.roi; always True when roi is None."""
        if self.roi is None:
            return True
        rx, ry, rw, rh = self.roi
        x1, y1, x2, y2 = box_xyxy[:4]
        cx = (float(x1) + float(x2)) / 2
        cy = (float(y1) + float(y2)) / 2
        return rx <= cx <= rx + rw and ry <= cy <= ry + rh

    def set_stage2_status(self, stage2_status: bool) -> None:
        """
        Set the stage 2 status, which is the status of the post-processing worker.
        """
        self.stage2_status = stage2_status
        # print(f"Stage 2 status: {self.stage2_status}")

    def set_confidence(self, conf:float) -> None:
        """ Set the confidence threshold """
        self.conf = conf

    def set_confidencePost(self, conf_post:float) -> None:
        """ Set the confidence Post threshold """
        self.conf_post = conf_post

    def start(self) -> None:
        """ Start the running thread of the firetruck detection """

        self._is_running = True

        if MULTI_MODEL:
            self.inference_thread.start()

        self.result_thread.start()


    def stop(self) -> None:
        """ Stop the running thread of the firetruck detection """
        self._is_running = False
        self.video_stream.stop()
        print(f'video stream {self.video_stream_idx+1} stopped')


    def _fetch_frame_worker(self) -> None:
        """
        Grab the video frames of a single video stream.
        """
        print(f'video stream {self.video_stream_idx+1} start fetching frames... ')

        while self._is_running:
            success, frame = self.video_stream.fetch_next_frame()
            if self.cfg['source'] == 0 :
                sleep(1/10)
            if not success:
                #self.stop()
                self.video_stream.capture_next()
            self.last_frame = frame


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
    
    
    def _process_frame(self, frame) -> List[Results]:
        """ ROI cropping and inference on a single frame """
        # [DEPRECATED] Downscale the frame for the model to process if needed

        self.processed_frames_num += 1 # For profiling, but not used for now

        # Crop the frame to the region of interest (ROI) if specified
        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
        
        return results


    def _inference_worker(self) -> None:
        """
        The inference thread: get the frame from the buffer, preprocess it, and run inference.
        """
        # Load and warm up the model before starting the inference thread
        print(f'Inference worker for video stream {self.video_stream_idx+1} is loading and warming up the models... ')
        self.model = self._setup_model(self.weight)
        self.post_processing_worker.setup_model()
        print(f'Inference worker for video stream {self.video_stream_idx+1} models are ready.')

        self.video_stream.capture_next()

        # Start inference loop
        while self._is_running:
            # Get the latest frame from the soucre
            success, frame = self.video_stream.fetch_next_frame()
            # frame = self.last_frame_buffer.get()

            inference_timestamp = datetime.now()
            results = self._process_frame(frame)

            # Put the results into the result queue for the result worker to process
            # self.result_buffer = (results[0], frame, inference_timestamp)
            try:
                self.result_queue.put((results[0], None, frame, inference_timestamp))
            except Full:
                print(f"video stream {self.video_stream_idx+1} result queue is full, skipping...")
                continue

            # If the result thread is killed unexpectedly or just for debugging
            if not self.result_thread.is_alive():
                self._process_results()


    def _results_worker(self) -> None:
        """
        Process the inference results for detection
        """
        print(f'video stream {self.video_stream_idx+1} start processing results... ')
        while self._is_running:
            self._process_results() #第一階段
    

    def _process_results(self) -> None:
        """ Process inference results: bridge (red/green) + road damage (orange) merged on frame. """
        main_results, road_results, frame, inference_timestamp = self.result_queue.get()

        frame_h = frame.shape[0]
        annotated_frame = frame.copy()

        # Draw bridge bboxes first (anomaly=red, normal=green)
        if main_results is not None and self.bridge_enabled:
            annotated_frame = self._plot_bridge_bboxes(main_results, annotated_frame, inference_timestamp)

        # Draw road damage bboxes on top (orange), ch2/ch5 only
        if road_results is not None and self.run_road_damage:
            annotated_frame = self._plot_road_bboxes(road_results, annotated_frame, frame_h, inference_timestamp)

        # Draw ROI boundary for reference
        if self.roi is not None:
            rx, ry, rw, rh = self.roi
            cv2.rectangle(annotated_frame, (rx, ry), (rx + rw, ry + rh), (255, 255, 255), 2)

        # Show annotated frame in GUI
        showed_frame = annotated_frame
        if VIEW_DOWNSCALE_RATIO > 1:
            showed_frame = cv2.resize(
                annotated_frame,
                (annotated_frame.shape[1] // VIEW_DOWNSCALE_RATIO, annotated_frame.shape[0] // VIEW_DOWNSCALE_RATIO)
            )
        if self._is_running:
            self.video_stream.update_frame(showed_frame)

    def _plot_bridge_bboxes(self, bridge_results, frame: np.ndarray, inference_timestamp) -> np.ndarray:
        """Draw bridge bboxes: anomaly classes in red, normal classes in green.
        Stage1-only anomalies and Stage2-confirmed anomalies are logged and saved."""
        anomaly_class_ids = self.cfg.get('detection_modules', {}).get('bridge', {}).get('desired_class_ids', [])
        allowed_class_ids = self.ch_rule.get('bridge_allowed_class_ids', None)

        boxes_xyxy = bridge_results.boxes.xyxy.cpu()
        clss = bridge_results.boxes.cls.cpu().tolist()
        confs = bridge_results.boxes.conf.cpu().tolist()
        names = bridge_results.names

        orig_frame = frame.copy()  # clean snapshot for Stage2 crops; also reserved for future original-save
        run_cls  = self.stage2_enabled and self.stage2_cfg is not None and self.stage2_method == 'yolo_cls'
        run_vlm  = self.vlm_worker is not None and self.stage2_method == 'vlm'

        annotator = Annotator(frame, line_width=5, font_size=10)
        stage1_anomalies = []  # (s1_label, s1_conf)           — non-trigger anomaly classes
        stage2_anomalies = []  # (s2_label, s2_conf, s1_label, s1_conf) — Stage2 CLS confirmed
        vlm_jobs = []          # jobs to submit after all bboxes drawn

        for box_xyxy, cls, conf in zip(boxes_xyxy, clss, confs):
            orig_cls_int = int(cls)
            if allowed_class_ids is not None and orig_cls_int not in allowed_class_ids:
                continue
            if not self._bbox_center_in_roi(box_xyxy):
                continue

            display_cls_int = orig_cls_int
            label = f"{names[orig_cls_int]}: {conf:.2f}"

            if run_cls and orig_cls_int in self.stage2_trigger_class_ids:
                x1, y1, x2, y2 = map(int, box_xyxy.tolist())
                crop = orig_frame[max(0, y1):y2, max(0, x1):x2]
                s2 = self.post_processing_worker.classify_crop(crop)
                if s2 is not None:
                    s2_label, s2_conf = s2
                    normal_name = self.stage2_cfg.get('normal_class_name', '')
                    if s2_label != normal_name:
                        display_cls_int = 6
                        stage2_anomalies.append((s2_label, s2_conf, names[orig_cls_int], conf))
                    label = f"{names[orig_cls_int]}: {conf:.2f} | {s2_label}:{s2_conf:.2f}"
            elif run_vlm and orig_cls_int in self.stage2_trigger_class_ids:
                # VLM mode: draw Stage1 bbox in orange (pending), queue async job
                x1, y1, x2, y2 = map(int, box_xyxy.tolist())
                crop = orig_frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop.size == 0:
                    # degenerate bbox — skip VLM, fall back to plain Stage1 display
                    pass
                else:
                    label = f"{names[orig_cls_int]}: {conf:.2f} [LLM...]"
                    display_cls_int = -1  # orange = pending
                    vlm_jobs.append({
                        'crop': crop,
                        'box_xyxy': box_xyxy,
                        'stage1_label': names[orig_cls_int],
                    })
            elif orig_cls_int in anomaly_class_ids:
                # Stage1-only anomaly: not a Stage2 trigger, log immediately
                stage1_anomalies.append((names[orig_cls_int], conf))

            if display_cls_int == -1:
                color = (0, 165, 255)  # orange = VLM pending
            else:
                color = (0, 0, 255) if display_cls_int in anomaly_class_ids else (0, 255, 0)
            annotator.box_label(box_xyxy, label=label, color=color)

        # ── Save & log after all bboxes are drawn ────────────────────
        result_dir = self.cfg.get('result_img_path', 'detection_result_image_data')
        timestamp_str = inference_timestamp.strftime('%Y-%m-%d_%H-%M-%S')
        frame_filename = f"{timestamp_str}_{self.ch_key}.jpeg"

        if stage1_anomalies or stage2_anomalies:
            # Currently: save annotated frame only
            save_image(frame, frame_filename, self.ch_key, f"{result_dir}/annotated")
            # Future extension:
            # save_image(orig_frame, frame_filename, self.ch_key, f"{result_dir}/original")

            for s1_label, s1_conf in stage1_anomalies:
                msg = (f"{timestamp_str} | {self.ch_key} | {self.video_stream.location} | "
                       f"{s1_label} ({s1_conf:.2f})")
                self.post_processing_worker.add_crop_info_signal.emit(msg)

            for s2_label, s2_conf, s1_label, s1_conf in stage2_anomalies:
                msg = (f"{timestamp_str} | {self.ch_key} | {self.video_stream.location} | "
                       f"{s1_label} → {s2_label} ({s2_conf:.2f})")
                self.post_processing_worker.add_crop_info_signal.emit(msg)

        if vlm_jobs:
            # Save Stage1 annotated frame first; VLM worker will overwrite with result
            save_image(frame, frame_filename, self.ch_key, f"{result_dir}/annotated")
            for job in vlm_jobs:
                self.vlm_worker.submit(
                    ch_key=self.ch_key,
                    crop=job['crop'],
                    annotated_frame=frame,
                    box_xyxy=job['box_xyxy'],
                    stage1_label=job['stage1_label'],
                    timestamp=inference_timestamp,
                    location=self.video_stream.location,
                )

        return frame

    def _plot_road_bboxes(self, road_results, frame: np.ndarray, frame_h: int, inference_timestamp) -> np.ndarray:
        """Draw road damage bboxes in orange, applying bbox_y ratio filter. All detections are logged and saved."""
        y_ratio = self.ch_rule.get('road_filter_bbox_y_ratio', None)
        desired_classes = self.cfg.get('detection_modules', {}).get('road_damage', {}).get('desired_class_ids', None)

        boxes_xyxy = road_results.boxes.xyxy.cpu()
        boxes_xywh = road_results.boxes.xywh.cpu()
        clss = road_results.boxes.cls.cpu().tolist()
        confs = road_results.boxes.conf.cpu().tolist()
        names = road_results.names

        annotator = Annotator(frame, line_width=5, font_size=10)
        road_detections = []  # (label, conf)

        for box_xyxy, box_xywh, cls, conf in zip(boxes_xyxy, boxes_xywh, clss, confs):
            cls_int = int(cls)
            if desired_classes is not None and cls_int not in desired_classes:
                continue
            center_y = box_xywh[1].item()
            if y_ratio is not None and center_y > frame_h * y_ratio:
                continue
            if not self._bbox_center_in_roi(box_xyxy):
                continue
            label = f"{names[cls_int]}: {conf:.2f}"
            annotator.box_label(box_xyxy, label=label, color=(0, 165, 255))  # orange BGR
            road_detections.append((names[cls_int], conf))

        # ── Save & log after all bboxes are drawn ────────────────────
        if road_detections:
            result_dir = self.cfg.get('result_img_path', 'detection_result_image_data')
            timestamp_str = inference_timestamp.strftime('%Y-%m-%d_%H-%M-%S')
            frame_filename = f"{timestamp_str}_{self.ch_key}.jpeg"

            # Currently: save annotated frame only
            save_image(frame, frame_filename, self.ch_key, f"{result_dir}/annotated")
            # Future extension:
            # save_image(orig_frame, frame_filename, self.ch_key, f"{result_dir}/original")
            # for idx, crop in enumerate(crops):
            #     save_image(crop, f"{timestamp_str}_{self.ch_key}_{idx}.jpeg", self.ch_key, f"{result_dir}/crop")

            for rd_label, rd_conf in road_detections:
                msg = (f"{timestamp_str} | {self.ch_key} | {self.video_stream.location} | "
                       f"road: {rd_label} ({rd_conf:.2f})")
                self.post_processing_worker.add_crop_info_signal.emit(msg)

        return frame
            
    def _plot_bboxes(self, results: Results, im0: np.ndarray) -> Tuple[np.ndarray, float, List[int]]:
        """Plots bounding boxes on an image given detection results; returns annotated image, confidence value and class IDs."""
        class_ids = [] 

        annotated_frame = im0.copy()
        annotator = Annotator(annotated_frame, 10, results.names)

        boxes = results.boxes.xyxy.cpu()
        # Make the boxes top-left origin lower than y = 100
        boxes[:, 1] = boxes[:, 1].clamp(min=200)
        
        clss = results.boxes.cls.cpu().tolist()
        confs = results.boxes.conf.cpu().tolist()  # Assuming confidence scores are here
        names = results.names

        for box, cls, conf in zip(boxes, clss, confs):
            label = f"{names[int(cls)]}: {conf:.2f}"  # Format label to include confidence
            annotator.box_label(box, label=label, color=colors(int(cls), True))

        return annotated_frame, confs[0], class_ids