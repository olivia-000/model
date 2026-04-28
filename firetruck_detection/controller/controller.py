from PyQt5 import QtWidgets
from time import perf_counter

from model import Detection
from view import DetectionGUI

from utils import NUM_VIDEO_STREAMS
import numpy as np


class DetectionController:
    """ 
    The controller class of the firetruck detection system. 
    It is responsible for handling the communication between the model and the view.
    """
    def __init__(
        self, 
        model: Detection=None, 
        view: DetectionGUI=None
    ) -> None:
        """
        Initialize the firetruck detection controller class.

        Args:
            model (FiretruckDetectionModel): The model of the firetruck detection system.
            view (DetectionMonitorView): The view of the firetruck detection system.
        """

        self.model = model
        self.view = view

        self.view.setupUi()

        # Setup view's signals' connections
        self.view.parameters_changed_signal.connect(self._update_parameters)
        self.view.window_closed_signal.connect(self._end_detection)
        self.view.set_roi_signal.connect(self.model.set_roi)
        self.view.clear_roi_signal.connect(self.model.clear_roi)
        
        # Setup initial states of the components in the view
        bridge_enabled = self.model.get_bridge_enabled()
        road_enabled = self.model.get_road_enabled()
        stage2_status = self.model.get_stage2_status()
        self.view.init_values(
            int(self.model.get_bridge_conf() * 100),
            0,
            stage2_status,
            conf_road=int(self.model.get_road_conf() * 100),
            conf_height=int(self.model.get_stage2_height_conf() * 100),
            conf_gap=int(self.model.get_stage2_gap_conf() * 100),
            bridge_enabled=bridge_enabled, road_enabled=road_enabled,
            vlm_enabled=self.model.get_vlm_enabled(),
            vlm_provider=self.model.get_vlm_provider(),
            vlm_model=self.model.get_vlm_model(),
            vlm_cooldown=self.model.get_vlm_cooldown(),
        )

        # Setup model's signals' connections
        for i in range(NUM_VIDEO_STREAMS):
            self.model.workers[i].video_stream\
                .new_processed_frame_signal.connect(
                    lambda frame, video_stream_idx=i: self._update_frame(frame, video_stream_idx)
                )
            self.model.workers[i].post_processing_worker\
                .show_post_processing_result_signal.connect(
                    lambda frame, title, video_stream_idx=i: self.view.show_post_processing_result(frame, title, video_stream_idx)
                )
            self.model.workers[i].post_processing_worker\
                .add_crop_info_signal.connect(self.view.append_crop_info)

        # VLM worker log signal → GUI log list
        self.model.vlm_worker.add_log_signal.connect(self.view.append_crop_info)

        
        # self.model.post_processing_worker.add_crop_info_signal.connect(self.view.append_crop_info)

        self.last_update_times = [0] * NUM_VIDEO_STREAMS
        self.update_counters = [0] * NUM_VIDEO_STREAMS
        self.mv_avg_fps = [0] * NUM_VIDEO_STREAMS  # Exponential moving average of the FPS
        self.alpha = 0.15  # Smoothing factor for exponential moving average


    def start_detection(self):
        """ Start the detection with the specified video directory paths """
        self.model.start()


    def _end_detection(self):
        """ Stop the running thread and end the detection """
        print('Terminating the detection system...')
        self.model.stop()
        print("====================================")
        for i in range(NUM_VIDEO_STREAMS):
            fps = self.mv_avg_fps[i]
            if fps >= 0:
                print(f'Average FPS for video stream {i+1} at {self.model.get_location(i)}: {fps:.2f}')
        print('Detection system terminated.')
    
    
    def _update_parameters(self):
        """ Update confidence & iou threshold of the model """

        stage2_status = self.view.stage2_status
        self.model.set_bridge_enabled(self.view.bridge_enabled)
        self.model.set_road_enabled(self.view.road_enabled)
        self.model.set_stage2_status(stage2_status)
        self.model.set_bridge_conf(self.view.slider_bridge_conf.value() / 100)
        self.model.set_road_conf(self.view.slider_road_conf.value() / 100)
        self.model.set_stage2_height_conf(self.view.slider_stage2_height_conf.value() / 100)
        self.model.set_stage2_gap_conf(self.view.slider_stage2_gap_conf.value() / 100)

        # VLM settings — only update if changed to avoid redundant yaml_save + prints
        vlm_enabled  = self.view.vlmToggle.isChecked()
        vlm_provider = self.view.vlmProviderCombo.currentText()
        vlm_model    = self.view.vlmModelCombo.currentText()
        if vlm_enabled != self.model.get_vlm_enabled():
            self.model.set_vlm_enabled(vlm_enabled)
        if vlm_provider != self.model.get_vlm_provider() or vlm_model != self.model.get_vlm_model():
            self.model.set_vlm_provider_model(vlm_provider, vlm_model)

        vlm_cooldown = self.view.vlmCooldownSpinBox.value()
        if vlm_cooldown != self.model.get_vlm_cooldown():
            self.model.set_vlm_cooldown(vlm_cooldown)


    def _update_frame(self, frame: np.ndarray, video_stream_idx: int) -> None:
        """ Update the frame and calculate FPS """
        self.update_counters[video_stream_idx] += 1
        now = perf_counter()
        fps = 0
        if self.last_update_times[video_stream_idx] == 0:
            self.last_update_times[video_stream_idx] = now
        else:
            elapsed_time = (now - self.last_update_times[video_stream_idx])
            if elapsed_time >= 1:
                self.last_update_times[video_stream_idx] = now
                fps = self.update_counters[video_stream_idx] / elapsed_time
                self.mv_avg_fps[video_stream_idx] = self.alpha * fps + (1 - self.alpha) * self.mv_avg_fps[video_stream_idx]
                self.update_counters[video_stream_idx] = 0
                self.view.update_fps(video_stream_idx, self.mv_avg_fps[video_stream_idx])
        self.view.update_frame(frame, video_stream_idx)