import os
from typing import Tuple
from datetime import datetime

import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from PIL import Image, ImageDraw, ImageFont

from utils import (
    SPEED,
    LIVE_VIDEO_STREAM,
    VIDEO_DEMO
)


class VideoStream(QObject):
    """ A class responsible for hyperparameters and manipulation of video/streaming source for each video stream """

    new_processed_frame_signal = pyqtSignal(np.ndarray)
    """
        Signal to update the frame of a single video stream in the GUI

        Args:
            frame (np.ndarray): The frame to be updated to the GUI.
    """
    
    def __init__(
            self,
            video_stream_idx: int,
            cfg: dict,
        ) -> None:
        """
        Initialize the video stream class.

        Args:
            video stream (int): The video stream number of the video stream. (start from 0)
            
        """
        super().__init__()
        self.video_stream_idx = video_stream_idx
        self._cfg_setup(cfg)
        self.video_idx = -1
        self._retry = True

        # Loading screen
        frame = Image.new('RGB', (640, 480), color = (0, 0, 0))
        d = ImageDraw.Draw(frame)
        font = ImageFont.truetype(self.font_path, 50)
        d.text((10,10), "Loading...", fill=(255,255,255), font=font)
        self.loading_screen = np.array(frame)

        # Finish screen
        frame = Image.new('RGB', (640, 480), color = (0, 0, 0))
        d = ImageDraw.Draw(frame)
        font = ImageFont.truetype(self.font_path, 50)
        d.text((10,10), "Finish reading all videos", fill=(255,255,255), font=font)
        self.finish_screen = np.array(frame)
        
    
    def _cfg_setup(self, cfg_dict: dict) -> None:
        """ Load the configuration for the video stream """
        self.cfg = cfg_dict

        # Info for loading the video source for video demo/testing
        video_dir_path = self.cfg['test_video_dir']
        video_path = os.path.join(video_dir_path, f"ch{self.video_stream_idx+1}") # There should be a folder for each video stream, named as ch1, ch2, ch3, ch4, ...

        # Dictionary containing information about this video stream: location(str), source(str).
        self.location = self.cfg[f'ch{self.video_stream_idx+1}_location']

        self.source_type = self.cfg['source'] # 0: video demo, 1: live video stream
        if self.source_type == LIVE_VIDEO_STREAM:
            self.source = self.cfg[f'ch{self.video_stream_idx+1}_cctv_IP']
            self.num_of_video = 0
        else:
            self.source = video_path
            self.num_of_video = len(os.listdir(video_path))
        
        self.font_path = f"font/{self.cfg['font']}.ttf"


    def capture_next(self) -> None:
        """ Capture the streaming source or the next video in the video path """
        source_identifier = self.source
        if self.source_type == VIDEO_DEMO:
            if self.video_idx >= self.num_of_video - 1:
                # print(f"Finish reading {self.video_stream_idx+1} videos in {self.source}")
                self.stop()
                return
            self.video_idx = self.video_idx + 1
            source_identifier = os.path.join(self.source, os.listdir(self.source)[self.video_idx]) 
        print(f"video stream {self.video_stream_idx+1} try to capture video source of {self.location}")


        self.capture = cv2.VideoCapture(source_identifier)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0

        print(f"FPS of video stream {self.video_stream_idx+1}: {self.fps}")
    

    def stop(self) -> None:
        """ Stop the retry mechanism """
        self._retry = False 
        self.release()


    def release(self) -> None:
        """ Release the video capture """
        try:
            if self.capture.isOpened():
                print(f"Releasing video capture of {self.location}")  
                self.capture.release()
        except Exception as e:
            print(f"Error releasing video capture: {e}")


    def _read(self) -> Tuple[bool, np.ndarray]:
        """ Get the next frame from the video/streaming source """
        for i in range(SPEED):
            success, frame = self.capture.read()
        return success, frame


    def fetch_next_frame(self) -> Tuple[bool, np.ndarray]:
        """ Fetch the next frame from the video/streaming source with retry mechanism """

        # Flag to indicate the connection state of the video stream
        connection_state = True 
        
        success, frame = self._read()

        while self._retry and not success:
            # Handle the case when the frame fetching fails due to connection issues or other unknown reasons
            connection_state = False
            print(f"video stream {self.video_stream_idx+1} failed to fetch frame, retrying...")
            self.release()
            self.update_frame()
            self.capture_next()

            success, frame = self._read()
                
        if not connection_state and success:
            print(f"video stream {self.video_stream_idx+1} reconnected!")
        elif not success:
            return False, self.finish_screen
        
        # Let inference thread process the latest frame
        return success, frame
    

    def update_frame(self, frame:np.ndarray=None) -> None:
        """ Update the frame of the video stream """
        if frame is None:
            frame = self.loading_screen
        try:
            self.new_processed_frame_signal.emit(frame)
        except:
            print(f"Error updating frame of video stream {self.video_stream_idx}")
            print(frame)
    
