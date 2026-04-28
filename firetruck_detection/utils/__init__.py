from .screen_shot import save_image, open_crop_image
from .detection_logger import log_detection
from .yaml_operation import yaml_load, yaml_save

__all__ = ['save_image', 'log_detection', 'yaml_load', 'yaml_save', 'open_crop_image']

########################################### SYSTEM PARAMETERS ###########################################

DEFAULT_CFG = 'cfg/default_settings.yaml'
""" The default configuration file path """

COSTUMIZED_CFG = 'cfg/my_last_settings.yaml'
""" The costumized configuration file path which stores the last settings """

MULTI_MODEL = False
""" 
    Whether to use multiple models for different video streams 
    Don't use False, it's not optimal.
"""

NUM_VIDEO_STREAMS = 5
""" The number of video streams """

CAMERA_DIRECTIONS = ["right", "front", "left", "back-down", "back"]
""" Direction flag for each channel: ch1=right, ch2=front, ch3=left, ch4=back-down, ch5=back """

SPEED = 1
""" Replay speed of the video demo (by skipping frames) """

VIEW_DOWNSCALE_RATIO = 2
""" The ratio of downscaling the frame for the GUI to show """

CROPPING_THRESHOLD = 1/3
"""
After the center of the bbox is below this horizontal line indicated by the ratio of the frame height, 
the screenshot will be taken.
"""

MAX_CONFIDENCE = 90
""" The maximum value of the confidence threshold """

MIN_CONFIDENCE = 20
""" The minimum value of the confidence threshold """


# Source type (used as enum, don't change the values)
VIDEO_DEMO = 0
LIVE_VIDEO_STREAM = 1