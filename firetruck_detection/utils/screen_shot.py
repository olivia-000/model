import os
import cv2
from datetime import datetime

BASE_IMAGE_DIR = 'crop_images'
""" The base directory for cropped images. """

def _ensure_base_dir_exists():
    """Ensure the base image directory exists."""
    if not os.path.exists(BASE_IMAGE_DIR):
        os.makedirs(BASE_IMAGE_DIR)

def _get_image_file_path(image_name, location, base_path):
    """Get the file path for the current day's image file."""
    now = datetime.now()
    year_dir = os.path.join(base_path, str(now.year))
    month_dir = os.path.join(year_dir, f'{now.month:02}')
    day_dir = os.path.join(month_dir, f'{now.day:02}')
    location_dir = os.path.join(day_dir, location)
    if not os.path.exists(location_dir):
        os.makedirs(location_dir)
    image_file = os.path.join(location_dir, image_name)
    return image_file

def save_image(image, image_name, location, base_path):
    """Save the image to the appropriate directory."""
    image_file_path = _get_image_file_path(image_name, location, base_path)
    cv2.imwrite(image_file_path, image, [cv2.IMWRITE_JPEG_QUALITY, 30])

def open_crop_image(image_name, location, dir):
    """Open the image with the specified name."""
    # Turn this into a full path: 2024-09-04_09-42-39_路口三
    date = image_name.split('_')[0]
    year, month, day = date.split('-')
    image_file_path = os.path.join(dir, year, month, day, location, image_name)
    try:
        image = cv2.imread(image_file_path)
    except Exception as e:
        print(f"Error opening image: {e}")
        image = None
    return image

# Ensure the base image directory exists when the module is imported
_ensure_base_dir_exists()
