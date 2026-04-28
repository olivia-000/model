import os
from datetime import datetime

BASE_LOG_DIR = 'logs'
""" The base directory for log files. """

def _ensure_log_dir_exists():
    """Ensure the base log directory exists."""
    if not os.path.exists(BASE_LOG_DIR):
        os.makedirs(BASE_LOG_DIR)

def _get_log_file_path(now: datetime = datetime.now()):
    """Get the file path for the current day's log file."""
    year_dir = os.path.join(BASE_LOG_DIR, str(now.year))
    month_dir = os.path.join(year_dir, f'{now.month:02}')
    if not os.path.exists(month_dir):
        os.makedirs(month_dir)
    log_file = os.path.join(month_dir, f'{now.day:02}.txt')
    return log_file

def log_detection(message, time: datetime = datetime.now()):
    """Log a message to the appropriate log file."""
    log_file_path = _get_log_file_path(time)

    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f'{timestamp} - {message}\n')

# Ensure the log directory exists when the module is imported
_ensure_log_dir_exists()