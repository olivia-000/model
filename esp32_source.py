import cv2
import numpy as np
import requests
import threading
import time

class ESP32Stream:
    def __init__(self, url):
        self.is_file = not str(url).startswith("http")
        self.frame = None
        self.running = True

        if self.is_file:
            self.cap = cv2.VideoCapture(url)
            # 直接讀第一幀確認可以用
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            self.thread = threading.Thread(target=self._read_file)
        else:
            if ":81/stream" in url:
                base_ip = url.replace(":81/stream", "")
                self.url = base_ip + "/capture"
            else:
                self.url = url
            print(f"使用 capture 模式：{self.url}")
            self.thread = threading.Thread(target=self._read_frames)

        self.thread.daemon = True
        self.thread.start()

    def _read_file(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            time.sleep(0.001)

    def _read_frames(self):
        while self.running:
            try:
                r = requests.get(self.url, timeout=5)
                if r.status_code == 200:
                    img = cv2.imdecode(
                        np.frombuffer(r.content, dtype=np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    if img is not None:
                        self.frame = img
            except Exception as e:
                print(f"Stream error: {e}")
            time.sleep(0.05)

    def read(self):
        return self.frame is not None, self.frame

    def release(self):
        self.running = False
        if self.is_file and hasattr(self, 'cap'):
            self.cap.release()

#http://172.20.10.2:81/stream