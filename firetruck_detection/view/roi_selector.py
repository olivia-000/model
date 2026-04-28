from PyQt5 import QtWidgets, QtCore, QtGui
import cv2
from utils import VIEW_DOWNSCALE_RATIO

class ROIDialog(QtWidgets.QDialog):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.image = image
        self.roi = None
        self.setWindowTitle("設定ROI")

        # Scale down the image to fit within the screen
        screen = QtWidgets.QApplication.primaryScreen()
        screen_size = screen.size()
        max_width = screen_size.width() * 0.8
        max_height = screen_size.height() * 0.8

        height, width, _ = image.shape
        scale_factor = min(max_width / width, max_height / height, 1)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        self.scaled_image = cv2.resize(image, (new_width, new_height))

        self.setGeometry(100, 100, new_width, new_height)
        self.setFixedSize(new_width, new_height)
        self.label = QtWidgets.QLabel(self)
        self.label.setPixmap(QtGui.QPixmap.fromImage(self._convert_to_qimage(self.scaled_image)))
        self.label.setGeometry(0, 0, new_width, new_height)
        self.label.mousePressEvent = self._mouse_press_event
        self.label.mouseReleaseEvent = self._mouse_release_event
        self.label.mouseMoveEvent = self._mouse_move_event
        self.start_pos = None
        self.end_pos = None

    def _convert_to_qimage(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        return QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()

    def _mouse_press_event(self, event):
        self.start_pos = event.pos()
        self.end_pos = None
        self.update()

    def _mouse_release_event(self, event):
        self.end_pos = event.pos()
        x1, y1 = self.start_pos.x() * VIEW_DOWNSCALE_RATIO, self.start_pos.y() * VIEW_DOWNSCALE_RATIO
        x2, y2 = self.end_pos.x() * VIEW_DOWNSCALE_RATIO, self.end_pos.y() * VIEW_DOWNSCALE_RATIO
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)
        self.roi = (int(x / self.scaled_image.shape[1] * self.image.shape[1]),
                    int(y / self.scaled_image.shape[0] * self.image.shape[0]),
                    int(w / self.scaled_image.shape[1] * self.image.shape[1]),
                    int(h / self.scaled_image.shape[0] * self.image.shape[0]))
        self.accept()

    def _mouse_move_event(self, event):
        if self.start_pos:
            self.end_pos = event.pos()
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.start_pos and self.end_pos:
            self.label.setPixmap(QtGui.QPixmap.fromImage(self._convert_to_qimage(self.scaled_image)))
            painter = QtGui.QPainter(self.label.pixmap())
            painter.setPen(QtGui.QPen(QtCore.Qt.red, 2))
            painter.drawRect(QtCore.QRect(self.start_pos, self.end_pos))
            painter.end()
            self.label.update()

    def get_roi(self):
        return self.roi