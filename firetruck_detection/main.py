import sys

from PyQt5 import QtWidgets

from model import Detection
from view import DetectionGUI
from controller import DetectionController
from utils import DEFAULT_CFG


def main():
    print('Starting firetruck detection system...')

    # Initialize the Qt application
    app = QtWidgets.QApplication(sys.argv)

    # Initialize the model, view, and controller (MVC pattern)
    model = Detection(DEFAULT_CFG)
    view = DetectionGUI(DEFAULT_CFG)
    controller = DetectionController(model, view)

    # Show the view (GUI)
    view.center()
    view.show()
    view.activateWindow()

    # Start the detection thread
    controller.start_detection()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 