import sys
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage
import cv2


class LiveVideoThread(QThread):

    updateFrame = Signal(QImage)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.captureDevice = True

    def run(self):
        print("Starting video thread")
        self.captureDevice = cv2.VideoCapture(0)
        while True:
            ret, frame = self.captureDevice.read()
            if not ret:
                print("Error reading frame")
                continue
            # Reading the image in RGB to display it
            color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Creating and scaling QImage
            h, w, ch = color_frame.shape
            img = QImage(color_frame.data, w, h, ch * w, QImage.Format_RGB888)
            scaled_img = img.scaled(640, 480, Qt.KeepAspectRatio)
            # Emit signal
            self.updateFrame.emit(scaled_img)