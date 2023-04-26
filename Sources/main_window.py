from MainWindow_ui import Ui_MainWindow

from PySide6.QtWidgets import QMainWindow
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap

from live_video_thread import LiveVideoThread

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, app):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # Thread in charge of updating the image
        self.video_thread = LiveVideoThread(self)
        self.video_thread.updateFrame.connect(self.setImage)
        self.video_thread.start()

    @Slot(QImage)
    def setImage(self, image):
        self.ui.labelLiveVideo.setPixmap(QPixmap.fromImage(image))