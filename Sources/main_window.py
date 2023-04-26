"""
File Name: main_window.py
Author: Jinesh Mehta
File Description: This file contains the MainWindow class which is used to
                    create the main window of the application.
"""
import os
from MainWindow_ui import Ui_MainWindow
from PySide6.QtWidgets import QMainWindow
from PySide6.QtCore import Slot
from PySide6.QtGui import QImage, QPixmap
from live_video_thread import LiveVideoThread
from constants import EMOJI_DIRECTORY


class MainWindow(QMainWindow, Ui_MainWindow):
    """MainWindow class is used to create the main window of the application.

    Args:
        QMainWindow (_type_): QMainWindow class
        Ui_MainWindow (_type_): Ui_MainWindow class
    """

    def __init__(self):
        """Constructor for MainWindow class
        """
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # Thread in charge of updating the image
        self.video_thread = LiveVideoThread(self)
        self.video_thread.updateFrame.connect(self.setImage)
        self.video_thread.updateEmotion.connect(self.updateEmotionProbability)
        # Default Model is ResNet
        self.ui.comboBoxModelSelection.setCurrentIndex(0)

    def startVideo(self):
        """Starts the video capture
        """
        self.video_thread.start()

    @Slot(QImage)
    def setImage(self, image):
        """Sets the image in the label

        Args:
            image (_type_): image to be set
        """
        self.ui.labelLiveVideo.setPixmap(QPixmap.fromImage(image))

    @Slot(list)
    def updateEmotionProbability(self, emotion_probability):
        """Updates the emotion probability in the UI

        Args:
            emotion_probability (_type_): list of emotions and their probabilities
        """
        self.ui.labelTopEmotion_1.setText(emotion_probability[0][0])
        self.ui.labelTopEmotion_2.setText(emotion_probability[1][0])
        self.ui.labelTopEmotion_3.setText(emotion_probability[2][0])
        self.ui.progressBarTopEmotion1.setValue(
            int(emotion_probability[0][1]*100))
        self.ui.progressBarTopEmotion2.setValue(
            int(emotion_probability[1][1]*100))
        self.ui.progressBarTopEmotion3.setValue(
            int(emotion_probability[2][1]*100))
        self.ui.labelDetectedEmojiPhoto.setPixmap(QPixmap.fromImage(QImage(
            os.path.join(EMOJI_DIRECTORY, emotion_probability[0][0]+'.svg'))))
