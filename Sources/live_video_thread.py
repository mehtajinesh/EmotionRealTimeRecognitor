"""
File: live_video_thread.py
Author: Jinesh Mehta
File Description: This file contains the LiveVideoThread class which is used to
                    run the video capture in a separate thread.
"""
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage
import cv2
import numpy as np
from constants import HAAR_CASCADE_PATH, RESNET_MODEL_PATH, RESNET_FER_MEAN, RESNET_FER_STD, \
    RESNET_FER_IMG_WIDTH, RESNET_FER_IMG_HEIGHT, EMOTIONS
from keras.models import load_model


class LiveVideoThread(QThread):
    """LiveVideoThread class is used to run the video capture in a separate thread.

    Args:
        QThread (_type_): QThread class

    """

    updateFrame = Signal(QImage)
    updateEmotion = Signal(list)

    def __init__(self, parent=None):
        """Constructor for LiveVideoThread class

        Args:
            parent (_type_, optional): Parent Class. Defaults to None.
        """
        QThread.__init__(self, parent)
        self.captureDevice = True
        self.faceCascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        self.model = load_model(RESNET_MODEL_PATH)

    def preprocess_input(self, image):
        """Preprocesses the input image for the model

        Args:
            image (_type_): image to be preprocessed

        Returns:
            _type_: preprocessed image
        """
        # Resizing images for the trained model
        image = cv2.resize(
            image, (RESNET_FER_IMG_WIDTH, RESNET_FER_IMG_HEIGHT))
        ret = np.empty((RESNET_FER_IMG_HEIGHT, RESNET_FER_IMG_WIDTH, 3))
        ret[:, :, 0] = image
        ret[:, :, 1] = image
        ret[:, :, 2] = image
        x = np.expand_dims(ret, axis=0)
        x -= RESNET_FER_MEAN
        x /= RESNET_FER_STD
        return x

    def predict(self, emotion):
        """Predicts the emotion from the image

        Args:
            emotion (_type_): image to be predicted

        Returns:
            _type_: prediction
        """
        prediction = self.model.predict(emotion)
        return prediction

    def run(self):
        """Runs the video capture in a separate thread
        """
        print("Starting video thread")
        self.captureDevice = cv2.VideoCapture(0)
        while True:
            ret, frame = self.captureDevice.read()
            if not ret:
                print("Error reading frame")
                continue
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30))
            if len(faces) != 0:
                x, y, w, h = faces[0]
                ROI_gray = gray_frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

                emotion = self.preprocess_input(ROI_gray)
                prediction = self.predict(emotion)[0]
                # sort probabilities in descending order
                sorted_predictions = np.argsort(prediction)[::-1]
                # get top 3 predictions
                top_3_predictions = sorted_predictions[:3]
                # get emotions mapped to predictions
                top_3_emotions = [EMOTIONS[i] for i in top_3_predictions]
                # create a list of tuples with emotion and probability
                emotion_probabilities = [(emotion, prediction[i])
                                         for i, emotion in enumerate(top_3_emotions)]
                # sort list in based on probability
                emotion_probabilities.sort(key=lambda x: x[1], reverse=True)
                # Emit signal
                self.updateEmotion.emit(emotion_probabilities)
            # Reading the image in RGB to display it
            color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Creating and scaling QImage
            h, w, ch = color_frame.shape
            img = QImage(color_frame.data, w, h, ch * w, QImage.Format_RGB888)
            scaled_img = img.scaled(640, 480, Qt.KeepAspectRatio)
            # Emit signal
            self.updateFrame.emit(scaled_img)
