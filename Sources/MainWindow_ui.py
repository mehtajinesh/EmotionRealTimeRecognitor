# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainWindow.ui'
##
## Created by: Qt User Interface Compiler version 6.4.3
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QProgressBar, QSizePolicy,
    QSpacerItem, QSplitter, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.setWindowModality(Qt.NonModal)
        MainWindow.resize(940, 568)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout_2 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.splitterRealTimeLabelOthers = QSplitter(self.centralwidget)
        self.splitterRealTimeLabelOthers.setObjectName(u"splitterRealTimeLabelOthers")
        self.splitterRealTimeLabelOthers.setOrientation(Qt.Vertical)
        self.horizontalLayoutWidget = QWidget(self.splitterRealTimeLabelOthers)
        self.horizontalLayoutWidget.setObjectName(u"horizontalLayoutWidget")
        self.horizontalLayoutRealTimeLabel = QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayoutRealTimeLabel.setObjectName(u"horizontalLayoutRealTimeLabel")
        self.horizontalLayoutRealTimeLabel.setContentsMargins(0, 0, 0, 0)
        self.horizontalSpacerLeftRealTime = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayoutRealTimeLabel.addItem(self.horizontalSpacerLeftRealTime)

        self.labelRealTimeEmotionRecognition = QLabel(self.horizontalLayoutWidget)
        self.labelRealTimeEmotionRecognition.setObjectName(u"labelRealTimeEmotionRecognition")
        font = QFont()
        font.setFamilies([u"Arial"])
        font.setPointSize(18)
        font.setBold(True)
        self.labelRealTimeEmotionRecognition.setFont(font)
        self.labelRealTimeEmotionRecognition.setTextFormat(Qt.PlainText)
        self.labelRealTimeEmotionRecognition.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)

        self.horizontalLayoutRealTimeLabel.addWidget(self.labelRealTimeEmotionRecognition)

        self.horizontalSpacerRightRealTime = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayoutRealTimeLabel.addItem(self.horizontalSpacerRightRealTime)

        self.splitterRealTimeLabelOthers.addWidget(self.horizontalLayoutWidget)
        self.splitterLiveVideoOthers = QSplitter(self.splitterRealTimeLabelOthers)
        self.splitterLiveVideoOthers.setObjectName(u"splitterLiveVideoOthers")
        self.splitterLiveVideoOthers.setOrientation(Qt.Horizontal)
        self.groupBoxLiveVideo = QGroupBox(self.splitterLiveVideoOthers)
        self.groupBoxLiveVideo.setObjectName(u"groupBoxLiveVideo")
        self.verticalLayout_3 = QVBoxLayout(self.groupBoxLiveVideo)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.labelLiveVideo = QLabel(self.groupBoxLiveVideo)
        self.labelLiveVideo.setObjectName(u"labelLiveVideo")
        self.labelLiveVideo.setMinimumSize(QSize(640, 480))

        self.verticalLayout_3.addWidget(self.labelLiveVideo)

        self.splitterLiveVideoOthers.addWidget(self.groupBoxLiveVideo)
        self.splitterModelSelectionOutput = QSplitter(self.splitterLiveVideoOthers)
        self.splitterModelSelectionOutput.setObjectName(u"splitterModelSelectionOutput")
        self.splitterModelSelectionOutput.setOrientation(Qt.Vertical)
        self.groupBoxModelSelection = QGroupBox(self.splitterModelSelectionOutput)
        self.groupBoxModelSelection.setObjectName(u"groupBoxModelSelection")
        self.verticalLayout_6 = QVBoxLayout(self.groupBoxModelSelection)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.labelModelSelection = QLabel(self.groupBoxModelSelection)
        self.labelModelSelection.setObjectName(u"labelModelSelection")

        self.horizontalLayout_2.addWidget(self.labelModelSelection)

        self.comboBoxModelSelection = QComboBox(self.groupBoxModelSelection)
        self.comboBoxModelSelection.addItem("")
        self.comboBoxModelSelection.addItem("")
        self.comboBoxModelSelection.addItem("")
        self.comboBoxModelSelection.setObjectName(u"comboBoxModelSelection")

        self.horizontalLayout_2.addWidget(self.comboBoxModelSelection)

        self.horizontalSpacerEndModelSelection = QSpacerItem(47, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacerEndModelSelection)


        self.verticalLayout_6.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer)

        self.labelUpdatingModel = QLabel(self.groupBoxModelSelection)
        self.labelUpdatingModel.setObjectName(u"labelUpdatingModel")
        font1 = QFont()
        font1.setBold(True)
        self.labelUpdatingModel.setFont(font1)
        self.labelUpdatingModel.setStyleSheet(u"color:red")

        self.horizontalLayout_3.addWidget(self.labelUpdatingModel)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_2)


        self.verticalLayout_6.addLayout(self.horizontalLayout_3)

        self.splitterModelSelectionOutput.addWidget(self.groupBoxModelSelection)
        self.groupBoxOutput = QGroupBox(self.splitterModelSelectionOutput)
        self.groupBoxOutput.setObjectName(u"groupBoxOutput")
        self.verticalLayout = QVBoxLayout(self.groupBoxOutput)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayoutDetectedEmoji = QHBoxLayout()
        self.horizontalLayoutDetectedEmoji.setObjectName(u"horizontalLayoutDetectedEmoji")
        self.labelDetectedEmoji = QLabel(self.groupBoxOutput)
        self.labelDetectedEmoji.setObjectName(u"labelDetectedEmoji")
        self.labelDetectedEmoji.setFont(font1)

        self.horizontalLayoutDetectedEmoji.addWidget(self.labelDetectedEmoji)

        self.labelDetectedEmojiPhoto = QLabel(self.groupBoxOutput)
        self.labelDetectedEmojiPhoto.setObjectName(u"labelDetectedEmojiPhoto")

        self.horizontalLayoutDetectedEmoji.addWidget(self.labelDetectedEmojiPhoto)

        self.horizontalSpacerEndEmoji = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayoutDetectedEmoji.addItem(self.horizontalSpacerEndEmoji)


        self.verticalLayout.addLayout(self.horizontalLayoutDetectedEmoji)

        self.groupBoxDetailedStats = QGroupBox(self.groupBoxOutput)
        self.groupBoxDetailedStats.setObjectName(u"groupBoxDetailedStats")
        self.horizontalLayout = QHBoxLayout(self.groupBoxDetailedStats)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.labelTopEmotion_1 = QLabel(self.groupBoxDetailedStats)
        self.labelTopEmotion_1.setObjectName(u"labelTopEmotion_1")
        self.labelTopEmotion_1.setMinimumSize(QSize(100, 0))

        self.verticalLayout_5.addWidget(self.labelTopEmotion_1)

        self.labelTopEmotion_2 = QLabel(self.groupBoxDetailedStats)
        self.labelTopEmotion_2.setObjectName(u"labelTopEmotion_2")
        self.labelTopEmotion_2.setMinimumSize(QSize(100, 0))

        self.verticalLayout_5.addWidget(self.labelTopEmotion_2)

        self.labelTopEmotion_3 = QLabel(self.groupBoxDetailedStats)
        self.labelTopEmotion_3.setObjectName(u"labelTopEmotion_3")
        self.labelTopEmotion_3.setMinimumSize(QSize(100, 0))

        self.verticalLayout_5.addWidget(self.labelTopEmotion_3)


        self.horizontalLayout.addLayout(self.verticalLayout_5)

        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.progressBarTopEmotion1 = QProgressBar(self.groupBoxDetailedStats)
        self.progressBarTopEmotion1.setObjectName(u"progressBarTopEmotion1")
        self.progressBarTopEmotion1.setValue(24)

        self.verticalLayout_4.addWidget(self.progressBarTopEmotion1)

        self.progressBarTopEmotion2 = QProgressBar(self.groupBoxDetailedStats)
        self.progressBarTopEmotion2.setObjectName(u"progressBarTopEmotion2")
        self.progressBarTopEmotion2.setValue(24)

        self.verticalLayout_4.addWidget(self.progressBarTopEmotion2)

        self.progressBarTopEmotion3 = QProgressBar(self.groupBoxDetailedStats)
        self.progressBarTopEmotion3.setObjectName(u"progressBarTopEmotion3")
        self.progressBarTopEmotion3.setValue(24)

        self.verticalLayout_4.addWidget(self.progressBarTopEmotion3)


        self.horizontalLayout.addLayout(self.verticalLayout_4)


        self.verticalLayout.addWidget(self.groupBoxDetailedStats)

        self.splitterModelSelectionOutput.addWidget(self.groupBoxOutput)
        self.splitterLiveVideoOthers.addWidget(self.splitterModelSelectionOutput)
        self.splitterRealTimeLabelOthers.addWidget(self.splitterLiveVideoOthers)

        self.verticalLayout_2.addWidget(self.splitterRealTimeLabelOthers)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Real Time Emotion Recognition", None))
        self.labelRealTimeEmotionRecognition.setText(QCoreApplication.translate("MainWindow", u"Real Time Emotion Recognition", None))
        self.groupBoxLiveVideo.setTitle(QCoreApplication.translate("MainWindow", u"Live Video", None))
        self.labelLiveVideo.setText(QCoreApplication.translate("MainWindow", u"Loading Video", None))
        self.groupBoxModelSelection.setTitle(QCoreApplication.translate("MainWindow", u"Model Selection", None))
        self.labelModelSelection.setText(QCoreApplication.translate("MainWindow", u"Model Name:", None))
        self.comboBoxModelSelection.setItemText(0, QCoreApplication.translate("MainWindow", u"ResNet50", None))
        self.comboBoxModelSelection.setItemText(1, QCoreApplication.translate("MainWindow", u"Inception-v2", None))
        self.comboBoxModelSelection.setItemText(2, QCoreApplication.translate("MainWindow", u"Inception-v3", None))

        self.labelUpdatingModel.setText(QCoreApplication.translate("MainWindow", u"Updating Model. Please wait ...", None))
        self.groupBoxOutput.setTitle(QCoreApplication.translate("MainWindow", u"Output", None))
        self.labelDetectedEmoji.setText(QCoreApplication.translate("MainWindow", u"Detected Emotion Emoji:", None))
        self.labelDetectedEmojiPhoto.setText(QCoreApplication.translate("MainWindow", u"Emoji Pic", None))
        self.groupBoxDetailedStats.setTitle(QCoreApplication.translate("MainWindow", u"Detailed Stats", None))
        self.labelTopEmotion_1.setText(QCoreApplication.translate("MainWindow", u"Top Emotion-1:", None))
        self.labelTopEmotion_2.setText(QCoreApplication.translate("MainWindow", u"Top Emotion-2: ", None))
        self.labelTopEmotion_3.setText(QCoreApplication.translate("MainWindow", u"Top Emotion-3:", None))
    # retranslateUi

