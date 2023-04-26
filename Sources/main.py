"""
File Name: main.py
Author: Jinesh Mehta
File Description: This file contains the main function of the application.
"""
import sys
from PySide6.QtWidgets import (QApplication)
from main_window import MainWindow

if __name__ == "__main__":
    """Main function of the application
    """
    app = QApplication()
    w = MainWindow(app)
    w.startVideo()
    w.show()
    sys.exit(app.exec())
