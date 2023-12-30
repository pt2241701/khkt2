import sys
import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
from main_ui import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        self.uic.play_video.clicked.connect(self.start_capture_video)
        self.uic.stop_video.clicked.connect(self.stop_capture_video)
        self.thread = {}
    def stop_capture_video(self):
        self.thread[1].stop()
    def start_capture_video(self):
        self.thread[1] = capture_video(index=1)
        self.thread[1].start()
        self.thread[1].signal.connect(self.show_wedcam)
    def show_wedcam(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.uic.img_label.setPixmap(qt_img)
    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(800, 600, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
class capture_video(QThread):
    signal = pyqtSignal(np.ndarray)
    def __init__(self, index):
        self.index = index
        print("start threading", self.index)
        super(capture_video, self).__init__()
    def run(self):
        cap = cv2.VideoCapture(r"D:\a\output_video.mp4")
        while True:
            ret, cv_img = cap.read()
            if ret:
                self.signal.emit(cv_img)
    def stop(self):
        print("stop threading", self.index)
        self.terminate()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())