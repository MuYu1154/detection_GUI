from PySide6.QtWidgets import QApplication
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QTimer
from PySide6.QtGui import QImage, QPixmap
import cv2
import PySide6

from predictor import Predicter
import os

dirname = os.path.dirname(PySide6.__file__) 
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

class Camera():
    def __init__(self):
        qfile_states = QFile("./UI/Windows.ui")
        qfile_states.open(QFile.ReadOnly)
        qfile_states.close()
        self.ui = QUiLoader().load(qfile_states)   

        self.ui.pushButton_2.clicked.connect(self.camera_init)
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.read_img)

        self.predicter = Predicter()

    def camera_init(self):
        self.cap = cv2.VideoCapture(0)
        self.camera_timer.start(30)

    def read_img(self):
        ret, img = self.cap.read()
        if ret:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = self.predicter.inference(img)
            qimg = QImage(result, result.shape[1], result.shape[0], result.strides[0], QImage.Format_RGB888)
            self.ui.label.setPixmap(QPixmap.fromImage(qimg))

if __name__ == '__main__':
    app = QApplication([])
    camera = Camera()
    camera.ui.show()
    app.exec_()