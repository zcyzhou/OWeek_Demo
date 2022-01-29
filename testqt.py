import cv2
import sys

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

image_paths = ['Inputs/feynman.jpeg', 'Inputs/Monalisa.png', 'Inputs/orlando_bloom.jpg']

image_source = image_paths[0]

class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.VBL = QVBoxLayout(self)
        self.HBL = QHBoxLayout()
        self.VBL1 = QVBoxLayout()
        self.VBL2 = QVBoxLayout()
        self.VBL3 = QVBoxLayout()
        self.VBL.addLayout(self.HBL)
        self.HBL.addLayout(self.VBL1)
        self.HBL.addLayout(self.VBL2)
        self.HBL.addLayout(self.VBL3)

        # widget for video
        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)

        # widget for image preview
        self.ImagePreview1 = QLabel()
        self.ImagePreview2 = QLabel()
        self.ImagePreview3 = QLabel()
        self.ImagePreview1.setPixmap(QPixmap(image_paths[0]))
        self.ImagePreview2.setPixmap(QPixmap(image_paths[1]))
        self.ImagePreview3.setPixmap(QPixmap(image_paths[2]))
        self.VBL1.addWidget(self.ImagePreview1)
        self.VBL2.addWidget(self.ImagePreview2)
        self.VBL3.addWidget(self.ImagePreview3)

        # Add button for each image preview
        self.BTN1 = QPushButton(image_paths[0].split('/')[-1])
        self.BTN1.clicked.connect(self.ChooseFirst)
        self.BTN2 = QPushButton(image_paths[1].split('/')[-1])
        self.BTN2.clicked.connect(self.ChooseSecond)
        self.BTN3 = QPushButton(image_paths[2].split('/')[-1])
        self.BTN3.clicked.connect(self.ChooseThird)
        self.VBL1.addWidget(self.BTN1)
        self.VBL2.addWidget(self.BTN2)
        self.VBL3.addWidget(self.BTN3)

        # Add Stop button
        self.CancelBTN = QPushButton("Stop")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)

        self.worker1 = Worker1()
        self.worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.worker1.start()

        self.setLayout(self.VBL)
    
    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))
    
    def CancelFeed(self):
        self.worker1.stop()
    
    def ChooseFirst(self):
        # self.worker1.stop()
        self.worker1.ThreadActive = False
        global image_source
        image_source = image_paths[0]
        # self.worker1.start()
        self.worker1.ThreadActive = True
    
    def ChooseSecond(self):
        self.worker1.ThreadActive = False
        global image_source
        image_source = image_paths[1]
        self.worker1.ThreadActive = True
    
    def ChooseThird(self):
        self.worker1.ThreadActive = False
        global image_source
        image_source = image_paths[2]
        self.worker1.ThreadActive = True
    
class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    
    def run(self):
        print(" I wanna run ")
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)
        while 1:
            if not self.ThreadActive:
                continue
            elif self.ThreadActive:
                print(image_source)
                ret, frame = Capture.read()
                if ret:
                    Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    FlippedImage = cv2.flip(Image, 1)
                    ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                    Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                    self.ImageUpdate.emit(Pic)
            # Capture.release()
    
    def stop(self):
        print("--- I wanna stop ---")
        self.ThreadActive = False
        self.quit()


if __name__ == "__main__":
    print(" I create app ")
    App = QApplication(sys.argv)
    print("I load window")
    Root = MainWindow()
    Root.show()
    print(" I run exec() ")
    sys.exit(App.exec())  