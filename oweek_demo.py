import os
from threading import Thread
import cv2
import sys
import time
import torch
import imageio
import qimage2ndarray
import face_recognition

import numpy as np

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from animate import normalize_kp
from demo import load_checkpoints
from skimage import img_as_ubyte
from skimage.transform import resize

image_paths = ['Inputs/feynman.jpeg', 'Inputs/Monalisa.png', 'Inputs/orlando_bloom.jpg']

image_source = image_paths[0]
checkpoint_path = "./checkpoints/vox-cpk.pth.tar"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

ThreadActive = True

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
        self.ImagePreview1.setPixmap(QPixmap(image_paths[0]).scaled(256, 256))
        self.ImagePreview2.setPixmap(QPixmap(image_paths[1]).scaled(256, 256))
        self.ImagePreview3.setPixmap(QPixmap(image_paths[2]).scaled(256, 256))
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
        # self.worker1.ThreadActive = False
        global ThreadActive
        ThreadActive = False
        global image_source
        image_source = image_paths[0]
        # self.worker1.start()
        # self.worker1.ThreadActive = True
    
    def ChooseSecond(self):
        # self.worker1.ThreadActive = False
        global ThreadActive
        ThreadActive = False
        global image_source
        image_source = image_paths[1]
        # self.worker1.ThreadActive = True
        # ThreadActive = True
    
    def ChooseThird(self):
        global ThreadActive
        ThreadActive = False
        global image_source
        image_source = image_paths[2]
        # self.worker1.ThreadActive = True
    
class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    
    def run(self):
        global ThreadActive
        ThreadActive = True
        Capture = cv2.VideoCapture(0)
        needReload = True
        # Load the model
        generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml', checkpoint_path=checkpoint_path)
        source_image = None
        source = None
        cv2_source = None
        kp_source = None
        predictions = []
        count = 0
        while 1:
            if not ThreadActive:
                needReload = True
                ThreadActive = True
                continue
            elif ThreadActive:
                with torch.no_grad():
                    if needReload:
                        needReload = False
                        count = 0
                        source_image = imageio.imread(image_source)
                        source_image = resize(source_image, (256,256))[..., :3]
                        cv2_source = cv2.cvtColor(source_image.astype('float32'),cv2.COLOR_BGR2RGB)
                        cv2_source = cv2.cvtColor(cv2_source, cv2.COLOR_RGB2BGR)
                        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                        source.to(device)
                        kp_source = kp_detector(source)
                        continue
                    ret, frame = Capture.read()
                    frame = cv2.flip(frame, 1)
                    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    if ret:
                        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                        rgb_small_frame = small_frame[:, :, ::-1]
                        face_locations = face_recognition.face_locations(rgb_small_frame)
                        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                        if len(face_locations) == 0:
                            continue

                        top, right, bottom, left = face_locations[0]
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4

                        vertical_offset = (1000 - bottom + top) / 2
                        horizental_offset = (1000 - left + right) / 2

                        top -= vertical_offset
                        bottom += vertical_offset
                        right -= horizental_offset
                        left += horizental_offset

                        top = int(top)
                        bottom = int(bottom)
                        right = int(right)
                        left = int(left)

                        frame1 = resize(frame[top:bottom, right:left],(256,256))[..., :3]

                        if count == 0:
                            source_image1 = frame1
                            source1 = torch.tensor(source_image1[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                            kp_driving_initial = kp_detector(source1)

                        frame_test = torch.tensor(frame1[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                        driving_frame = frame_test
                        driving_frame.to(device)

                        kp_driving = kp_detector(driving_frame)
                        kp_norm = normalize_kp(kp_source=kp_source,
                                    kp_driving=kp_driving,
                                    kp_driving_initial=kp_driving_initial, 
                                    use_relative_movement=True,
                                    use_relative_jacobian=True, 
                                    adapt_movement_scale=True)
                        out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
                        im = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
                        # im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
                        joinedFrame = np.concatenate((cv2_source,im,frame1),axis=1)
                        ConvertToQtFormat = qimage2ndarray.array2qimage(joinedFrame, normalize = True)
                        self.ImageUpdate.emit(ConvertToQtFormat)
                        count += 1
                    else:
                        break
    
    def stop(self):
        self.ThreadActive = False
        sys.exit()
        self.quit()


if __name__ == "__main__":
    if not os.path.exists('output'):
        os.mkdir('output')
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())  