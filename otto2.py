import sys
import cv2
import torch
import urllib3
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import os
from datetime import datetime
from main_window import Ui_MainWindow  # The generated file from .ui
import pathlib
from pathlib import Path

pathlib.PosixPath = pathlib.WindowsPath

# Disable SSL verification warnings (not recommended for production)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Custom HTTPS handler to bypass SSL verification
import ssl
from urllib.request import build_opener, HTTPSHandler

ssl._create_default_https_context = ssl._create_unverified_context
opener = build_opener(HTTPSHandler(context=ssl._create_unverified_context()))

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self, model, video_path):
        super().__init__()
        self.model = model
        self.video_path = video_path
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 2)  # Number of frames to skip to get a 2-second interval
        frame_count = 0

        output_dir = "extracted_frames"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        while self._run_flag and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Process frame with YOLOv5 model
            results = self.model(frame)
            annotated_frame = results.render()[0]
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.change_pixmap_signal.emit(convert_to_qt_format)

            # Save frame every 2 seconds
            if frame_number % frame_interval == 0:
                self.save_frame(frame, frame_count, output_dir)
                frame_count += 1

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

    def save_frame(self, frame, frame_count, output_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_filename = os.path.join(output_dir, f"frame_{timestamp}_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.openButton.clicked.connect(self.open_file)
        self.thread = None
        # self.model = torch.hub.load('ultralytics/yolov5','yolov5s')
        self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                    path='yolov5/runs/train/yolov5s_results22/weights/best.pt', force_reload=True)

    def open_file(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if video_path:
            if self.thread:
                self.thread.stop()
            self.thread = VideoThread(self.model, video_path)
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.start()

    def update_image(self, qt_image):
        pixmap = QPixmap.fromImage(qt_image)
        self.videoLabel.setPixmap(pixmap.scaled(self.videoLabel.size(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        if self.thread:
            self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
