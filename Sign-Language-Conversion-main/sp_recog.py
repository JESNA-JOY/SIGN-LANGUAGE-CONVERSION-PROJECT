import sys
import cv2
import numpy as np
import speech_recognition as sr
from sp_proc import SpProc
from PyQt5 import QtWidgets, QtGui, QtCore
import qdarkstyle


class SpRecog(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(SpRecog, self).__init__(parent)
        self.setWindowTitle("Speech-Recog")
        self.sp = SpProc()
        self.res_text = ""
        frame = np.zeros((self.sp.height, self.sp.width, 3), dtype=np.uint8)
        self.image = QtGui.QImage(
            frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        self.seq = None

        self.landing_page = LandingPage(self, self.sp.height, self.sp.width)
        self.setCentralWidget(self.landing_page)

    def start_app(self):
        self.landing_page.close()
        self.recording_page = RecordingPage(
            self, self.sp.height, self.sp.width)
        self.setCentralWidget(self.recording_page)

    def rec_audio(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
            try:
                self.res_text = r.recognize_google(audio_data=audio)
                print("Res text : ", self.res_text)
                self.recording_page.text_box.setPlainText(self.res_text)
                gloss = self.sp.get_gloss(self.res_text)
                self.seq = self.sp.get_seq(gloss, num=None)
                if gloss:
                    print("Result gloss : ", gloss)
                else:
                    print("No matching gloss found")
                self.timer.start(10)
            except Exception as err:
                print(f"Error Recognizing audio : {err}")

    def display_seq(self):
        if self.seq:
            img = self.seq.pop(0)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channel = img.shape
            bytesPerLine = 3 * width
            self.image = QtGui.QImage(
                img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            self.recording_page.canvas.setPixmap(
                QtGui.QPixmap.fromImage(self.image))
            self.recording_page.canvas.setScaledContents(True)
            QtCore.QTimer.singleShot(10000, self.display_seq)
        else:
            print("No sequence to display")
            self.timer.stop()


class LandingPage(QtWidgets.QWidget):
    def __init__(self, parent=None, height=480, width=640):
        super(LandingPage, self).__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        title_label = QtWidgets.QLabel("Speech Recognition", self)
        title_font = QtGui.QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        description_label = QtWidgets.QLabel("Application to show ASL sequences from recognized sentences", self)
        description_font = QtGui.QFont()
        description_font.setPointSize(12)
        description_label.setFont(description_font)
        layout.addWidget(description_label)

        frame = np.zeros((height, width, 3), dtype=np.uint8)
        image = QtGui.QImage(
            frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        canvas = QtWidgets.QLabel(self)
        canvas.setPixmap(QtGui.QPixmap.fromImage(image))
        layout.addWidget(canvas)

        start_btn = QtWidgets.QPushButton("Start", self)
        start_btn.clicked.connect(parent.start_app)
        layout.addWidget(start_btn)


class RecordingPage(QtWidgets.QWidget):
    def __init__(self, parent=None, height=480, width=640):
        super(RecordingPage, self).__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        self.canvas = QtWidgets.QLabel(self)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        image = QtGui.QImage(
            frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        self.canvas.setPixmap(QtGui.QPixmap.fromImage(image))
        layout.addWidget(self.canvas)

        self.text_label = QtWidgets.QLabel("Res text : ", self)
        layout.addWidget(self.text_label)

        self.text_box = QtWidgets.QTextEdit(self)
        self.text_box.setFixedHeight(100)
        layout.addWidget(self.text_box)

        self.rec_start_btn = QtWidgets.QPushButton(self)
        self.rec_start_btn.setText("Record")
        self.rec_start_btn.clicked.connect(parent.rec_audio)
        layout.addWidget(self.rec_start_btn)

        parent.timer = QtCore.QTimer(parent)
        parent.timer.timeout.connect(parent.display_seq)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    sp_recog = SpRecog()
    sp_recog.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
