import sys
import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import librosa as ls
from PyQt5 import uic
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from db import callSelectAll, callInsert, callDelete
from recording import rec_voice
from fourier import mfcc_alg, difference

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(sys.path[0] + r'\AppDesign.ui', self)
        self.setFixedSize(750, 690)

        self.resultWindow = self.findChild(QTextEdit, 'resultWindow')
        self.input = self.findChild(QLineEdit, 'inputName')

        self.checkRB = self.findChild(QRadioButton, 'checkRB')
        self.insertRB = self.findChild(QRadioButton, 'insertRB')
        self.deleteRB = self.findChild(QRadioButton, 'deleteRB')

        self.checkButton = self.findChild(QPushButton, 'checkButton')
        self.checkButton.clicked.connect(self.checkVoicesClick)
        self.recButton = self.findChild(QPushButton, 'recButton')
        self.recButton.clicked.connect(self.recordVoiceClick)
        self.insertButton = self.findChild(QPushButton, 'insertButton')
        self.insertButton.clicked.connect(self.insertVoiceClick)
        self.deleteButton = self.findChild(QPushButton, 'deleteButton')
        self.deleteButton.clicked.connect(self.deleteVoiceClick)

    def recordVoiceClick(self):
        rec_voice()
        self.resultWindow.setText("Voice has successfully been recordered")
    
    def insertVoiceClick(self):
        try:
            if self.insertRB.isChecked() and self.inputName.text() != "":
                speaker, _ = ls.load(sys.path[0] + r'\Data\output.wav', sr=16000)

                callInsert(self.inputName.text(), str(mfcc_alg(speaker)))
                self.resultWindow.setText("Voice has successfully been inserted")
            elif self.inputName.text() == "":
                QMessageBox.about(self, "Error!", "Please, enter your name!")
            else:
                QMessageBox.about(self, "Error!", "Please, set insert radio button before adding data!")
        except Exception as e:
            QMessageBox.about(self, "Error!", repr(e))

    def deleteVoiceClick(self):
        if self.deleteRB.isChecked() and self.inputName.text() != "":
            callDelete(self.inputName.text())
            self.resultWindow.setText("Voice has successfully been deleted")
        elif self.inputName.text() == "":
            QMessageBox.about(self, "Error!", "Please, enter your name!")
        else:
            QMessageBox.about(self, "Error!", "Please, set delete radio button before deleting!")
    
    def checkVoicesClick(self):
        if self.checkRB.isChecked():
            key_speaker, _ = ls.load(sys.path[0] + r'\Data\output.wav')
            key_coefs = mfcc_alg(key_speaker)

            all_data = callSelectAll()

            distances = []
            for data in all_data:
                distances.append(difference(eval(data['coefs']), key_coefs))

            min_index = distances.index(min(distances))
            self.resultWindow.setText("I think this voice is like {}\'s".format(all_data[min_index]['name']))
        else:
            QMessageBox.about(self, "Error!", "Please, set check radio button before checking!")

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
    
if __name__ == "__main__":
    main()
