# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:08:04 2019

@author: patri
"""

import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton
from PyQt5.QtGui import QIcon


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import random

class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.left = 100
        self.top = 100
        self.title = 'PyQt5 matplotlib example - pythonspot.com'
        self.width = 640
        self.height = 400
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        m = PlotCanvas(self, width=5, height=4)
        m.move(0,0)

        button = QPushButton('PyQt5 button', self)
        button.setToolTip('This s an example button')
        button.move(500,0)
        button.resize(140,100)

        self.show()


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot()


    def plot(self):
        data = [random.random() for i in range(25)]
        ax = self.figure.add_subplot(111)
        ax.plot(data, 'r-')
        ax.set_title('PyQt Matplotlib Example')
        self.draw()

if __name__ == '__main__':
#    app = QApplication(sys.argv)
#    ex = App()
#    for i in range(890):
#        ex = App()
#    sys.exit(app.exec_())
#    fig = plt.figure()
#    a = plt.subplot2grid((6, 1), (0, 0), rowspan=2, colspan=1)
#    b = plt.subplot2grid((6, 1), (2, 0), rowspan=8, colspan=1, sharex=a)
#    plt.subplots_adjust(left=0.11, bottom=0.24, right=0.90, 
#      top=0.90, wspace=0.2, hspace=0)
#    plt.show(block=False)
    plt.ion()
    for i in range(5000):
        data = [random.random() for i in range(25)]
        plt.plot(data)
        plt.draw()
        plt.pause(0.001)
        plt.clf()
        #plt.plot(data)