import sys
from PySide2.QtWidgets import (QDialog, QPushButton, QVBoxLayout, QApplication, QMainWindow, QWidget)
from PySide2.QtCore import QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import multiprocessing as mp
from threading import Thread
import pyqtgraph as pg
import numpy as np
import sys
import random
import time


import random

class CrosshairPlotWidget(QWidget):
    """Scrolling plot with crosshair"""

    def __init__(self, parent=None):
        super(CrosshairPlotWidget, self).__init__(parent)

        # Use for time.sleep (s)
        self.FREQUENCY = .025
        # Use for timer.timer (ms)
        self.TIMER_FREQUENCY = self.FREQUENCY * 1000

        self.LEFT_X = -10
        self.RIGHT_X = 0
        self.x_axis = np.arange(self.LEFT_X, self.RIGHT_X, self.FREQUENCY)
        self.buffer = int((abs(self.LEFT_X) + abs(self.RIGHT_X))/self.FREQUENCY)
        self.data = []

        self.crosshair_plot_widget = pg.PlotWidget()
        self.crosshair_plot_widget.setXRange(self.LEFT_X, self.RIGHT_X)
        self.crosshair_plot_widget.setLabel('left', 'Value')
        self.crosshair_plot_widget.setLabel('bottom', 'Time (s)')
        self.crosshair_color = (101,255,183)

        self.crosshair_plot = self.crosshair_plot_widget.plot()

        self.layout = QtGui.QGridLayout()
        self.layout.addWidget(self.crosshair_plot_widget)

        self.crosshair_plot_widget.plotItem.setAutoVisible(y=True)
        self.vertical_line = pg.InfiniteLine(angle=90)
        self.horizontal_line = pg.InfiniteLine(angle=0, movable=False)
        self.vertical_line.setPen(self.crosshair_color)
        self.horizontal_line.setPen(self.crosshair_color)
        self.crosshair_plot_widget.setAutoVisible(y=True)
        self.crosshair_plot_widget.addItem(self.vertical_line, ignoreBounds=True)
        self.crosshair_plot_widget.addItem(self.horizontal_line, ignoreBounds=True)

        self.crosshair_update = pg.SignalProxy(self.crosshair_plot_widget.scene().sigMouseMoved, rateLimit=60, slot=self.update_crosshair)

        self.update_data_thread = Thread(target=self.plot_updater, args=())
        self.update_data_thread.daemon = True
        self.update_data_thread.start()

    def plot_updater(self):
        """Updates data buffer with data value"""

        while True:
            self.data_point = random.randint(1,101)
            if len(self.data) >= self.buffer:
                del self.data[:1]
            self.data.append(float(self.data_point))
            self.crosshair_plot.setData(self.x_axis[len(self.x_axis) - len(self.data):], self.data)
            time.sleep(self.FREQUENCY)

    def update_crosshair(self, event):
        """Paint crosshair on mouse"""

        coordinates = event[0]
        if self.crosshair_plot_widget.sceneBoundingRect().contains(coordinates):
            mouse_point = self.crosshair_plot_widget.plotItem.vb.mapSceneToView(coordinates)
            index = mouse_point.x()
            if index > self.LEFT_X and index <= self.RIGHT_X:
                self.crosshair_plot_widget.setTitle("<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y=%0.1f</span>" % (mouse_point.x(), mouse_point.y()))
            self.vertical_line.setPos(mouse_point.x())
            self.horizontal_line.setPos(mouse_point.y())

    def get_crosshair_plot_layout(self):
        return self.layout

if __name__ == '__main__':
    # Create main application window
    app = QtGui.QApplication([])
    app.setStyleSheet("""
        QWidget {
            background-color: #19232D;
            border: 0px solid #32414B;
            padding: 0px;
            color: #F0F0F0;
            selection-background-color: #1464A0;
            selection-color: #F0F0F0;
        }""")
    app.setStyle(QtGui.QStyleFactory.create("Cleanlooks"))
    mw = QtGui.QMainWindow()
    mw.setWindowTitle('Crosshair Plot')

    # Create and set widget layout
    # Main widget container
    cw = QtGui.QWidget()
    ml = QtGui.QGridLayout()
    cw.setLayout(ml)
    mw.setCentralWidget(cw)

    # Create crosshair plot
    crosshair_plot1 = CrosshairPlotWidget()
    crosshair_plot2 = CrosshairPlotWidget()

    ml.addLayout(crosshair_plot1.get_crosshair_plot_layout(),0,0,1,1)
    ml.addLayout(crosshair_plot2.get_crosshair_plot_layout(),0,1,1,1)
    mw.show()

    ## Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()