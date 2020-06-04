import sys
from PySide2.QtWidgets import (QDialog, QPushButton, QVBoxLayout, QApplication, QMainWindow, QGridLayout, QComboBox, QSlider)
from PySide2.QtCore import QTimer, Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import threading
import time
import random
#
# class Window(QDialog):
#     def __init__(self, parent=None):
#         super(Window, self).__init__(parent)
#
#         # a figure instance to plot on
#         self.figure, self.ax = plt.subplots()
#         self.lines, = self.ax.plot([], [], 'o')
#
#         self._plot = self.figure
#
#         self.n_data = 30
#         self.xdata = list(range(self.n_data))
#         self.ydata = [random.randint(0, 10) for i in range(self.n_data)]
#
#         # this is the Canvas Widget that displays the `figure`
#         # it takes the `figure` instance as a parameter to __init__
#         self.canvas = FigureCanvas(self.figure)
#
#         # this is the Navigation widget
#         # it takes the Canvas widget and a parent
#         self.toolbar = NavigationToolbar(self.canvas, self)
#
#         # Just some button connected to `plot` method
#         self.button = QPushButton('Plot')
#         self.button.clicked.connect(self.plot)
#
#         # set the layout
#         layout = QVBoxLayout()
#         layout.addWidget(self.toolbar)
#         layout.addWidget(self.canvas)
#         layout.addWidget(self.button)
#         self.setLayout(layout)
#
#         self.timer = QTimer()
#         self.timer.setInterval(100)
#         self.timer.timeout.connect(self.update_plot)
#         self.timer.start()
#
#     def update_plot(self):
#         # Drop off the first y element, append a new one.
#         self.ydata = self.ydata[1:] + [random.randint(0, 10)]
#
#         # Note: we no longer need to clear the axis.
#         if self._plot is None:
#             # First time we have no plot reference, so do a normal plot.
#             # .plot returns a list of line <reference>s, as we're
#             # only getting one we can take the first element.
#             plot_refs = self.canvas.axes.plot(self.xdata, self.ydata, 'r')
#             self._plot = plot_refs[0]
#         else:
#             # We have a reference, we can use it to update the data for that line.
#             self._plot.set_ydata(self.ydata)
#             self.figure.set_
#
#         # Trigger the canvas to update and redraw.
#         self.canvas.draw()
#
#     def plot(self):
#         ''' plot some random stuff '''
#         # random data
#         self.ydata = [random.random() for i in range(self.n_data)]
#
#         # instead of ax.hold(False)
#         self.figure.clear()
#
#         # create an axis
#         ax = self.figure.add_subplot(111)
#
#         # discards the old graph
#         # ax.hold(False) # deprecated, see above
#
#         # plot data
#         p = ax.plot(self.ydata, '*-')
#         self._plot = p
#
#         # refresh canvas
#         self.canvas.draw()
#
#
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#
#     main = Window()
#     main.show()
#
#     sys.exit(app.exec_())

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(dpi=dpi)
        self.axes = fig.add_subplot(111)

        n_data = 50
        self.xdata = list(range(n_data))
        self.arrivals = []
        self.ydata = [random.randint(0, 10) for i in range(n_data)]
        super(MplCanvas, self).__init__(fig)


class MainWindow(QDialog):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__()


        self.control = False
        self.canvas = []
        self.update_plot()

        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer = QTimer()
        self.timer.setInterval(1500)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

        for x in range(4):


            canvasWid = MplCanvas(self, width=5, height=4, dpi=100)
            self.canvas.append(canvasWid)
            #self.setCentralWidget(self.canvas)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent

        # # this is the Canvas Widget that displays the `figure`
        # # it takes the `figure` instance as a parameter to __init__
        # self.canvas = FigureCanvas(self.figure)

        #self.toolbar = NavigationToolbar(self.canvas[0], self)

        # Just some button connected to `plot` method
        self.button = QPushButton('Plot')
        self.button.clicked.connect(self.pause)

        self.cb = QComboBox()
        self.ci = "0.7"
        self.cb.addItems(args[0])
        self.cb.currentIndexChanged.connect(self.selectionchange)

        self.sl = QSlider(Qt.Horizontal)
        self.sl.setMinimum(10)
        self.sl.setMaximum(2000)
        self.sl.setValue(1500)
        self.sl.setTickPosition(QSlider.TicksLeft)
        self.sl.setTickInterval(100)
        self.sl.valueChanged.connect(self.valuechange)
        self.cb.currentText()


        # set the layout
        layout = QGridLayout()
        layout.addWidget(self.cb,0,0,1,1)
        layout.addWidget(self.sl, 0, 1, 1, 1)

        layout.addWidget(self.canvas[0],2,0)
        layout.addWidget(self.canvas[1], 2, 1)
        layout.addWidget(self.canvas[2], 3, 0)
        layout.addWidget(self.canvas[3], 3, 1)
        layout.addWidget(self.button,4,0,1,1)
        self.setLayout(layout)



        self.show()

    def update_plot(self):
        print("DRAWING")
        for c in self.canvas:
            c.axes.cla()  # Clear the canvas.
            s = len(c.ydata)
            if len(c.arrivals) > 1:
                c.axes.plot(list(range(max(0,s-100),s)), c.ydata[-100:], 'r')
                c.axes.plot(list(range(max(0,s-100),s)), c.arrivals[-100:], 'b')
            else:
                c.axes.plot(list(range(s)), c.ydata, 'r')
            # Trigger the canvas to update and redraw.
            c.draw()

    def pause(self):
        if self.control:
            print("Stop")
            self.timer.stop()
        else:
            print("Start")
            self.timer.start()
            x = 1
        self.control = not self.control

    def selectionchange(self, i):
        print(self.cb.currentText())
        self.ci = self.cb.currentText()

    def valuechange(self):
        size = self.sl.value()
        self.timer.setInterval(size)
        print(size)

# def threader(window):
#
#     for x in range(100):
#         n = random.randint(0, 100)
#         window.canvas[0].ydata.append(n)
#         window.canvas[1].ydata.append(n*2)
#         window.canvas[2].ydata.append(n/2)
#         window.canvas[3].ydata.append(n*4)
#         time.sleep(2)
#
#
#
# app = QApplication(sys.argv)
# w = MainWindow()
# t = threading.Thread(target=threader,args=(w,))
# t.start()
# app.exec_()