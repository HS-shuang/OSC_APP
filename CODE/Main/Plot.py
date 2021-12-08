# -*- coding: utf-8 -*-
# @Time: 2021/11/5 10:17
# @Author: HS
from PyQt5 import QtCore
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas,
                                                NavigationToolbar2QT as NavigationToolbar)  # 用户界面后端渲染，用来以绘图的形式输出
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton
from matplotlib.figure import Figure  # 图表类
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy


class QFC(QWidget):
    mouseMove = QtCore.pyqtSignal(numpy.float64, mpl.lines.Line2D)  # 自定义触发信号，用于与UI交互

    def __init__(self, parent=None, toolbarVisible=True, showHint=False):
        super().__init__(parent)
        self.figure = Figure()  # 公共属性figure
        figCanvas = FigureCanvas(self.figure)  # 创建FigureCanvas对象

        self.naviBar = NavigationToolbar(figCanvas, self)  # 创建工具栏
        self.naviBar.setIconSize(figCanvas.sizeHint()/19)
        self.naviBar.setOrientation(QtCore.Qt.Vertical)

        actList = self.naviBar.actions()
        count = len(actList)
        self.__lastActtionHint = actList[count - 1]
        self.__showHint = showHint  # 是否显示坐标提示
        self.__lastActtionHint.setVisible(self.__showHint)
        self.__showToolbar = toolbarVisible  # 是否显示工具栏
        self.naviBar.setVisible(self.__showToolbar)

        layout = QHBoxLayout(self)
        layout.addWidget(figCanvas)  # 添加FigureCanvas对象
        layout.addWidget(self.naviBar)  # 添加工具栏
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.__cid = figCanvas.mpl_connect("scroll_event", self.do_scrollZoom)  # 支持鼠标滚轮缩放
        self.__cid1 = figCanvas.mpl_connect("pick_event", self.do_series_pick)  # 支持曲线抓取
        # self.__cid2 = figCanvas.mpl_connect("button_press_event",self.do_pressMouse)#支持鼠标按下
        self.__cid3 = figCanvas.mpl_connect("button_release_event", self.do_releaseMouse)  # 支持鼠标释放
        self.__cid4 = figCanvas.mpl_connect("motion_notify_event", self.do_moveMouse)  # 支持鼠标移动
        self.mouseIsPress = False
        self.pickStatus = False

    # 公共函数接口
    def setToolbarVisible(self, isVisible=True):  # 是否显示工具栏
        self.__showToolbar = isVisible
        self.naviBar.setVisible(isVisible)

    def setDataHintVisible(self, isVisible=True):  # 是否显示坐标提示
        self.__showHint = isVisible
        self.__lastActtionHint.setVisible(isVisible)

    def do_scrollZoom(self, event):  # 通过鼠标滚轮缩放
        ax = event.inaxes  # 产生事件axes对象
        if ax == None:
            return
        self.naviBar.push_current()
        xmin, xmax = ax.get_xbound()
        xlen = xmax - xmin
        ymin, ymax = ax.get_ybound()
        ylen = ymax - ymin

        xchg = event.step * xlen / 20
        xmin = xmin + xchg
        xmax = xmax - xchg
        ychg = event.step * ylen / 20
        ymin = ymin + ychg
        ymax = ymax - ychg
        ax.set_xbound(xmin, xmax)
        ax.set_ybound(ymin, ymax)
        event.canvas.draw()

    def do_series_pick(self, event):  # picker事件获取抓取曲线
        self.series = event.artist
        # index = event.ind[0]
        # print("series",event.ind)
        if isinstance(self.series, mpl.lines.Line2D):
            self.pickStatus = True

    def do_releaseMouse(self, event):  # 鼠标释放，释放抓取曲线
        if event.inaxes == None:
            return
        if self.pickStatus == True:
            self.series.set_color(color="black")
            self.figure.canvas.draw()
            self.pickStatus = False
        # self.mouseRelease.emit(event.xdata,event.ydata)

    def do_moveMouse(self, event):  # 鼠标移动，重绘抓取曲线
        if event.inaxes == None:
            return
        if self.pickStatus == True:
            self.series.set_xdata([event.xdata, event.xdata])
            self.series.set_color(color="red")
            self.figure.canvas.draw()
            self.mouseMove.emit(event.xdata, self.series)  # 自定义触发信号，用于与UI交互

    def my_ax(self, ax):
        ax.tick_params(direction='in')


class CanvasBack(QFC):
    def __init__(self):
        super(CanvasBack, self).__init__()
        self.ax1_lines = []
        self.ax1_scatters = []
        self.ax2_lines = []
        self.initFig()

    def initFig(self):
        x, y = [], []
        ax1 = self.figure.add_subplot(2, 1, 1)
        self.my_ax(ax1)
        ax1.set_ylabel('$R_{raw} (\Omega)$')
        ax1.set_xlabel('B (T)')
        ax1.xaxis.set_ticks_position('top')
        ax1.plot(x, y, '-o', c='k', ls='--', lw=1)
        ax1.plot(x, y, c='r')

        ax2 = self.figure.add_subplot(2, 1, 2)
        self.my_ax(ax2)
        ax2.set_ylabel('$R_{osc} (\Omega)$')
        ax2.set_xlabel('1/B (1/T)')
        ax2.plot(x, y, c='k', lw=2)
        ax2.plot(x, y, c='r', lw=1.5)

    def change_x_type(self, text):
        if text == 'B':
            self.figure.axes[1].set_xlabel('B (T)')
        elif text == '1/B': self.figure.axes[1].set_xlabel('1/B (1/T)')
        elif text == 'log(B)': self.figure.axes[1].set_xlabel('log(B)')

        self.figure.canvas.draw()

    def update(self, data_raw, data_filter, data_osc, data_osc_calcu, t):
        self.figure.suptitle(f'{t}')
        axes = self.figure.axes
        axes[0].lines[0].set_data(data_raw[0], data_raw[1])
        axes[0].lines[1].set_data(data_filter[0], data_filter[1])
        axes[1].lines[0].set_data(data_osc[0], data_osc[1])
        axes[1].lines[1].set_data(data_osc_calcu[0], data_osc_calcu[1])
        axes[0].relim()
        axes[0].autoscale()
        axes[1].relim()
        axes[1].autoscale()
        self.figure.canvas.draw()


class CanvasFFT(QFC):
    def __init__(self):
        super(CanvasFFT, self).__init__()
        self.initFig()

    def initFig(self):
        ax = self.figure.add_subplot()
        self.my_ax(ax)
        ax.grid(b=None, which='both', axis='both', ls='--')
        ax.set_ylabel('Amplitude (a.u.)')
        ax.set_xlabel('F (T)')

    def change_x_type(self, text):
        if text == 'B': self.figure.axes[0].set_xlabel('F (1/T)')
        elif text == '1/B': self.figure.axes[0].set_xlabel('F (T)')
        elif text == 'log(B)': self.figure.axes[0].set_xlabel('F (T)')
        self.figure.canvas.draw()

    def inputData(self, data_fft: dict):
        self.ax_lines = []
        self.ax_lines_stack = []
        n = len(data_fft)
        maximum = 0
        for i, t in enumerate(data_fft):
            data = data_fft[t]
            maximum = max(maximum, max(data[1]))
        for i, t in list(enumerate(data_fft))[::-1]:
            data = data_fft[t]
            self.ax_lines.append(self.figure.axes[0].plot(
                data[0], data[1], label=t, c=plt.cm.turbo(0.9 * (i+1) / n), lw=1.3)[0])
            self.ax_lines_stack.append(self.figure.axes[0].plot(
                data[0], data[1] + 0.7*i*maximum, label=t, c=plt.cm.turbo(0.9 * (i+1) / n), lw=1.3)[0])
        self.ax_lines = self.ax_lines[::-1]
        self.figure.axes[0].lines = []

    def update(self, num, r):
        ax = self.figure.axes[0]
        ax.lines = [self.ax_lines[num]]
        ax.relim()
        ax.autoscale()
        ax.set_xlim(r[0], r[1])
        ax.legend()
        self.figure.canvas.draw()

    def showAll(self, r, stack=False):
        ax = self.figure.axes[0]
        if stack:
            ax.lines = self.ax_lines_stack
        else:
            ax.lines = self.ax_lines
        ax.relim()
        ax.autoscale()
        ax.set_xlim(r[0], r[1])
        ax.legend(bbox_to_anchor=(0.99, 1.1), loc="upper left", fontsize=11)
        self.figure.canvas.draw()


class CanvasFitM(QFC):
    def __init__(self):
        super().__init__()
        self.initFig()

    def initFig(self):
        ax = self.figure.add_subplot()
        self.my_ax(ax)
        ax.set_ylabel('Amplitude (a.u.)')
        ax.set_xlabel('T (K)')

    def inputData(self, d_r: dict, d_c: dict, m: dict):
        self.ax_lines = []
        self.ax_scatters = []
        for i, peak_num in enumerate(d_r):
            self.ax_lines.append(self.figure.axes[0].plot(
                d_r[peak_num][0], d_r[peak_num][1], c='k')[0])
            self.ax_scatters.append(self.figure.axes[0].plot(
                d_c[peak_num][0], d_c[peak_num][1], 'o', c=f'C{peak_num}', label=f'$m_{peak_num}$={m[peak_num]:.3g}')[0])
        self.figure.axes[0].lines = []

    def update(self, num):
        ax = self.figure.axes[0]
        ax.lines = [self.ax_lines[num], self.ax_scatters[num]]
        ax.relim()
        ax.autoscale()
        ax.legend()
        self.figure.canvas.draw()

    def showAll(self):
        ax = self.figure.axes[0]
        ax.lines = [*self.ax_lines, *self.ax_scatters]
        ax.relim()
        ax.autoscale()
        ax.legend()
        self.figure.canvas.draw()


class CanvasFitFFT(QFC):
    def __init__(self):
        super().__init__()
        self.initFig()

    def initFig(self):
        ax = self.figure.add_subplot()
        self.my_ax(ax)
        ax.grid(b=None, which='both', axis='both', ls='--')
        ax.set_ylabel('Amplitude (a.u.)')
        ax.set_xlabel('F (T)')

    def draw_data(self, curve_fft, curve_fit, r=None, label=''):
        ax = self.figure.axes[0]
        ax.lines = []
        ax.plot(curve_fft[0], curve_fft[1], c='k', label=label)
        ax.plot(curve_fit[0], curve_fit[1], c='r', label='fit')
        ax.legend()
        ax.relim()
        ax.autoscale()
        if r:
            ax.set_xlim(r[0], r[1])
        self.figure.canvas.draw()


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    win = QWidget()
    L = QHBoxLayout()
    win.setLayout(L)
    plot = CanvasFFT()
    plot.inputData({i: [range(10), [x**i for x in range(10)]] for i in range(10)})
    L.addWidget(plot)

    global num
    num = 0


    def change():
        global num
        plot.update(num, [0, 10])
        num += 1
        print(plot.figure.axes[0].lines)
        print(plot.ax_lines)
        # plot.showAll([0, 20])

    c = QPushButton()
    L.addWidget(c)
    c.clicked.connect(change)

    win.show()
    sys.exit(app.exec_())

