# -*- coding: utf-8 -*-
# @Time: 2021/11/8 20:17
# @Author: HS

from PyQt5.QtWidgets import QLineEdit, QHBoxLayout, QLabel, \
    QPushButton, QScrollArea, QWidget, QGridLayout, QVBoxLayout, \
    QCheckBox, QDoubleSpinBox

from math import log10


class MypeakL(QGridLayout):

    def __init__(self):
        super(MypeakL, self).__init__()
        self.n = 2
        self.peaks = [Peak('peak1'), Peak('peak2')]
        self.initWid()
        self.initUI()
        self.setSpacing(10)

    def initWid(self):
        self.btnAdd = QPushButton('+')
        self.btnAdd.clicked.connect(self.changeNum)
        self.btnAdd.setMinimumSize(30, 15)
        self.btnSub = QPushButton('-')
        self.btnSub.clicked.connect(self.changeNum)
        self.btnSub.setMinimumSize(30, 15)

    def initUI(self):
        self.addLayout(self.peaks[0], 1, 1, 1, 4)
        self.addLayout(self.peaks[1], 2, 1, 1, 4)
        self.addWidget(self.btnAdd, 1, 5)
        self.addWidget(self.btnSub, 1, 6)

    def changeNum(self):
        t = self.sender().text()
        if t == '+':
            self.n += 1
            self.peaks.append(Peak('peak'+str(self.n)))
            self.addLayout(self.peaks[self.n-1], self.n, 1, 1, 4)
        elif t == '-' and self.n > 1:
            self.n -= 1
            deletL(self.peaks.pop())


class Peak(QHBoxLayout):
    def __init__(self, peak_name):
        super(Peak, self).__init__()
        self.range = [QLineEdit(), QLineEdit()]
        self.range[0].setMinimumSize(20, 15)
        self.range[1].setMinimumSize(20, 15)
        self.addWidget(QLabel(peak_name))

        self.addWidget(self.range[0])
        self.addWidget(QLabel('~'))
        self.addWidget(self.range[1])


class MyFitPeakL(QVBoxLayout):

    def __init__(self):
        super().__init__()
        self.peaks = []
        self.set_peaks()

    def set_peaks(self, peaks={}):
        # 没有删除干净 注意
        while self.peaks:
            deletL(self.peaks.pop())
        for peak, val in peaks.items():
            peakL = FitPeak(peak, val)
            self.addLayout(peakL)
            self.peaks.append(peakL)


class FitPeak(QHBoxLayout):
    def __init__(self, peak, val):
        super(FitPeak, self).__init__()
        self.setSpacing(0)
        self.check_box = QCheckBox(f'{peak}={val:.3g}+')
        self.check_box.setChecked(True)
        self.delta = QDoubleSpinBox()
        self.delta.setMinimum(-val)
        self.delta.setMaximum(val)
        self.delta.setDecimals(2-log10(val)//1)
        self.delta.setSingleStep(val/500)

        self.addWidget(self.check_box)
        self.addWidget(self.delta)


def deletL(myLayout):
    item_list = list(range(myLayout.count()))
    item_list.reverse()  # 倒序删除，避免影响布局顺序

    for i in item_list:
        item = myLayout.itemAt(i)
        myLayout.removeItem(item)
        if item.widget():
            item.widget().deleteLater()


class ChooseTemL(QGridLayout):
    def __init__(self, ts):
        super(ChooseTemL, self).__init__()
        self.setSpacing(0)

        # self.scroll = QScrollArea()
        # self.scroll.setWidgetResizable(True)
        # self.content = QWidget()
        # self.scroll .setWidget(self.content)
        # self.content.setLayout(self)
        self.btnList = []
        self.addT(ts)

    def addT(self, ts):
        self.choosing = ts
        for i, t in enumerate(ts):
            btn = QPushButton(f'{t}')
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.clicked.connect(self.change)
            btn.setMinimumSize(10, 20)
            self.btnList.append(btn)
            self.addWidget(btn, i//5, i % 5)

    def change(self):
        sender = self.sender()
        t = sender.text()
        t = float(t) if '.' in t else int(t)
        if t in self.choosing:
            self.choosing.remove(t)
        else:
            self.choosing.append(t)
        self.choosing.sort()

    def changeT(self, ts):
        deletL(self)
        self.addT(ts)


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    win = QWidget()
    win.setLayout(ChooseTemL(list(range(10))))
    win.show()
    sys.exit(app.exec_())
