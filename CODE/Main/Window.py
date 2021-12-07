# -*- coding: utf-8 -*-
# @Time: 2021/11/4 17:26
# @Author: HS
import pandas as pd
from PyQt5.QtWidgets import QMainWindow, QLineEdit, QHBoxLayout, QVBoxLayout, \
    QGridLayout, QLabel, QAction, QCheckBox, QComboBox, QMenu, QPushButton, \
    QDesktopWidget, qApp, QFileDialog, QFrame, QDoubleSpinBox, QSpinBox, \
    QScrollArea, QWidget
from PyQt5.QtGui import QIcon
from pathlib import Path
from DataProcess import R_B, getT, np
from CODE.Main import data_prepare
import scipy.signal as sig
from CODE.Main.MyWidgets import MypeakL, ChooseTemL
from CODE.Main.Plot import CanvasBack, CanvasFFT, CanvasFitM
import os


class Win(QMainWindow):

    def __init__(self):
        super().__init__()
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.setCentralWidget(self.scrollArea)

        # 布局为垂直
        self.Vbox = QVBoxLayout(self.scrollArea)
        self.scrollWidget = QWidget()
        self.scrollArea.setWidget(self.scrollWidget)
        self.scrollWidget.setLayout(self.Vbox)

        self.dataType = 'T'
        self.x_type = '1/B'
        self.filter = 'polynomial'
        self.backPage = 0
        self.fitPage = 0
        self.Win = None
        self.fft = None

        self.center()
        self.initCanvas()
        self.initWidgets()
        self.initUI()
        self.initFitMUI()

        self.save_dir = None

        self.statusBar().showMessage('Ready')

    def initCanvas(self):
        self.canvasRB = CanvasBack()
        self.canvasRB.setMinimumSize(self.size() / 3)
        self.canvasFFT = CanvasFFT()
        self.canvasFFT.setMinimumSize(self.size() / 3)
        self.canvasFitM = CanvasFitM()
        self.canvasFitM.setMinimumSize(self.size() / 3)

    def center(self):
        win = self.frameGeometry()
        desk = QDesktopWidget().availableGeometry().center()
        self.resize(QDesktopWidget().width()/1.5, QDesktopWidget().height()/1.2)
        win.moveCenter(desk)

    def initWidgets(self):
        self.FilePath = QLineEdit(self)
        # self.FilePath.setText('D:/科研数据/NbP 振荡/DATA/S1-xx-B-变温.csv')
        # self.TList = getT(self.FilePath.text())
        self.TList = []
        self.btnImp = QPushButton('...', self)
        self.btnImp.clicked.connect(self.showImportDialog)

        # 变温/变角
        self.dataTypeComB = QComboBox(self)
        self.dataTypeComB.addItem('变温')
        self.dataTypeComB.addItem('变角')
        self.dataTypeComB.activated[str].connect(self.changeDataType)

        # 升场/降场
        self.modeComB = QComboBox(self)
        self.modeComB.addItem('all')
        self.modeComB.addItem('up')
        self.modeComB.addItem('down')
        self.modeComB.activated[str].connect(self.changeMode)

        # 振荡类型
        self.xComB = QComboBox(self)
        self.xComB.addItem('1/B')
        self.xComB.addItem('B')
        self.xComB.addItem('log(B)')
        self.xComB.activated[str].connect(self.change_x_type)

        # 平滑范围
        self.rangeSmoothLE = [QDoubleSpinBox(), QDoubleSpinBox()]
        self.rangeSmoothLE[0].setDecimals(1)
        self.rangeSmoothLE[1].setDecimals(1)
        self.rangeSmoothLE[0].setValue(1)
        self.rangeSmoothLE[1].setValue(32)

        # 计算范围
        self.rangeLE = [QDoubleSpinBox(), QDoubleSpinBox()]
        self.rangeLE[0].setDecimals(1)
        self.rangeLE[1].setDecimals(1)
        self.rangeLE[0].setValue(2)
        self.rangeLE[1].setValue(8)

        # 背底方式与参数
        self.filterComB = QComboBox(self)
        self.filterComB.addItem('polynomial')
        self.filterComB.addItem('sg')
        self.filterComB.addItem('butter')
        self.filterComB.activated[str].connect(self.changeFilter)

        self.filterParameter = [QSpinBox(), QDoubleSpinBox()]
        self.filterParameter[1].setDecimals(1)
        self.filterParameter[0].setValue(5)
        self.filterParameter[1].setValue(0)

        # FFT精度
        self.dfCheck = QCheckBox('FFT精度调整:', self)
        self.dfCheck.setChecked(False)
        self.dfCheck.clicked.connect(self.showChangedf)
        self.df = QDoubleSpinBox()
        self.df.setDecimals(1)
        self.df.setValue(1)
        self.df.setMinimum(0.1)
        self.df.hide()

        # FFT窗口
        self.winComB = QComboBox(self)
        self.winComB.setMinimumSize(5, 5)
        self.winComB.addItem('')
        self.winComB.addItem('hamming')
        self.winComB.addItem('hann')
        self.winComB.addItem('barthann')
        self.winComB.addItem('blackman')
        self.winComB.addItem('blackmanharris')
        self.winComB.addItem('flattop')
        self.winComB.activated[str].connect(self.changeWin)

        # FFT展示
        self.FFTDisplayLE = [QDoubleSpinBox(), QDoubleSpinBox()]
        self.FFTDisplayLE[0].setDecimals(1)
        self.FFTDisplayLE[1].setDecimals(1)
        self.FFTDisplayLE[0].setMaximum(9999)
        self.FFTDisplayLE[1].setMaximum(9999)
        self.FFTDisplayLE[0].setValue(0)
        self.FFTDisplayLE[1].setValue(200)
        self.FFTDisplayLE[1].setSingleStep(50)

        # 是否微分
        self.dBCheck = QCheckBox('dB计算', self)

        # 运行1
        self.Run1 = QPushButton('Run1', self)
        self.Run1.clicked.connect(self.run1)

    def initUI(self):
        # menu
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('File')
        viewMenu = menubar.addMenu('View')
        runMenu = menubar.addMenu('Run')

        impMenu = QMenu('Import', self)
        imp1 = self.importAct()
        impMenu.addAction(imp1)
        fileMenu.addMenu(impMenu)
        fileMenu.addAction(self.exitAct())

        runBack = self.run1Act()
        runMenu.addAction(runBack)

        # 工具栏
        self.addToolBar('退出').addAction(self.exitAct())
        self.addToolBar('保存').addAction(self.saveAct())

        # 部件
        nx = QPushButton('->', self)
        nx.setMaximumHeight(self.height() / 40)
        nx.clicked.connect(self.changePage)
        self.allFFT = QCheckBox('show all')
        self.allFFT.setChecked(True)
        pre = QPushButton('<-', self)
        pre.clicked.connect(self.changePage)
        pre.setMaximumHeight(self.height() / 40)

        # 小Layout
        self.chooseTL = ChooseTemL(self.TList)

        rangeSub, rangeSmooth, DT = QHBoxLayout(), QHBoxLayout(), QHBoxLayout()
        xtypeL, modeL, filterL = QHBoxLayout(), QHBoxLayout(), QVBoxLayout()

        rangeSmooth.addWidget(QLabel('平滑范围:'))
        rangeSmooth.addWidget(self.rangeSmoothLE[0])
        rangeSmooth.addWidget(QLabel('to'))
        rangeSmooth.addWidget(self.rangeSmoothLE[1])

        rangeSub.addWidget(QLabel('计算范围:'))
        rangeSub.addWidget(self.rangeLE[0])
        rangeSub.addWidget(QLabel('to'))
        rangeSub.addWidget(self.rangeLE[1])

        DT.addWidget(QLabel('变温/变角:'))
        DT.addWidget(self.dataTypeComB)

        modeL.addWidget(QLabel('升场/降场:'))
        modeL.addWidget(self.modeComB)

        xtypeL.addWidget(QLabel('振荡类型:'))
        xtypeL.addWidget(self.xComB)

        filterChoose, filterPa = QHBoxLayout(), QHBoxLayout()
        filterChoose.addWidget(QLabel('背底扣除:'))
        filterChoose.addWidget(self.filterComB)
        filterPa.addWidget(QLabel('阶数:'))
        filterPa.addWidget(self.filterParameter[0])
        filterPa.addWidget(QLabel('窗数:'))
        filterPa.addWidget(self.filterParameter[1])
        filterL.addLayout(filterChoose)
        filterL.addLayout(filterPa)

        dfL = QHBoxLayout()
        dfL.addWidget(self.dfCheck)
        dfL.addWidget(self.df)

        WinL = QHBoxLayout()
        WinL.addWidget(QLabel('FFT窗口:'))
        WinL.addWidget(self.winComB)

        FFTDisplayL = QHBoxLayout()
        FFTDisplayL.addWidget(QLabel('FFT显示:'))
        FFTDisplayL.addWidget(self.FFTDisplayLE[0])
        FFTDisplayL.addWidget(QLabel('to'))
        FFTDisplayL.addWidget(self.FFTDisplayLE[1])

        RL = QHBoxLayout()
        RL.addWidget(self.dBCheck)
        RL.addWidget(self.Run1)

        changePageL = QHBoxLayout()
        changePageL.addStretch(1)
        changePageL.addWidget(pre)
        changePageL.addWidget(self.allFFT)
        changePageL.addWidget(nx)
        changePageL.addStretch(1)

        # 大布局
        grid = QGridLayout()
        grid.addWidget(QLabel('FilePath'), 1, 1)
        grid.addWidget(self.FilePath, 1, 2)
        grid.addWidget(self.btnImp, 1, 3)

        paL = QVBoxLayout()
        paL.addLayout(self.chooseTL)
        paL.addLayout(DT)
        paL.addLayout(modeL)
        paL.addLayout(xtypeL)
        paL.addLayout(rangeSmooth)
        paL.addLayout(rangeSub)
        paL.addLayout(filterL)
        paL.addLayout(dfL)
        paL.addLayout(WinL)
        paL.addLayout(FFTDisplayL)  # FFT
        paL.addLayout(RL)

        plotL = QVBoxLayout()
        plotL.addWidget(self.canvasRB)
        plotL.addLayout(changePageL)
        plotL.addWidget(self.canvasFFT)

        # 最终布局
        mainBox1 = QHBoxLayout()
        mainBox1.addLayout(paL)
        mainBox1.addLayout(plotL)
        mainBox1.setStretch(0, 1)
        mainBox1.setStretch(1, 2)

        self.Vbox.addLayout(grid)
        self.Vbox.addLayout(mainBox1)

        self.show()

    def initFitMUI(self):
        # 部件
        self.peaks = MypeakL()
        btnRun2 = QPushButton('Run2', self)
        btnRun2.clicked.connect(self.run2)

        nx = QPushButton('->', self)
        nx.setMaximumHeight(self.height()/40)
        nx.clicked.connect(self.changePageFit)
        self.allFit = QCheckBox('show all')
        self.allFit.setChecked(True)
        pre = QPushButton('<-', self)
        pre.clicked.connect(self.changePageFit)
        pre.setMaximumHeight(self.height() / 40)

        # 小布局
        changePageL2 = QHBoxLayout()
        changePageL2.addStretch(1)
        changePageL2.addWidget(pre)
        changePageL2.addWidget(self.allFit)
        changePageL2.addWidget(nx)
        changePageL2.addStretch(1)

        # 中布局
        findPa = QVBoxLayout()
        findPa.addLayout(self.peaks)
        findPa.addWidget(btnRun2)

        findPlot = QVBoxLayout()
        findPlot.addWidget(self.canvasFitM)
        findPlot.addLayout(changePageL2)

        # 大布局
        mainBox2 = QHBoxLayout()
        mainBox2.addLayout(findPa)
        mainBox2.addLayout(findPlot)
        mainBox2.setStretch(0, 1)
        mainBox2.setStretch(1, 2)

        # 终极布局

        self.mF = QFrame()
        # self.mF = QWidget()
        self.mF.setLayout(mainBox2)
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setFrameShadow(QFrame.Sunken)
        self.Vbox.addWidget(line1)

        self.Vbox.addWidget(self.mF)
        self.show()

    def exitAct(self):
        act = QAction(QIcon('../FIG/exit.ico'), 'Exit', self)
        act.setShortcut('Ctrl+Q')
        act.setStatusTip('退出')
        act.triggered.connect(qApp.quit)
        return act

    def importAct(self):
        act1 = QAction('.csv', self)
        act1.triggered.connect(self.showImportDialog)
        return act1

    def saveAct(self):
        act = QAction('Save', self)
        act.triggered.connect(self.showSaveDialog)
        return act

    def showImportDialog(self):
        homeDir = str(Path.home()) if not self.FilePath.text() else self.FilePath.text()
        fname = QFileDialog.getOpenFileName(self, '导入.csv文件', homeDir)
        if fname[0]:
            self.FilePath.setText(fname[0])
            self.analysisFile(fname[0])

    def showSaveDialog(self):
        if self.x_type == 'B': f = lambda x: x
        elif self.x_type == '1/B': f = lambda x: 1/x
        elif self.x_type == 'log(B)': f = lambda x: np.exp(x)

        if self.fft:
            if self.save_dir:
                save_dir = self.save_dir
            else:
                save_dir = self.FilePath.text()
            title = self.FilePath.text().split('/')[-1]
            f_name = QFileDialog.getExistingDirectory(self, '导出数据', save_dir)

            if f_name:
                self.save_dir = f_name
                path = f_name + f'/{title}'
                if not os.path.exists(path):
                    os.mkdir(path)
                save_osc(self.fft.data_raw, self.fft.data_osc_f, filename=path+r'\osc.csv', f_x=f)
                save(self.fft.fft, filename=path+r'\fft.csv')
                if self.fft.data_m_fit:
                    save(self.fft.data_m_fit, filename=path+r'\mass_fit.csv')
                if self.fft.data_m:
                    save(self.fft.data_m, filename=path+r'\mass_point.csv')


    def analysisFile(self, file_path):
        self.TList = getT(file_path)
        self.chooseTL.changeT(self.TList)
        self.fft = R_B(file_path, self.TList)
        first = data_prepare.read_data(self.fft.or_data, 0)
        self.rangeLE[1].setValue(max(first[0])//0.1/10 - 0.1)

    def showChangedf(self):
        if self.dfCheck.checkState():
            self.df.show()
        else:
            self.df.hide()

    def changePage(self):
        t = self.sender().text()
        if t == '<-': self.backPage -= 1
        elif t == '->': self.backPage += 1
        self.backPage %= len(self.chooseTL.choosing)
        t = self.chooseTL.choosing[self.backPage]
        fft_range = [float(x.text()) for x in self.FFTDisplayLE]

        # 背底
        d_r = self.fft.data_raw[t]
        if self.x_type == 'B': f = lambda x: x
        elif self.x_type == '1/B': f = lambda x: 1/x
        elif self.x_type == 'log(B)': f = lambda x: np.exp(x)
        self.canvasRB.update([f(d_r[0]), d_r[1]],
                             [f(d_r[0]), self.fft.data_filter_f[t](d_r[0])],
                             [d_r[0], self.fft.data_osc_f[t](d_r[0])],
                             self.fft.data_fft_calcu[t],
                             t)

        if self.allFFT.checkState() == 0:
            self.canvasFFT.update(self.backPage, fft_range)
        else:
            if self.dataType == 'R':
                self.canvasFFT.showAll(fft_range, stack=True)
            elif self.dataType == 'T':
                self.canvasFFT.showAll(fft_range)

    def changePageFit(self):
        t = self.sender().text()
        if t == '<-': self.fitPage -= 1
        elif t == '->': self.fitPage += 1
        self.fitPage %= self.peak_num

        if self.allFit.checkState() == 0:
            self.canvasFitM.update(self.fitPage)
        else:
            self.canvasFitM.showAll()

    def changeDataType(self, text):
        if text == '变温':
            self.dataType = 'T'
            self.mF.show()
        else:
            self.dataType = 'R'
            self.allFFT.setChecked(False)
            self.mF.hide()

    def changeMode(self, text):
        pass

    def change_x_type(self, text):
        self.x_type = text
        self.canvasRB.change_x_type(text)
        self.canvasFFT.change_x_type(text)

    def changeFilter(self, text):
        if text == 'sg':
            self.filterParameter[0].setValue(3)
            self.filterParameter[1].setValue(22)
        elif text == 'butter':
            self.filterParameter[0].setValue(3)
            self.filterParameter[1].setValue(10.0)
        elif text == 'polynomial':
            self.filterParameter[0].setValue(5)
            self.filterParameter[1].setValue(0.0)
        self.filter = text

    def changeWin(self, text):
        self.Win = getattr(sig.windows, text) if text else None

    def run1Act(self):
        act = QAction('Run1', self)
        act.setStatusTip('去除背底/平稳化,并计算FFT')
        act.setShortcut('Ctrl+Enter')
        act.triggered.connect(self.run1)
        return act

    def run1(self):
        if not self.fft:
            self.analysisFile(self.FilePath.text())
        range = [x.value() for x in self.rangeLE]
        rangeSmooth = [x.value() for x in self.rangeSmoothLE]
        filterParameter = [x.value() for x in self.filterParameter]
        self.fft.TList = np.array(self.chooseTL.choosing)
        self.df_value = self.df.value() if self.dfCheck.checkState() else None

        self.fft.background(range=range, smooth_range=rangeSmooth,
                       show_background=1,
                       mode=self.modeComB.currentText(),
                       sg_wn=filterParameter[1],
                       filter_type=self.filter,
                       filter_order=filterParameter[0],
                       wn=filterParameter[1],
                       range_T={},
                       xtype=self.x_type,
                       calcu_use_diff=self.dBCheck.checkState() == 2,
                       Win=self.Win,
                       df=self.df_value)
        self.canvasFFT.inputData(self.fft.fft)
        self.changePage()
        self.statusBar().showMessage('Finished', 1000)

    def run2(self):

        m_p0 = [1e-1, 1e-1]
        peak_dic = {}
        i = 1
        for peak in self.peaks.peaks:
            rl = peak.range[0].text()
            rr = peak.range[1].text()
            if rl and rr:
                peak_dic[i] = [float(rl), float(rr)]
                i += 1
        self.peak_num = i-1

        if self.fft:
            self.fft.find_peak(peak_dic)
            self.my_phy = self.fft.fit_m(p0=m_p0)

        self.canvasFitM.inputData(self.fft.data_m_fit, self.fft.data_m, self.fft.m)
        self.changePageFit()
        self.statusBar().showMessage('Finished', 1000)


def save_osc(data_raw: dict, data_osc_f, filename, f_x=None):
    out = pd.DataFrame()
    for key in data_raw.keys():
        current = pd.DataFrame()
        current[None] = data_raw[key][0] if not f_x else f_x(data_raw[key][0])
        current[str(key)] = data_osc_f[key](data_raw[key][0])
        out = pd.concat((out, current), axis=1)
    out.to_csv(filename, index=None)


def save(data: dict, filename):
    out = pd.DataFrame()
    for key in data.keys():
        current = pd.DataFrame()
        current[None] = data[key][0]
        current[str(key)] = data[key][1]
        out = pd.concat((out, current), axis=1)
    out.to_csv(filename, index=None)


