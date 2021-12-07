# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 16:54:01 2020
@author: HS
"""
from CODE.Main import calculate, data_prepare
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from scipy.optimize import leastsq
from numpy import pi
kb = 1.380649e-23
e = 1.6e-19
h_bar = 1.05457e-34
me = 9.11e-31


class R_B:
    def __init__(self, file_path, TList):
        """
        :param file_path: 文件路径，.csv文件，奇数列为相应FFT的原始横坐标
        :param TList: 计算时所包含的数据列名，当为变温数据时，列名为温度，改变TLIST可调整参与计算丁格尔温度的数据点
        :param datatype: 输入数据列名的变化：'T'对应变温，'R'对应变角，默认：'T'
        :param mode: 计算FFT的数据扫场方向，'up'升场（前半段，非严格升），'down'降场（后半段），默认：'up'
        :param range: FFT的计算范围。默认：15-31
        :param smooth_range: 数据平滑去背景时所用磁场范围。默认：0-32
        :param show_background: 是否用图片展示背地，默认：展示
        :param filter_type: 计算背地方式。‘sg'对应S-G滑动平均滤波，’butter‘对应巴特沃斯滤波，默认：‘sg’
        :param filter_order: ’sg'或'butter'的阶数，阶数越低越平滑，默认：2
        :param wn: 'butter’的截至频率，越低越平滑，选择'sg'时该参数无效，默认：100(背地为低频)
        :param sg_wn: 'sg'的窗口跨越的磁场范围，越大越平滑，'butter'时无效，默认：5
        :param range_T: 一个dic，key为温度点，value为对应数据FFT的range
        :param calcu: str, 是否计算diff
        """
        self.title = file_path[-9:-4]
        self.TListOr = getT(file_path)
        self.or_data = pd.read_csv(file_path, encoding='gbk')
        self.TList = np.array(TList)
        self.data_m = None
        self.data_m_fit = None
        self.m = None

    def background(self, mode='up', range=[15, 31], smooth_range=[1, 32],
                 filter_type='sg', filter_order=2, wn=100, sg_wn=5, range_T={}, xtype='1/B', show_background=1,
                 calcu_use_diff=True, Win=None, df=None):
        self.fft = {}
        self.fft_calcu = {}
        self.data_fft_calcu = {}
        self.data = {}
        self.data_raw = {}
        self.data_filter_f = {}
        self.data_osc_f = {}
        self.range = range
        self.Win = Win
        self.df = df
        for T in self.TList:
            i = self.TListOr.index(T)
            DataPre = data_prepare.read_data(self.or_data, i)
            DataPre = DataPre[:, (DataPre[0] <= smooth_range[1]) * (DataPre[0] >= smooth_range[0])]

            range = range_T[T] if T in range_T else self.range

            if mode == 'up':
                if DataPre[0][0] < DataPre[0][10]: DataPre = DataPre[:, :int(np.where(DataPre[0] == max(DataPre[0]))[0])]
                else: DataPre = DataPre[:, int(np.where(DataPre[0] == min(DataPre[0]))[0]):]
            elif mode == 'down':
                if DataPre[0][0] < DataPre[0][10]: DataPre = DataPre[:, int(np.where(DataPre[0] == max(DataPre[0]))[0]):]
                else: DataPre = DataPre[:, :int(np.where(DataPre[0] == min(DataPre[0]))[0])]

            if xtype == '1/B':
                DataPre[0], ibMin, ibMax = 1/DataPre[0], 1 / range[1], 1 / range[0]
                x_f = x_f1 = lambda x: 1/x
            elif xtype == 'log(B)':
                DataPre[0], ibMin, ibMax = np.log(DataPre[0]), np.log(range[0]), np.log(range[1])
                x_f1 = lambda x: np.exp(x)
                x_f = lambda x: np.log(x)
            else:
                ibMin, ibMax = range[0], range[1]
                x_f = x_f1 = lambda x: x

            self.data[T] = DataPre
            if calcu_use_diff:
                DataPre = np.array([DataPre[0][:-1], np.diff(DataPre[1])/np.diff(DataPre[0])])

            f_filter, f_osc = data_prepare.my_filter(DataPre, filter_type, filter_order, wn, sg_wn, x_f=x_f, x_f1=x_f1)

            N = len(DataPre[0])

            # 采样频率
            DeltaF = 1/(ibMax-ibMin)
            step = (ibMax - ibMin) / (N - 1)

            DeltaF = self.df if self.df and self.df < DeltaF else DeltaF
            my_ibMax = ibMin + 1/DeltaF
            N = int((my_ibMax - ibMin)/step) + 1

            i = 0
            while N:
                N >>= 1
                i += 1
            N = 1 << (i+2)

            ibfft = np.linspace(ibMin, my_ibMax, N)
            FrePre = DeltaF * np.arange(N)

            MyData = f_osc(ibfft)
            MyData[ibfft > ibMax] = 0

            if self.Win:
                MyData *= self.Win(N)

            f = FrePre[:int(N/2)]

            y_f_pre = fft(MyData)*2/N
            y_f_pre[0] = y_f_pre[0]/2
            y_f_pre[int(N/2)] = y_f_pre[int(N/2)]/2
            y_f_pre[int(N/2)+1] = y_f_pre[int(N/2)+1]/2
            y_f_pre[-1] = y_f_pre[-1]/2

            y_f1 = abs(y_f_pre)
            y_f = y_f1[:int(N/2)]

            self.data_raw[T] = DataPre
            self.data_filter_f[T] = f_filter
            self.data_osc_f[T] = f_osc
            self.fft[T] = np.vstack((f, y_f))
            self.fft_calcu[T] = np.vstack((FrePre, y_f_pre))
            self.data_fft_calcu[T] = np.vstack((ibfft, MyData))

    def find_peak(self, peak_range_dic={1: [1400, 1600], 2: [1900, 2100], 3: [2800, 3000]}):
        self.TD = []
        self.peakdic = {}
        for ii in range(len(peak_range_dic)):
            x = np.array([])
            for T in self.TList:
                f = self.fft[T][0]
                y_f = self.fft[T][1]

                lim = peak_range_dic[ii+1]
                inten = max(y_f[(f > lim[0])*(f < lim[1])])
                x = np.hstack((x, f[y_f == inten], inten))
            x = np.array(x)
            x = x.reshape(-1, 2).T
            self.peakdic[ii+1] = x
        for i in range(len(self.peakdic)):
            self.TD.append(None)
        return self.peakdic

    def fit_m(self, p0=[1, 1]):
        self.data_m = {}
        self.data_m_fit = {}
        self.m = {}
        if 1 not in self.peakdic:
            self.find_peak()
        for peak_num in self.peakdic:
            data = self.peakdic[peak_num]

            def f(pa, x):
                return pa[0]*x/np.sinh(pa[1]*x)

            def loss(pa):
                return f(pa, self.TList) - data[1]

            pa = leastsq(loss, p0)[0]
            B = sum(self.range)/2
            # 用25T计算
            m = h_bar*e*B*pa[1]/(2*pi*pi*kb*me)
            F = data[0][peak_num-1]
            phy = calculate.get_phy(m, F, TD=self.TD[peak_num - 1])
            self.data_m[peak_num] = np.vstack((self.TList, data[1]))
            Tplot = np.linspace(min(self.TList), max(self.TList), 100)
            self.data_m_fit[peak_num] = np.vstack((Tplot, f(pa, Tplot)))
            self.m[peak_num] = m
            myphy = [phy] if peak_num == 1 else np.vstack((myphy, phy))
        return myphy

    def show_fft(self, T=4, FFT_range=[0, 3000]):
        b_min = self.range[0]
        b_max = self.range[1]
        f = self.fft[T][0]
        y_f = self.fft[T][1]
        title = self.title

        fig = plt.figure()
        fig.text(0.3, 0.9, '%s range=%s~%s(T)' % (self.title, b_min, b_max), fontsize=13)
        fig.text(0.4, 0.82, 'T=%s' % T, c='b', fontsize=13)
        plt.plot(f, y_f, label=title, c='b')
        plt.scatter(f, y_f, c='k', s=5)
        plt.legend()
        plt.xlabel('f (T)')
        plt.ylabel('$FFT\ Amplitude^{}$')
        for ii in range(len(self.peakdic)):
            plt.scatter(self.peakdic[ii+1][0][self.TList == T],
                        self.peakdic[ii+1][1][self.TList == T], c='r')
        plt.xlim(FFT_range[0], FFT_range[1])

        print(T, ':', f[y_f == max(y_f)])
        self.FigFFT[T] = fig
        return fig

    def show_fft_all(self, FFT_range=[0, 3000]):
        b_min = self.range[0]
        b_max = self.range[1]
        fig = plt.figure()
        fig.text(0.3, 0.9, '%s range=%s~%s(T)' % (self.title, b_min, b_max), fontsize=13)
        i = 0
        for T in self.TList:
            f = self.fft[T][0]
            y_f = self.fft[T][1]
            plt.plot(f, y_f, label=T, c=plt.cm.turbo_r(0.9 * (i+1) / len(self.TList)), lw=1.3)
            i += 1
        plt.legend()
        plt.xlabel('f (T)')
        plt.ylabel('FFT Amplitude')
        plt.xlim(FFT_range[0], FFT_range[1])
        self.FigFFT['all'] = fig
        return fig

    def show_fft_all_stacked(self, FFT_range=[0, 3000]):
        b_min = self.range[0]
        b_max = self.range[1]
        fig = plt.figure()
        fig.text(0.3, 0.9, '%s range=%s~%s(T)' % (self.title, b_min, b_max), fontsize=13)
        for i, T in enumerate(self.TList):
            f = self.fft[T][0]
            y_f = self.fft[T][1]
            y_f = y_f/max(y_f) + i
            plt.plot(f, y_f, label=T, c=plt.cm.turbo_r(0.9 * (i+1) / len(self.TList)), lw=1.3)
        plt.legend()
        plt.xlabel('f (T)')
        plt.ylabel('FFT Amplitude')
        plt.xlim(FFT_range[0], FFT_range[1])
        self.FigFFT['all'] = fig
        return fig

    def fit_FFT(self, T=1.8, peak_num=5, f_range=[0, 6000], mym=[], F=[], x0=[]):
        data = self.fft_calcu[T]
        data = data[:, :int(len(data[0])/2)]

        datar = self.data_fft_calcu[T]

        ind = (data[0] >= f_range[0])*(data[0] <= f_range[1])
        fftdata = data[:, ind]

        n = len(F)
        print(T)

        def f(pa):
            x = 1/datar[0]
            y = 0
            # F = pa[3*n:4*n]
            for i in range(n):
                lam = 2*pi*pi*kb*T*mym[i] * me / (h_bar*e*x)
                lamd = 2*pi*pi*kb * pa[i] * mym[i] * me / (h_bar*e*x)
                y += pa[i+n] * 5/2 * (x/(2*F[i]))**(1/2) * lam/np.sinh(lam) * np.exp(-lamd) * np.cos(2*pi*F[i]/x + pa[i+2*n])
            return y

        m = int(len(datar[0])/2)

        def loss(pa):
            myfft = fft(f(pa))[:m]/m
            myfft[0] = myfft[0]*2
            myfft[-1] = myfft[-1]*2
            return abs(myfft[ind] - fftdata[1])

        pa = leastsq(loss, x0=x0, maxfev=50000)[0]
        calcu_fft = abs(fft(f(pa)))[:m][ind]/m
        calcu_fft[0] = calcu_fft[0]*2
        calcu_fft[-1] = calcu_fft[-1]*2

        f1 = plt.figure()
        for i in range(n):
            x = 0.014+0.23*(np.mod(i, 5))
            if i >= 5:
                y = 1-0.05
            else:
                y = 1
            f1.text(x, y, '$T_{D%s}=%.2f$' % (i+1, pa[i]), c='k', fontsize=15)
        # plt.title('fit of FFT')
        plt.scatter(fftdata[0], abs(fftdata[1]), ec='k', fc='w', s=7)
        plt.plot(fftdata[0], abs(fftdata[1]), c='C1', label='exp')
        plt.plot(fftdata[0], calcu_fft, c='r', label='fit')
        plt.xlabel('Frequecy(T)')
        plt.ylabel('Amplitude')
        plt.legend()
        self.FigFitFFT[T] = f1

        if self.Win:
            datar[1] /= self.Win(len(datar[1]))
            fit = f(pa)/self.Win(len(datar[1]))

        self.FigFitFFT[str(T)+'-2'] = plt.figure(dpi=400)
        plt.scatter(datar[0], datar[1], ec='b', fc='w', s=3)
        plt.plot(datar[0], datar[1], c='b', lw=1)
        plt.plot(datar[0], fit, c='r', label='fit')
        plt.legend()

        self.TD = pa[:n]
        print(pa)


def getT(file):
    with open(file) as f:
        line = f.readline()[:-1]
    T = []
    for t in line.split(',')[1::2]:
        t = t.strip('kK°')
        T.append(float(t) if '.' in t else int(t))
    return T
