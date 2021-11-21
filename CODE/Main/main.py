# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 20:54:06 2020

@author: HS
"""
import matplotlib.pyplot as plt
import numpy as np

from DataProcess import *
import sys
import scipy.signal as sig
sys.path.append(r'../../')
plt.figure()
plt.show()
p1 = p2 = p3 = p4 = p5 = p6 = 10
Tlist = [1.8, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
# Tlist = [1.8, 3, 4]

peakdic = {}
m_p0 = [1e-1, 1e-1]
peak_range = [[10, 51], [70, 101]]
num = len(peak_range)
delta = 50

for i in range(len(peak_range)):
    peakdic[i+1] = peak_range[i]

title = '20-08-22#2_up'
smooth_range = [1, 32]
fftrange = [0, 200]

a = R_B('D:/科研数据/振荡处理/DATA/20-08-22#2.csv', Tlist,
        range=[4, 8],
        smooth_range=[2, 32],
        show_background=1,
        datatype='T',
        mode='down',
        sg_wn=22,
        filter_type='sg',
        filter_order=4,
        wn=0.004,
        range_T={},
        xtype='1/B',
        calcu_use_diff=False,
        Win=sig.windows.hamming)
a.FigBackground[1.8].show()
a.show_fft_all(fftrange)
a.FigFFT['all'].show()

# a.find_peak(peak_range_dic=peakdic)
#
# for i in range(num):
#     locals()['p%s' % (i+1)] = a.peakdic[i+1][0][0]
#
# # 展示FFT
# # a.show_fft_all(fftrange)
# for i in Tlist:
#     T = str(i)+'K'
#     a.show_fft(T, fftrange)
#
# # 拟合有效质量
# data_peak = pd.DataFrame()
# data_peak_fit = pd.DataFrame()
# data_peak['T(K)'] = Tlist
# for i in range(num):
#     data1 = pd.DataFrame()
#     i += 1
#     mydata, phy = a.fit_m(i, p0=m_p0)
#     data_peak['peak_' + str(i)] = mydata['FFT Amplitude']
#     data_peak_fit['T_fit(K)' + str(i)] = mydata['T_fit(K)']
#     data_peak_fit['peak_' + str(i)] = mydata['fit']
#     if i == 1:
#         myphy = [phy]
#     else:
#         myphy = np.vstack((myphy, phy))
# data_peak.to_csv('../save/Tem/%s_peak_exp.csv' % title, index=None)
# data_peak_fit.to_csv('../save/Tem/%s_peak_fit.csv' % title, index=None)
#
# #%% 数据保存
# data1 = pd.DataFrame()
# data2 = pd.DataFrame()
# for i in Tlist:
#     T = str(i)+'K'
#     datar = pd.DataFrame()
#     datar[T+'_1/B'] = a.data_without_background[T][0]
#     datar[T+'_R'] = a.data_without_background[T][1]
#     data1 = pd.concat((data1, datar), axis=1)
#
#     data_fft = pd.DataFrame()
#     data_fft[T+'_f'] = a.fft[T][0]
#     data_fft[T] = a.fft[T][1]
#     data2 = pd.concat((data2, data_fft), axis=1)
#
# data1.to_csv('../save/Tem/%s_osc_without_background.csv' % title, index=None)
# data2.to_csv('../save/Tem/%s_FFT_data.csv'%title, index=None)
#
# # TD
# mym = [x[2] for x in myphy]
#
# a.fit_FFT(T='3K', peak_num=num, f_range=[0, 150], mym=mym,
#           F=[p1, p2-5],
#           x0=[5.6, 2, 0.1, 0.1, 0.1, 0.1])

#
# F = [p1, p2]
# TD = a.TD
# for i in range(num):
#     if i == 0:
#         padata = calculate.get_phy(mym[i], F[i], TD[i])
#     else:
#         padata = np.vstack((padata, calculate.get_phy(mym[i], F[i], TD[i])))
#
# y = ifft(a.fft_calcu['1.8K'][1])
# # plt.plot(a.fft_calcu['1.8K'][0],abs(a.fft_calcu['1.8K'][1]))
# # plt.xlim(0,6000)
# plt.plot(y[:2000])
#
#
