# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:50:52 2020

@author: HS
"""
import numpy as np
from scipy.signal import savgol_filter as sg
from scipy import signal
from scipy.interpolate import interp1d


def read_data(data, i):     # 读取第i组数据
    dL = data.columns
    ret = np.array([data[dL[2*i]].dropna(), data[dL[2*i+1]].dropna()])
    if max(ret[0]) > 10000:
        ret[0] /= 10000
    return ret


def my_filter(data_pre, filter_type='sg', filter_order=2, wn=0.01, sg_wn=2):
    f = interp1d(data_pre[0], data_pre[1], fill_value=0, bounds_error=False)
    n = len(data_pre[0])
    filter_x = np.linspace(min(data_pre[0]), max(data_pre[0]), 8*n)

    if filter_type == 'butter':
        wn *= (filter_x[1] - filter_x[0]) * 2
        b, a = signal.butter(filter_order, wn, 'lp')
        filter_y = signal.filtfilt(b, a, f(filter_x))
    elif filter_type == 'sg':
        filter_y = sg(f(filter_x), int(sg_wn/31*n)*2+1, filter_order)

    elif filter_type == 'polynomial':
        filter_x1 = np.linspace(min(1/data_pre[0]), max(1/data_pre[0]), 8*n)
        poly_pa = np.polyfit(filter_x1, f(1/filter_x1), filter_order)
        filter_y = np.poly1d(poly_pa)(1/filter_x)

    f_filter = interp1d(filter_x, filter_y, fill_value=0, bounds_error=False)
    return f_filter, lambda x: f(x)-f_filter(x)


f, f2 = my_filter(np.ones((2, 1000)))
print(help(f))
