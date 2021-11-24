# -*- coding: utf-8 -*-
# @Time: 2021/11/15 15:34
# @Author: HS


import numpy as np
import matplotlib.pyplot as plt


kb = 1.380649e-23
e = 1.6e-19
h_bar = 1.05457e-34
me = 9.11e-31


def func(B, frequency, m, td=1, t=2, g=0, delta=0):
    alpha = 2 * np.pi * np.pi * kb * me / (h_bar * e)
    lam = alpha * t * m / B
    lamd = alpha * td * m / B
    RT = lam / np.sinh(lam)
    RD = np.exp(-lamd)
    RS = np.cos(np.pi * g * m / 2)
    return -B**(-0.5) * RT * RD * RS * np.sin(2 * np.pi * (frequency / B - delta))


def make_data(B, f_num=1, poly_order=3, rdio=[1, 2], f_range=[[15, 45]],
              m_range=[0.4, 0.8], t_range=[1, 2], td_range=[2, 4]):
    frequency, m, td, t = np.random.rand(4, f_num)
    poly_pa = np.random.rand(poly_order)-0.3
    rdio = np.random.randint(*rdio)

    gv = lambda rang, rand: rang[0]+(rang[1]-rang[0])*rand
    osc = 0

    for num in range(f_num):
        f0 = gv(f_range[num], frequency[num])
        m0 = gv(m_range, m[num])
        td0 = gv(td_range, td[num])
        t0 = gv(t_range, t[num])
        osc += func(B, f0, m0, td0, t0)
    max_osc = max(osc)

    poly = np.poly1d([*poly_pa, 0])(B/max(B))
    nom1 = max(abs(poly))
    poly *= rdio * max_osc / nom1
    poly_pa *= rdio * max_osc / nom1

    y_smooth = osc + poly
    error = np.random.randn(len(B))*np.mean(y_smooth)/25

    y = y_smooth + error
    fac = max(abs(y))

    return y/fac, osc/fac, poly/fac, poly_pa/fac


if __name__ == '__main__':
    x = np.linspace(0.1, 9, 512)
    total, osc, poly, pa = make_data(x, f_num=2, poly_order=4,
                                     rdio=[1, 21], f_range=[[15, 45], [50, 150]])
    # pa = np.polyfit(x, total, 5)
    fit = np.poly1d([*pa, 0])

    plt.figure()
    plt.plot(x, total)
    plt.plot(x, osc)
    # plt.plot(x, total-fit(x))
    plt.plot(x, poly, c='r')
    plt.plot(x, fit(x/9), c='k')
    plt.show()

    # 制造5000个训练数据
    features = []
    labels = []
    for i in range(5000):
        feature, a1, a2, label = make_data(x, f_num=2, poly_order=6, rdio=[1, 21],
                                           f_range=[[15, 45],
                                                    [50, 150]])
        features.append(feature)
        labels.append(label)

    np.save('../../Train/train_features.npy', features)
    np.save('../../Train/train_labels.npy', labels)

    # 制造1000个测试数据
    features_test = []
    labels_test = []
    for i in range(1000):
        feature, a1, a2, label = make_data(x, f_num=2, poly_order=6, rdio=[1, 26],
                                           f_range=[[5, 60],
                                                    [100, 250]]
                                           )
        features.append(feature)
        labels.append(label)

    print(labels[0])
    np.save('../../Train/test_features.npy', features)
    np.save('../../Train/test_labels.npy', labels)


