# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:32:21 2020

@author: HS
"""
from numpy import pi
kb = 1.380649e-23
e = 1.6e-19
h_bar = 1.05457e-34
me = 9.11e-31


def get_phy(m,F, TD = None):
    kf = (2*e*F/h_bar)**0.5
    sf = 2*pi*e*F/h_bar   
    vf = h_bar*kf/(m*me)
    Ef = m*me*vf**2
    ns = (1/3*pi*pi)*(2*e*F/h_bar)**(3/2)
    if TD == None:
        pa = {'F': F, 'n': ns, 'm': m, 'kf': kf/1e8, 'sf': sf/1e8,
              'vf': vf/1e5, 'Ef': Ef/e, 't': None, 'TD': None, 'l': None, 'miu': None}
    else:
        tar = h_bar/(2*pi*kb*TD)
        l = tar*vf
        miu = e*tar/(m*me)
        pa = {'F': F, 'n': ns, 'm': m, 'kf': kf / 1e8, 'sf': sf / 1e8,
              'vf': vf / 1e5, 'Ef': Ef / e, 't': tar*1e13, 'TD': l*1e9,
              'l': l*1e9, 'miu': miu*1e4}
    return pa