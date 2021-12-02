# -*- coding: utf-8 -*-
# @Time: 2021/11/21 15:45
# @Author: HS

from MakeData import make_data, np, plt
import pandas as pd


def main():
    data = {}
    x = np.linspace(0.1, 9, 512)
    for t in [2, 5, 10, 15, 20, 30]:
        data[str(t)+'x'] = x
        data[t], *a = make_data(B=x, f_num=2, poly_order=3, rdio=[1, 2], f_range=[[30, 30], [70, 70]],
                                m_range=[1, 1], td_range=[2, 2], t_range=[t, t])
    pd.DataFrame(data).to_csv('../../../DATA/test.csv', index=None)


if __name__ == '__main__':
    main()
