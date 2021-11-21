# -*- coding: utf-8 -*-
# @Time: 2021/11/4 17:22
# @Author: HS

import sys
import Window
from PyQt5.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    win = Window.Win()
    win.setWindowTitle('FFT Python')
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

