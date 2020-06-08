# No classifier, just GUI visualization
from mpl_toolkits.mplot3d import Axes3D

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

# Walabot stuff
from imp import load_source
from os.path import join
from sys import platform

if platform == 'win32':
    modulePath = join('C:/', 'Program Files', 'Walabot', 'WalabotSDK', 'python', 'WalabotAPI.py')
elif platform.startswith('linux'):
    modulePath = join('/usr', 'share', 'walabot', 'python', 'WalabotAPI.py')

# Below are Walabot configuration
THETA_MIN_VAL = -50
THETA_MAX_VAL = 50
THETA_RES_VAL = 10

PHI_MIN_VAL = -50
PHI_MAX_VAL = 50
PHI_RES_VAL = 5

R_MIN_VAL = 10
R_MAX_VAL = 200
R_RES_VAL = 3
THRESHOLD = 35
# NUM_POSES_COUNT = 500

NUM_R_STEPS = int(((R_MAX_VAL - R_MIN_VAL) / R_RES_VAL) + 1)
NUM_PHI_STEPS = int(((PHI_MAX_VAL - PHI_MIN_VAL) / PHI_RES_VAL) + 1)
NUM_THETA_STEPS = int(((THETA_MAX_VAL - THETA_MIN_VAL) / THETA_RES_VAL) + 1)
NUM_TOTAL_STEPS = int(NUM_R_STEPS) * int(NUM_PHI_STEPS) * int(NUM_THETA_STEPS)


class Ui_MainWindow(object):

    def setupWalabot(self):
        self.wlbt = load_source('WalabotAPI', modulePath)
        self.wlbt.Init()  # must be called before using WalabotAPI functions
        self.wlbt.Initialize()
        self.wlbt.ConnectAny()
        print('- Connection Established.')

        self.wlbt.SetProfile(self.wlbt.PROF_SENSOR)
        self.wlbt.SetArenaTheta(THETA_MIN_VAL, THETA_MAX_VAL, THETA_RES_VAL)
        self.wlbt.SetArenaPhi(PHI_MIN_VAL, PHI_MAX_VAL, PHI_RES_VAL)
        self.wlbt.SetArenaR(R_MIN_VAL, R_MAX_VAL, R_RES_VAL)
        self.wlbt.SetThreshold(THRESHOLD)
        self.wlbt.SetDynamicImageFilter(self.wlbt.FILTER_TYPE_NONE)
        print('- Walabot Configured.')

        self.wlbt.Start()
        self.wlbt.StartCalibration()
        print("- Calibration Done.")

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 660)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # set the background color of MainWindow as white
        MainWindow.setStyleSheet("QMainWindow {background: 'white';}")
        MainWindow.setCentralWidget(self.centralwidget)
        MainWindow.setWindowTitle("Pose Classifier")

        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(20, 520, 280, 140))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableWidget.sizePolicy().hasHeightForWidth())
        self.tableWidget.setFrameShape(QtWidgets.QFrame.NoFrame)

        self.tableWidget.setFocusPolicy(QtCore.Qt.NoFocus)
        self.tableWidget.setShowGrid(False)
        self.tableWidget.setSizePolicy(sizePolicy)
        self.tableWidget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tableWidget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.tableWidget.setRowCount(4)
        self.tableWidget.setColumnCount(4)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.horizontalHeader().setVisible(False)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(50)
        self.tableWidget.horizontalHeader().setMinimumSectionSize(30)
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.verticalHeader().setDefaultSectionSize(30)

        self.tableWidget.setItem(0, 0, QtWidgets.QTableWidgetItem())
        self.tableWidget.setItem(0, 1, QtWidgets.QTableWidgetItem("Min"))
        self.tableWidget.setItem(0, 2, QtWidgets.QTableWidgetItem("Max"))
        self.tableWidget.setItem(0, 3, QtWidgets.QTableWidgetItem("Res"))

        self.tableWidget.setItem(1, 0, QtWidgets.QTableWidgetItem("R"))
        self.tableWidget.setItem(1, 1, QtWidgets.QTableWidgetItem(str(R_MIN_VAL)))
        self.tableWidget.setItem(1, 2, QtWidgets.QTableWidgetItem(str(R_MAX_VAL)))
        self.tableWidget.setItem(1, 3, QtWidgets.QTableWidgetItem(str(R_RES_VAL)))

        self.tableWidget.setItem(2, 0, QtWidgets.QTableWidgetItem("Theta"))
        self.tableWidget.setItem(2, 1, QtWidgets.QTableWidgetItem(str(THETA_MIN_VAL)))
        self.tableWidget.setItem(2, 2, QtWidgets.QTableWidgetItem(str(THETA_MAX_VAL)))
        self.tableWidget.setItem(2, 3, QtWidgets.QTableWidgetItem(str(THETA_RES_VAL)))

        self.tableWidget.setItem(3, 0, QtWidgets.QTableWidgetItem("Phi"))
        self.tableWidget.setItem(3, 1, QtWidgets.QTableWidgetItem(str(PHI_MIN_VAL)))
        self.tableWidget.setItem(3, 2, QtWidgets.QTableWidgetItem(str(PHI_MAX_VAL)))
        self.tableWidget.setItem(3, 3, QtWidgets.QTableWidgetItem(str(PHI_RES_VAL)))

        # Radar image
        self.canvas = PlotCanvas(MainWindow, width=10, height=5)
        self.canvas.move(0, 0)

        # Radar image update
        self.qTimer = QTimer()
        self.qTimer.setInterval(10)  # 1000 ms = 1 s
        # self.qTimer.timeout.connect(self.autoStart)
        self.qTimer.timeout.connect(self.updateCanvas)
        self.qTimer.start()

    def updateCanvas(self):

        self.wlbt.Trigger()

        rasterImage_2d, _, _, sliceDepth, power = self.wlbt.GetRawImageSlice()
        # rasterImage = np.array(rasterImage)

        rasterImage_3d, X, Y, Z, power_3d = self.wlbt.GetRawImage()
        rasterImage_2d = np.array(rasterImage_2d)
        rasterImage_3d = np.array(rasterImage_3d)

        # if(sum(rasterImage_2d.ravel()) > 0):

        rasterImage_3d = rasterImage_3d.astype(np.float32).reshape(-1, NUM_TOTAL_STEPS)  # flatten
        rasterImage_3d = rasterImage_3d[0]

        self.canvas.plot(rasterImage_2d, rasterImage_3d)


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=10, height=5, dpi=100):

        fig = plt.figure(figsize=(width, height), dpi=dpi)

        # Adjust graph margin
        # fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
        # self.axes = fig.add_subplot(111)
        #
        # FigureCanvas.__init__(self, fig)
        # self.setParent(parent)
        #
        # FigureCanvas.setSizePolicy(self,
        #                            QtWidgets.QSizePolicy.Expanding,
        #                            QtWidgets.QSizePolicy.Expanding)
        # FigureCanvas.updateGeometry(self)

        # fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
        self.axes_2d = fig.add_subplot(1, 2, 1)
        self.axes_3d = fig.add_subplot(1, 2, 2, projection='3d')

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self, rasterImage_2d, rasterImage_3d):

        # verticalSide = np.linspace(-1, 1, NUM_R_STEPS)
        # horizontalSide = np.linspace(-1, 1, NUM_PHI_STEPS)
        # X, Y = np.meshgrid(horizontalSide, verticalSide)

        # 2D stuff
        PHI_side = np.linspace(-1, 1, NUM_PHI_STEPS)
        R_side = np.linspace(-1, 1, NUM_R_STEPS)
        PHI, R = np.meshgrid(PHI_side, R_side)
        vals = rasterImage_2d

        self.axes_2d.pcolormesh(PHI, R, vals)
        self.axes_2d.set_xlabel('PHI')
        self.axes_2d.set_ylabel('R')

        # R_side = np.linspace(-1, 1, NUM_R_STEPS)
        # PHI_side = np.linspace(-1, 1, NUM_PHI_STEPS)
        #
        # R, PHI = np.meshgrid(R_side,PHI_side)
        #
        # # print(rasterImage_2d)
        # print(R)
        # # vals[vals == 0] = np.nan # zeroes are not drawn
        # # self.axes_2d.clear()
        # self.axes_2d.pcolormesh(R, PHI, vals)
        # # self.axes_2d.set_xlim(-1,1)
        # # self.axes_2d.set_ylim(-1,1)

        # 3D stuff

        R_side = np.linspace(-1, 1, NUM_R_STEPS)
        PHI_side = np.linspace(-1, 1, NUM_PHI_STEPS)
        THETA_side = np.linspace(-1, 1, NUM_THETA_STEPS)
        R, PHI, THETA = np.meshgrid(R_side, PHI_side, THETA_side)

        vals = rasterImage_3d
        # print(rasterImage_3d)
        vals[vals == 0] = np.nan  # zeroes are not drawn
        self.axes_3d.clear()
        self.axes_3d.scatter(R, PHI, THETA, c=vals)
        self.axes_3d.set_xlim(-1, 1)
        self.axes_3d.set_ylim(-1, 1)
        self.axes_3d.set_zlim(-1, 1)
        self.axes_3d.set_xlabel('R')
        self.axes_3d.set_ylabel('PHI')
        self.axes_3d.set_zlabel('THETA')

        self.draw()


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupWalabot()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
