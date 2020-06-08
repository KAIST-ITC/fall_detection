from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

# Below are Walabot and tensorflow stuff
from imp import load_source
from os.path import join
import tensorflow as tf

# alexa integration stuff
import globalVariable

# Below are Walabot configuration
THETA_MIN_VAL = -45
THETA_MAX_VAL = 45
THETA_RES_VAL = 5
PHI_MIN_VAL = -45
PHI_MAX_VAL = 45
PHI_RES_VAL = 5
R_MIN_VAL = 10
R_MAX_VAL = 150
R_RES_VAL = 2
THRESHOLD = 35
# NUM_POSES_COUNT = 500

# Below are NN graph configuration
NUM_R_STEPS = ((R_MAX_VAL - R_MIN_VAL) / R_RES_VAL) + 1
NUM_THETA_STEPS = ((THETA_MAX_VAL - THETA_MIN_VAL) / THETA_RES_VAL) + 1
N_INPUTS = NUM_R_STEPS * NUM_THETA_STEPS
N_HIDDEN1 = 300
N_HIDDEN2 = 100
N_OUTPUTS = 2


# For deterministic character
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


class Ui_MainWindow(object):

    def setupClassifier(self, MainWindow):
        reset_graph()

        self.X = tf.placeholder(tf.float32, shape=(None, N_INPUTS), name="X")
        self.y = tf.placeholder(tf.int32, shape=(None), name="y")

        self.hidden1 = tf.layers.dense(self.X, N_HIDDEN1, name="hidden1", activation=tf.nn.relu)
        self.hidden2 = tf.layers.dense(self.hidden1, N_HIDDEN2, name="hidden2", activation=tf.nn.relu)
        self.logits = tf.layers.dense(self.hidden2, N_OUTPUTS, name="outputs")

        self.sess = tf.Session()
        self.setupWalabot()

        saver = tf.train.Saver()
        saver.restore(self.sess, "./my_model_final.ckpt")


    def setupWalabot(self):
        modulePath = join('WalabotAPI.py')
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
        MainWindow.resize(500, 660)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # set the background color of MainWindow as white
        MainWindow.setStyleSheet("QMainWindow {background: 'white';}")
        MainWindow.setCentralWidget(self.centralwidget)
        MainWindow.setWindowTitle("Pose Classifier")

        self.classificationResultLbl = QtWidgets.QLabel(self.centralwidget)
        self.classificationResultLbl.setGeometry(QtCore.QRect(310, 550, 140, 40))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(17)
        font.setBold(True)
        font.setWeight(75)
        self.classificationResultLbl.setFont(font)
        self.classificationResultLbl.setObjectName("classificationResultLbl")

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

        self.tableWidget.setItem(0,0, QtWidgets.QTableWidgetItem())
        self.tableWidget.setItem(0,1, QtWidgets.QTableWidgetItem("Min"))
        self.tableWidget.setItem(0,2, QtWidgets.QTableWidgetItem("Max"))
        self.tableWidget.setItem(0,3, QtWidgets.QTableWidgetItem("Res"))

        self.tableWidget.setItem(1,0, QtWidgets.QTableWidgetItem("R"))
        self.tableWidget.setItem(1,1, QtWidgets.QTableWidgetItem("10"))
        self.tableWidget.setItem(1,2, QtWidgets.QTableWidgetItem("150"))
        self.tableWidget.setItem(1,3, QtWidgets.QTableWidgetItem("2"))

        self.tableWidget.setItem(2,0, QtWidgets.QTableWidgetItem("Theta"))
        self.tableWidget.setItem(2,1, QtWidgets.QTableWidgetItem("-45"))
        self.tableWidget.setItem(2,2, QtWidgets.QTableWidgetItem("45"))
        self.tableWidget.setItem(2,3, QtWidgets.QTableWidgetItem("5"))

        self.tableWidget.setItem(3,0, QtWidgets.QTableWidgetItem("Phi"))
        self.tableWidget.setItem(3,1, QtWidgets.QTableWidgetItem("-45"))
        self.tableWidget.setItem(3,2, QtWidgets.QTableWidgetItem("45"))
        self.tableWidget.setItem(3,3, QtWidgets.QTableWidgetItem("5"))

        # Radar image
        self.canvas = PlotCanvas(MainWindow, width=5, height=5)
        self.canvas.move(0, 0)

        self.classificationResult = globalVariable.pose
        # This text label will show the classification result
        self.classificationResultLbl.setText(self.classificationResult)

        # Radar image update
        self.counter = 0
        self.qTimer = QTimer()
        self.qTimer.setInterval(100)  # 1000 ms = 1 s
        # self.qTimer.timeout.connect(self.autoStart)
        self.qTimer.timeout.connect(self.getClassificationResult)
        self.qTimer.start()


    def getClassificationResult(self):

        self.wlbt.Trigger()
        rasterImage, _, _, sliceDepth, power = self.wlbt.GetRawImageSlice()
        rasterImage = np.array(rasterImage)

        # Added for radar image animation

        if(sum(rasterImage.ravel()) > 0):
            # print(rasterImage)
            self.canvas.plot(rasterImage)

            rasterImage = rasterImage.astype(np.float32).reshape(-1, int(NUM_R_STEPS) * int(NUM_THETA_STEPS)) / 255.0  # flatten
            rasterImage = rasterImage[0]
            Z = self.sess.run(self.logits, feed_dict={self.X: [rasterImage]})
            y_pred = np.argmax(Z, axis=1)
            if(y_pred == 0):
                globalVariable.pose = "sitting"
                self.classificationResult = globalVariable.pose
            elif(y_pred == 1):
                globalVariable.pose = "standing"
                self.classificationResult = globalVariable.pose

            self.counter = self.counter + 1
            self.classificationResultLbl.setText(self.classificationResult)

        # if(self.counter == NUM_POSES_COUNT):
        #     self.wlbt.Stop()
        #     self.wlbt.Disconnect()
        #     self.wlbt.Clean()
        #     self.sess.close()


class PlotCanvas(FigureCanvas):


    def __init__(self, parent=None, width=5, height=5, dpi=100):

        fig = plt.figure(figsize=(width, height), dpi=dpi)

        # Adjust graph margin
        fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


    def plot(self, rasterImage):

        verticalSide = np.linspace(-1, 1, 71)
        horizontalSide = np.linspace(-1, 1, 19)
        X, Y = np.meshgrid(horizontalSide, verticalSide)

        Z = rasterImage

        self.axes.pcolormesh(X, Y, Z)
        self.axes.axis('off')
        self.draw()


# Main function that runs Walabot classifier and creates GUI
def startApp():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupClassifier(MainWindow)
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
