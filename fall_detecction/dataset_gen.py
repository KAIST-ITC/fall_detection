from imp import load_source
from os.path import join
from sys import platform
import numpy as np
import time
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=100)
import glob
import re


# if platform == 'win32':
#     modulePath = join('C:/', 'Program Files', 'Walabot', 'WalabotSDK',
#                       'python', 'WalabotAPI.py')
# elif platform.startswith('linux'):
#     modulePath = join('/usr', 'share', 'walabot', 'python', 'WalabotAPI.py')
modulePath = 'WalabotAPI.py'


def main():
    wlbt = load_source('WalabotAPI', modulePath)
    wlbt.Init()  # must be called before using WalabotAPI functions

    ############################################################################
    ## Modify only this block ##################################################
    ############################################################################
    THETA_MIN = -45
    THETA_MAX = 45
    THETA_RES = 5
    PHI_MIN = -45
    PHI_MAX = 45
    PHI_RES = 5
    R_MAX = 180
    R_MIN = 10
    R_RES = 2
    THRESHOLD = 35

    THREE_D = True
    MOTION = 'test'
    len_preparation = 5  # move to where the Walabot captures data within 5 secs
    len_data_collect = 6  # Walabot collects data for 7 seconds
    ############################################################################

    # Set the file name to save radar image
    find_str = './' + MOTION + '_data_*.npy'
    find_3d_str = './threeD' + MOTION + '_data_d_*.npy'

    fall_data_list = glob.glob(find_str)
    idx_list = [int(re.findall(r'\d+', elem)[0]) for elem in fall_data_list]
    if(len(idx_list) == 0):  # at the very first
        current_idx = 0
    else:
        current_idx = max(idx_list) + 1
    file_name = MOTION + '_data_' + str(current_idx) + '.npy'
    file_name_3d = 'threeD_' + MOTION + '_data_' + str(current_idx) + '.npy'

    wlbt.Initialize()
    wlbt.ConnectAny()
    print('- Connection Established.')

    wlbt.SetProfile(wlbt.PROF_SENSOR_NARROW)
    wlbt.SetArenaTheta(THETA_MIN, THETA_MAX, THETA_RES)
    wlbt.SetArenaPhi(PHI_MIN, PHI_MAX, PHI_RES)
    wlbt.SetArenaR(R_MIN, R_MAX, R_RES)
    wlbt.SetThreshold(THRESHOLD)
    wlbt.SetDynamicImageFilter(wlbt.FILTER_TYPE_NONE)
    print('- Walabot Configured.')

    wlbt.Start()
    wlbt.StartCalibration()
    print("- Calibration Done.")

    imageList_2d = []
    imageList_3d = []

    print("- Preparing Motion for 5 seconds")
    startTime = time.time()
    while(True):
        endTime = time.time()
        if(endTime - startTime > len_preparation):
            break

    print("- Collecting Data for 6 seconds")
    startTime = time.time()
    print()
    while(True):
        appStatus, calibrationProcess = wlbt.GetStatus()
        wlbt.Trigger()
        if(appStatus == 4):  # 4 is STATUS_SCANNING
            if(THREE_D):
                rasterImage_3d, _, _, sliceDepth, power = wlbt.GetRawImage()
                rasterImage_3d = np.array(rasterImage_3d)
            rasterImage_2d, _, _, sliceDepth, power = wlbt.GetRawImageSlice()
            rasterImage_2d = np.array(rasterImage_2d)
            if(THREE_D):
                imageList_3d = imageList_3d + [rasterImage_3d]
            imageList_2d = imageList_2d + [rasterImage_2d]
            endTime = time.time()
            print(rasterImage_2d)
        print()
        if(endTime - startTime > len_data_collect):
            break

    print("elpased time: ", endTime - startTime)
    print("num of frames: ", len(imageList_2d))
    np.save(file_name, imageList_2d)  # save a series of 2d radar images
    if(THREE_D):
        np.save(file_name_3d, imageList_3d)  # save a series of 3d radar images
    # test = np.load('fall_data_.npy')
    # print('test',test)
    # print(len(imageList_2d[0]))
    # print(len(imageList_2d[0][0]))
    # np.set_printoptions(threshold=np.inf)
    # print(imageList_2d[0])

    wlbt.Stop()
    wlbt.Disconnect()
    wlbt.Clean()


if __name__ == '__main__':
    main()
