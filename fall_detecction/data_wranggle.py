import numpy as np
import glob
import math
import re
from pathlib import Path
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=100)


def cut_data(data, len_std):
    """ To adjust the data len to len_std, cut some frames at the head
        and cut some frames at the tail
    """
    extra_len = len(data) - len_std
    head_len = math.ceil(extra_len / 2)
    tail_len = math.floor(extra_len / 2)
    # print(head_len)
    # print(tail_len)
    return data[head_len: len(data) - tail_len]  # cut


def pad_data(data, len_std):
    """ To adjust the data len to len_std, pad the first frame to the head
        and pad the last frame to the tail
    """
    extra_len = len_std - len(data)
    head_len = math.ceil(extra_len / 2)
    tail_len = math.floor(extra_len / 2)
    # print(head_len)
    # print(tail_len)
    padded_data = list([data[0]])  # np array to list to utilize * operation
    head_add_frames = np.array(padded_data * head_len)
    tail_add_frames = np.array(padded_data * tail_len)
    # print(head_add_frames)
    # print(head_add_frames.shape)
    # print(tail_add_frames.shape)
    if(head_len != 0 and tail_len != 0):
        data = np.vstack([head_add_frames, data, tail_add_frames])  # concat
    elif(tail_len == 0):
        data = np.vstack([head_add_frames, data])  # concat
    elif(head_len == 0):
        data = np.vstack([data, tail_add_frames])  # concat
    return data


def mkdir():
    """ Make directories for new data
    """
    Path("./data_ready/2d/").mkdir(parents=True, exist_ok=True)
    Path("./data_ready/3d/").mkdir(parents=True, exist_ok=True)


def main():
    mkdir()
    len_std = 25
    data_str = './data/*/*/*.npy'
    data_dir_list = glob.glob(data_str)

    # data_dic = {directory: np.load(directory) for directory in data_dir_list}  # dictionary, key==directory, value==data
    tup_list = [(directory, np.load(directory)) for directory in data_dir_list]  # list of tuples, [(directory, data)]
    tup_ready_list = []
    for tup in tup_list:
        (directory, data) = tup
        # print(len(data))
        if(len(data) > len_std):
            data_ready = cut_data(data, len_std)
        elif(len(data) < len_std):
            data_ready = pad_data(data, len_std)
        else:
            data_ready = data
        assert(len(data_ready) == len_std)
        directory = directory.replace('\\', '/')
        dir_split = re.split('/', directory)
        dim = dir_split[2]  # 2d or 3d
        action = dir_split[3]  # fall, squat, stand, etc
        index = re.findall(r'\d+', dir_split[4])[0]  # 0,1,2,...
        file_name = dir_split[4]  # fall_a_data_0.npy
        # print(index)
        if('fall_b' in file_name):
            index = int(index) + 50
        if('stand_b' in file_name):
            index = int(index) + 25
        if('stand_c' in file_name):
            index = int(index) + 50
        if('stand_d' in file_name):
            index = int(index) + 75
        if('stand_e' in file_name):
            index = int(index) + 100
        if('walk_b' in file_name):
            index = int(index) + 25
        if('walk_c' in file_name):
            index = int(index) + 50
        if('walk_d' in file_name):
            index = int(index) + 75
        if('walk_stop_b' in file_name):
            index = int(index) + 25
        directory_ready = './data_ready/' + dim + '/' + action + '_' + str(index) + '.npy'
        # print(directory_ready)
        tup_ready_list = tup_ready_list + [(directory_ready, data_ready)]

    for tup_ready in tup_ready_list:
        (directory_ready, data_ready) = tup_ready
        np.save(directory_ready, data_ready)

    # test = np.array([[0, 0, 0]])
    # for i in range(1, 21):
    #     # print(test)
    #     add_array = np.array([i for j in range(3)])
    #     test = np.vstack([test, add_array])
    #     # print(add_array)
    #     # test = np.append(test, add_array)
    # print(len(test))
    # new_test = pad_data(test, 25)
    # # new_test = cut_data(test, 25)
    # print(new_test)
    # assert(len(new_test) == 25)


if __name__ == '__main__':
    main()

    # print("test)
    # print("dont run this")

# import numpy as np
# x = np.array([[1, 2], [3, 4]])
# print(x.shape)
# arr = x
# arr = np.repeat(arr[np.newaxis, :, :], 3, axis=0)
# print(arr)
# print(arr.shape)

# import numpy as np
# test = np.load('./data/2d/fall/fall_a_data_0.npy')
# print(test)
# print(len(test))
# print(len(test[0]))
# print(len(test[0][0]))

# import glob
# # data_str = './data/2d/fall/*.npy'
# data_str = '*'
# data_dir_list = glob.glob(data_str)
# print(data_dir_list)


# import re

# test_str = './data/2d/fall/fall_a_data_0.npy'

# print(re.split('/', test_str))
# test_str = test_str.replace('./data/2d/fall/fall_a_data', './data_ready/2d/fall')
# print(test_str)
