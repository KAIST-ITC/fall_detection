import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import *
from collections import namedtuple

radarData = namedtuple('radarData', ['id', 'label', 'path'])


class RadarDataset(object):
    def __init__(self, csv_path_input, npy_path):
        self.csv_data = self.read_csv_input(csv_path_input, npy_path)

    def read_csv_input(self, csv_path_input, npy_path):
        csv_data = []
        with open(csv_path_input) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                item = radarData(row[0], row[1], npy_path)
                csv_data.append(item)
        return csv_data


class RadarFolder(Dataset):
    def __init__(self, csv_file_input, npy_path, model, transform=None):
        self.dataset_object = RadarDataset(csv_file_input, npy_path)
        self.csv_data = self.dataset_object.csv_data
        self.npy_path = npy_path  # where radar data(.npy) are stored
        self.transform = transform
        self.model = model

    def __getitem__(self, index):
        item = self.csv_data[index]
        file_path = item.path + "/" + item.label + "_" + item.id + ".npy"
        if(self.model == "Conv4D_light_ch_dim"):
            data = torch.tensor(np.load(file_path)).float()  # format data to tensor, add channel dimension
        else:
            data = torch.tensor(np.load(file_path)).unsqueeze(0).float()  # format data to tensor, add channel dimension
        if(item.label == 'fall'):
            target_idx = 1
        else:
            target_idx = 0
        return(data, target_idx)

    def __len__(self):
        return len(self.csv_data)


if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
        # Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225])
    ])
    loader = RadarFolder(csv_file_input="./annotations/train.csv",
                         npy_path="./data_ready/2d",
                         model="test",
                         transform=transform)
    data_item, target_idx = loader[0]
    print(data_item, target_idx)
    print(data_item.shape, target_idx)
    print(loader)
    # save_images_for_debug("input_images", data_item.unsqueeze(0))
