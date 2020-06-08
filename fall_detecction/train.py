import glob
import sys
import argparse
import numpy as np
import random
random.seed(42)
import json
from model import Conv3D, Conv3D_light, fc, Conv4D_light_ch_dim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch
from data_loader import RadarFolder
from torchvision.transforms import *

parser = argparse.ArgumentParser(
    description='Fall Detection Training, 3D-CNN, 4D-CNN')
parser.add_argument('--config', '-c', help='json config file path')

args = parser.parse_args()
if len(sys.argv) < 2:
    parser.print_help()
    sys.exit(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load config file
with open(args.config) as data_file:
    config = json.load(data_file)


def train(model, train_loader, criterion, optimizer, epoch, device, print_epoch=10):
    model.train()
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # print(data.shape)
        # print(target)
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        output = model(data)
        # print(output)
        _, preds = torch.max(output.data, 1)

        # print(torch.sum(preds == target))
        # print(torch.eq(preds, target).sum())
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        correct += torch.sum(preds == target)

    acc = correct.item() / len(train_loader.dataset)
    if(epoch % print_epoch == 0):
        print('Train Step: {}  \tLoss: {:.4f}  \tAcc: {:.4f}'.format(
            epoch, loss.item(), 100. * acc))


def test(model, test_loader, criterion, optimizer, epoch, device, print_epoch=10):
    model.eval()

    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, preds = torch.max(output.data, 1)
            loss = criterion(output, target)
            correct += torch.sum(preds == target)

    acc = correct.item() / len(test_loader.dataset)
    if(epoch % print_epoch == 0):
        print('Test set: Accuracy: {}/{} ({:05.2f}%)'.format(
            correct, len(test_loader.dataset), 100. * acc))
    return correct, acc

def main():

    if(config['dimension'] == '2d'):
        model = fc().to(device)
    else:
        assert(config['dimension'] == '3d')
        model = Conv4D_light_ch_dim().to(device)
    # Add Conv3D for 3d data

    transform_train = Compose([
        RandomAffine(degrees=[-10, 10], translate=[0.15, 0.15]),
        RandomVerticalFlip(),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])
    transform_test = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    train_data = RadarFolder(csv_file_input=config['train_csv'],
                             npy_path=config['npy_path'],
                             model=model.__class__.__name__,
                             transform=transform_train)
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    test_data = RadarFolder(csv_file_input=config['test_csv'],
                            npy_path=config['npy_path'],
                            model=model.__class__.__name__,
                            transform=transform_test)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, drop_last=False)

    criterion = nn.CrossEntropyLoss()  # in classification task
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], amsgrad=True)
    test_score_list = []

    for epoch in range(config["epochs"]):
        train(model, train_loader, criterion, optimizer, epoch, device, print_epoch=10)
        sum_score, acc = test(model, test_loader, criterion, optimizer, epoch, device, print_epoch=10)
        test_score_list = test_score_list + [sum_score.item()]

    print(test_score_list)
    torch.save(model.state_dict(), config["save_path"])


if __name__ == '__main__':
    main()
