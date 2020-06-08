import torch
import torch.nn as nn
from torch.autograd import Variable
from convNd import convNd

class Conv3D(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv_layer1 = self._make_conv_layer(1, 32)
        self.conv_layer2 = self._make_conv_layer(32, 64)
        self.conv_layer3 = self._make_conv_layer(64, 124)
        self.conv_layer4 = self._make_conv_layer(124, 256)
        self.conv_layer5 = nn.Conv3d(256, 256, kernel_size=(1, 3, 1), padding=0)

        self.fc5 = nn.Linear(256, 256)
        self.relu = nn.LeakyReLU()
        self.batch0 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(p=0.15)
        self.fc6 = nn.Linear(256, 124)
        self.relu = nn.LeakyReLU()
        self.batch1 = nn.BatchNorm1d(124)

        self.drop = nn.Dropout(p=0.15)
        self.fc7 = nn.Linear(124, 2)

    def _make_conv_layer(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(2, 3, 2), padding=0),
            nn.LeakyReLU(),
            nn.Conv3d(out_c, out_c, kernel_size=(2, 3, 2), padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, x):
        print(x.size())
        x = self.conv_layer1(x)
        print(x.size())
        x = self.conv_layer2(x)
        print(x.size())
        x = self.conv_layer3(x)
        print(x.size())
        x = self.conv_layer4(x)
        print(x.size())
        x = self.conv_layer5(x)
        print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.batch0(x)
        x = self.drop(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.batch1(x)
        x = self.drop(x)
        x = self.fc7(x)

        return x


class Conv3D_light(nn.Module):
    """
    Light Conv3D for 2D data.
    """

    def __init__(self):

        super().__init__()

        self.conv_layer1 = self._make_conv_layer_light(1, 32, (2, 4, 2))
        self.conv_layer2 = self._make_conv_layer_light(32, 64, (2, 4, 2))
        self.conv_layer3 = self._make_conv_layer_light(64, 128, (2, 4, 2))
        self.conv_layer4 = nn.Conv3d(128, 256, kernel_size=(2, 8, 1), padding=0)

        self.fc5 = nn.Linear(256, 256)
        self.relu = nn.LeakyReLU()
        self.batch0 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(p=0.15)
        self.fc6 = nn.Linear(256, 2)

    def _make_conv_layer_light(self, in_c, out_c, k_size):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=k_size, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, x):
        # print(x.size())
        x = self.conv_layer1(x)
        # print(x.size())
        x = self.conv_layer2(x)
        # print(x.size())
        x = self.conv_layer3(x)
        # print(x.size())
        x = self.conv_layer4(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.batch0(x)
        x = self.drop(x)
        x = self.fc6(x)

        return x


# class Conv4D(nn.Module):
#     """
#     Conv4D for 3D data utilizing ConvNd class.
#     """

#     def __init__(self):

#         super().__init__()

#         self.conv_layer1 = self._make_conv_layer_light(1, 32, (2, 4, 2))
#         self.conv_layer2 = self._make_conv_layer_light(32, 64, (2, 4, 2))
#         self.conv_layer3 = self._make_conv_layer_light(64, 128, (2, 4, 2))
#         self.conv_layer4 = nn.Conv3d(128, 256, kernel_size=(2, 8, 1), padding=0)

#         self.fc5 = nn.Linear(256, 256)
#         self.relu = nn.LeakyReLU()
#         self.batch0 = nn.BatchNorm1d(256)
#         self.drop = nn.Dropout(p=0.15)
#         self.fc6 = nn.Linear(256, 2)


# conv = convNd(inChans, outChans, 4, ks, stride, padding, use_bias=True,
#               padding_mode=padding_mode, groups=groups).to(device)

#     def _make_conv_layer_light(self, in_c, out_c, k_size):
#         conv_layer = nn.Sequential(
#             nn.ConvNd(in_c, out_c, 4, kernel_size=k_size, padding=0),
#             nn.LeakyReLU(),
#             nn.MaxPool3d((2, 2, 2)),
#         )
#         return conv_layer

#     def forward(self, x):
#         # print(x.size())
#         x = self.conv_layer1(x)
#         # print(x.size())
#         x = self.conv_layer2(x)
#         # print(x.size())
#         x = self.conv_layer3(x)
#         # print(x.size())
#         x = self.conv_layer4(x)
#         # print(x.size())
#         x = x.view(x.size(0), -1)
#         x = self.fc5(x)
#         x = self.relu(x)
#         x = self.batch0(x)
#         x = self.drop(x)
#         x = self.fc6(x)

#         return x


class Conv4D_light_ch_dim(nn.Module):
    """
    Light Conv3D for 3D data. Here it utilizes Conv3D's channel dimension for the time dimension of a series of 3D Radar data
    Thus, it's actually 3D convolution because kernel slides in 3 different dimension, not 4 different dimensions.
    """

    def __init__(self):

        super().__init__()

        self.conv_layer1 = self._make_conv_layer_light(25, 32, (2, 2, 3))
        self.conv_layer2 = self._make_conv_layer_light(32, 64, (2, 2, 3))
        self.conv_layer3 = self._make_conv_layer_light(64, 128, (2, 2, 3))
        self.conv_layer4 = nn.Conv3d(128, 256, kernel_size=(1, 1, 9), padding=0)

        self.fc5 = nn.Linear(256, 256)
        self.relu = nn.LeakyReLU()
        self.batch0 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(p=0.15)
        self.fc6 = nn.Linear(256, 2)

    def _make_conv_layer_light(self, in_c, out_c, k_size):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=k_size, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, x):
        # print(x.size())
        x = self.conv_layer1(x)
        # print(x.size())
        x = self.conv_layer2(x)
        # print(x.size())
        x = self.conv_layer3(x)
        # print(x.size())
        x = self.conv_layer4(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.batch0(x)
        x = self.drop(x)
        x = self.fc6(x)

        return x


class fc(nn.Module):
    """
    Fully Connected Neural Network for 2D Radar data
    """

    def __init__(self):

        super().__init__()
        self.fc1 = nn.Linear(40850, 100)
        self.batch0 = nn.BatchNorm1d(100)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(100, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        # print(x.size())
        x = self.fc1(x)
        # print(x.size())
        x = self.batch0(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        # print(x.size())
        return x


if __name__ == "__main__":

    input_tensor = torch.autograd.Variable(torch.rand(32, 25, 19, 19, 86))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Conv4D_light_ch_dim()

    output = model(input_tensor)  # model(input_tensor.cuda())
    # print(output.size())
