import torch
import torch.nn as nn

import numpy as np

"""
This architecture corresponds to the Fully Convolutional Net of Fawaz et. al.
the original code (using Tensorflow / Keras) can be found at:
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py
"""


class Fcn(nn.Module):
    """
    complete FCN model
    """

    def __init__(self, siglen, n_targets):
        super(Fcn, self).__init__()

        # gpu or cpu
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        print(f'device: {dev}')

        self.dev = torch.device(dev)

        self.siglen = siglen
        self.n_targets = n_targets

        # convolutional layers, all with padding = same
        self.pad1 = nn.ConstantPad1d(padding=(4, 3), value=0).to(self.dev)
        self.conv1 = nn.Conv1d(1, 128, kernel_size=8).to(self.dev)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2).to(self.dev)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1).to(self.dev)

        # batch norm
        self.bn1 = nn.BatchNorm1d(num_features=128).to(self.dev)
        self.bn2 = nn.BatchNorm1d(num_features=256).to(self.dev)
        self.bn3 = nn.BatchNorm1d(num_features=128).to(self.dev)

        # activation
        self.relu = nn.ReLU(inplace=True).to(self.dev)

        # global average poolig
        self.gap = nn.AvgPool1d(kernel_size=self.siglen).to(self.dev)

        # classification layer
        self.fc = nn.Linear(1 * 128, self.n_targets).to(self.dev)

    def forward(self, x):

        x = self.pad1(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # output of last conv-layer - used for CAM
        if x.size(0) == 1:
            self.A = x.detach().clone().squeeze()

        x = self.gap(x)
        x = x.view(x.size(0), 1 * 128)
        x = self.fc(x)

        return x

    def get_cam(self, signal):

        # activate evaluation mode
        self.eval()

        # forward pass
        out = self.forward(signal)

        # get weights of last layer
        wm = self.fc.weight.data

        # compute cam
        cam = torch.matmul(wm, self.A)

        # prediction of forward pass
        pred = np.argmax(out.squeeze().detach().numpy())

        # set back to train mode
        self.train()

        return cam.numpy(), pred
