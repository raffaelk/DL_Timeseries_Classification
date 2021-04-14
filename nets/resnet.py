import torch
import torch.nn as nn

import numpy as np

"""
This architecture corresponds to the ResNet of Fawaz et. al.
the original code (using Tensorflow / Keras) can be found at:
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py
"""


class RESBlock(nn.Module):
    """
    single block of a resnet - only used as part of a complete 
    resnet architecture
    """

    def __init__(self, n_filters, in_channels):
        super(RESBlock, self).__init__()

        # use a gpu if available
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        self.dev = torch.device(dev)

        # convolutional layers, all with padding = same
        self.pad1 = nn.ConstantPad1d(padding=(4, 3), value=0).to(self.dev)
        self.conv1 = nn.Conv1d(in_channels, n_filters, kernel_size=8).to(self.dev)
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=5, padding=2).to(self.dev)
        self.conv3 = nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1).to(self.dev)

        self.convsc = nn.Conv1d(in_channels, n_filters, kernel_size=1).to(self.dev)

        # batch norm
        self.bn1 = nn.BatchNorm1d(num_features=n_filters).to(self.dev)
        self.bn2 = nn.BatchNorm1d(num_features=n_filters).to(self.dev)
        self.bn3 = nn.BatchNorm1d(num_features=n_filters).to(self.dev)

        self.bnsc = nn.BatchNorm1d(num_features=n_filters).to(self.dev)

        # activation
        self.relu = nn.ReLU(inplace=True).to(self.dev)

    def forward(self, x):

        # conv layers
        cv = self.pad1(x)
        cv = self.relu(self.bn1(self.conv1(cv)))
        cv = self.relu(self.bn2(self.conv2(cv)))
        cv = self.bn3(self.conv3(cv))

        # shortcut
        sc = self.bnsc(self.convsc(x))

        # add conv layers to skip connection
        out = self.relu((cv + sc))

        return out


class RESNET(nn.Module):
    """
    complete ResNet model
    """

    def __init__(self, siglen, n_targets):
        super(RESNET, self).__init__()
        """
        Args:
            siglen (int): number of timesteps per signal
            n_targets (int): number of target classes
        """
        # use a gpu if available
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        print(f'Running on device: {dev}')

        self.dev = torch.device(dev)

        # initialize
        self.siglen = siglen
        self.n_targets = n_targets

        # stack resnet blocks
        self.n_feature_maps = 64
        self.block1 = RESBlock(self.n_feature_maps, 1)
        self.block2 = RESBlock(self.n_feature_maps*2, self.n_feature_maps)
        self.block3 = RESBlock(self.n_feature_maps*2, self.n_feature_maps*2)

        # global average poolig
        self.gap = nn.AvgPool1d(kernel_size=self.siglen).to(self.dev)

        # classification layer
        self.fc = nn.Linear(1 * (self.n_feature_maps*2), self.n_targets).to(self.dev)

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # output of last conv-layer - used for CAM
        if x.size(0) == 1:
            self.A = x.detach().clone().squeeze()

        x = self.gap(x)
        x = x.view(x.size(0), 1 * (self.n_feature_maps*2))
        x = self.fc(x)

        return x

    def get_cam(self, signal):
        """ Calculates the CAM map for one signal
        
        Args:
            signal (torch.Tensor): Signal to calculate the CAM for.
            
        Returns:
            A tuple with the CAM map as numpy array and the predicted class
            of the input signal.
        """
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
