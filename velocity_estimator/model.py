import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from termcolor import cprint


class DmVelNet(torch.nn.Module):
    def __init__(self, device, block_size = 50, dropout = 0.5, num_feature = 32):
        super(DmVelNet, self).__init__()
        self.device = device
        self.dropout = dropout
        self.block_size = block_size
        num_linear_feature = (block_size - 4 - 12) * num_feature
        num_linear_feature_half = int(num_linear_feature * 0.5)

        self.cov_net = torch.nn.Sequential(torch.nn.Conv1d(9, num_feature, 5),
                       torch.nn.ReLU(),
                       torch.nn.Dropout(p=self.dropout),
                       torch.nn.Conv1d(num_feature, num_feature, 5, dilation=3),
                       torch.nn.ReLU(),
                       torch.nn.Dropout(p=self.dropout),
                       ).double()
        "CNN for measurement covariance"
        self.cov_lin = torch.nn.Sequential(torch.nn.Linear(num_linear_feature, num_linear_feature_half),
                                           torch.nn.Linear(num_linear_feature_half, 3),
                                           ).double()

    def cov_net_forward(self, x):
        print(x.shape)
        for layer in self.cov_net:
            x = layer(x)
            print(layer, x.shape)
        return x

    def forward(self, input):
        output = self.cov_net(input)
        output = torch.flatten(output, 1)
        output = self.cov_lin(output)
        return output

    def load(self, model_path):
        cprint("  load IEKF from " + model_path, 'green')
        if os.path.isfile(model_path):
            mondict = torch.load(model_path)
            self.load_state_dict(mondict)
            cprint("  DmVelNet nets loaded", 'green')
        else:
            cprint("  DmVelNet nets NOT loaded", 'yellow')
