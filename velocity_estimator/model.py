import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from termcolor import cprint

class DmVelNet(torch.nn.Module):
    def __init__(self, device, dropout = 0.5, num_feature = 16):
        super(DmVelNet, self).__init__()
        self.device = device
        self.dropout = dropout

        self.cov_net = torch.nn.Sequential(torch.nn.Conv1d(6, num_feature, 5),
                       torch.nn.ReplicationPad1d(4),
                       torch.nn.ReLU(),
                       torch.nn.Dropout(p=self.dropout),
                       torch.nn.Conv1d(num_feature, num_feature, 5, dilation=3),
                       torch.nn.ReplicationPad1d(4),
                       torch.nn.ReLU(),
                       torch.nn.Dropout(p=self.dropout),
                       ).double()
        "CNN for measurement covariance"
        self.cov_lin = torch.nn.Sequential(torch.nn.Linear(num_feature, 3),
                                           torch.nn.Tanh(),
                                           ).double()

        # rescale the initial values, to make the network able to converge
        self.cov_lin[0].bias.data[:] /= 100
        self.cov_lin[0].weight.data[:] /= 100

    def cov_net_forward(self, x):
        for layer in self.cov_net:
            x = layer(x)
            print(layer, x)
        return x

    def forward(self, input):
        output = self.cov_net(input)
        output = output.transpose(0, 2).squeeze()
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
