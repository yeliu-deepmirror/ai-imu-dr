import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from termcolor import cprint
import copy

max_grad_norm = 1e1
criterion_mse = torch.nn.MSELoss(reduction="sum")
criterion_l1 = torch.nn.L1Loss(reduction="none")
imu_mean = np.array([0, 0, 0, 0, 0, 9.81], dtype=np.float64)
imu_std = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], dtype=np.float64)


def set_mes_net_optimizer(mes_net):
    # lr_mesnet = {'cov_net': 1e-4,
    #     'cov_lin': 1e-4,
    #     }
    # weight_decay_dmmesnet = {'cov_net': 1e-4,
    #     'cov_lin': 1e-4,
    #     }
    # param_list = []
    # for key, value in lr_mesnet.items():
    #     param_list.append({'params': getattr(mes_net, key).parameters(),
    #                        'lr': value,
    #                        'weight_decay': weight_decay_dmmesnet[key]
    #                        })
    # optimizer = torch.optim.Adam(param_list)

    optimizer = torch.optim.SGD(mes_net.parameters(), lr=0.01, momentum=0.9)
    return optimizer



def train_mes_net_loop(args, trainning_data, epoch, meanet, optimizer):
    optimizer.zero_grad()

    # compute loss
    output = meanet.forward(trainning_data["input"])

    # compute the covariance of the output
    std_output = torch.std(output, dim=0).to('cpu').detach().numpy()
    # print(std_output)

    loss_train_tensor = criterion_mse(output, trainning_data["output"]) * trainning_data["weights"]
    loss_train = loss_train_tensor.sum()
    # cprint("  - loss_train: {:.5f}".format(loss_train))

    if loss_train == 0:
        return
    loss_train.backward()  # loss_train.cuda().backward()

    g_norm = torch.nn.utils.clip_grad_norm_(meanet.parameters(), max_grad_norm).cpu()
    optimizer.step()
    optimizer.zero_grad()
    # cprint("  - gradient norm: {:.5f}".format(g_norm))
    # cprint('  Final Loss of Epoch {:2d} \tLoss: {:.5f}'.format(epoch, loss_train))
    return loss_train, g_norm, std_output
