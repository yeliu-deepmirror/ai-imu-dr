import sys
sys.path.insert(0, '..')

import os
import shutil
import numpy as np
import glob
import time
import datetime
import pickle
import torch
import matplotlib.pyplot as plt
from termcolor import cprint
from model import DmVelNet
from train_torch_filter import set_mes_net_optimizer, train_mes_net_loop, imu_mean, imu_std
from argparse import ArgumentParser, Namespace
from torchsummary import summary


import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class TrainArgs():
    training_data_path = "/ai-imu-dr/data_dm/ai_imu_trainning_data.npz"
    imu_data_name = "gyr_acc"
    vel_data_name = "car_vel"

    model_path = "/ai-imu-dr/temp/dmvelnet.p"
    plot_path = "/ai-imu-dr/temp/dmvelnet_train.png"

    device = "cuda"
    epochs = 20000
    save_every_epoch = 20
    continue_training = False
    constant_weight = False


def prepare_measurement_net(args):
    torch_meanet = DmVelNet(args.device, 0.6)
    if args.continue_training:
        torch_meanet.load(args.model_path)
    if args.device == "cuda":
        torch_meanet.cuda()
    return torch_meanet


def save_measurement_net(args, meanet):
    torch.save(meanet.state_dict(), args.model_path)
    print("  The DmVelNet nets are saved in the file " + args.model_path)


def prepare_data(args):
    trainning_data = dict(np.load(args.training_data_path))

    # normalize imu data
    imu_data_n = ((trainning_data[args.imu_data_name]-imu_mean)/imu_std)
    trainning_data["input"] = torch.from_numpy(imu_data_n[1:]).t().unsqueeze(0).to(args.device)

    # make output velocity cov maeaurement, y axis : front
    velocity_xyz = trainning_data[args.vel_data_name]
    velocity_delta = velocity_xyz[1:, :] - velocity_xyz[:-1, :]
    trainning_data["output"] = torch.from_numpy(velocity_delta).to(args.device)

    # make weights for each sample
    if args.constant_weight:
        trainning_data["weights"] = torch.ones_like(trainning_data["output"]).to(args.device)
    else:
        tmp = 10000 * np.square(velocity_delta).sum(axis=1)
        tmp[tmp > 1] = 1
        tmp[tmp < 0.001] = 0.001

        # plt.hist(tmp, 50, (0, 1))
        # plt.title("weight histogram")
        # plt.show()

        weights = np.array([tmp, tmp, tmp]).transpose()
        trainning_data["weights"] = torch.from_numpy(weights).to(args.device)

    print(" input shape", trainning_data["input"].shape)
    print(" output shape", trainning_data["output"].shape)
    print(" weights shape", trainning_data["weights"].shape)
    return trainning_data


if __name__ == '__main__':
    args = TrainArgs()

    cprint("Prepare data ...", 'green')
    trainning_data = prepare_data(args)

    cprint("Run Train ...", 'green')
    meanet = prepare_measurement_net(args)

    # summary(meanet, (6, 5))
    summary(meanet, (6, 20))

    # input = trainning_data["input"][:, :, 0:60]
    # # print(input)
    # print(meanet.forward(input)[40:, :])
    # print(meanet.forward(input[:, :, 20:]))

    # save_measurement_net(args, meanet)
    optimizer = set_mes_net_optimizer(meanet)

    losses = []
    g_norms = []
    for epoch in range(1, args.epochs + 1):
        loss_train, g_norm, std = train_mes_net_loop(args, trainning_data, epoch, meanet, optimizer)
        if epoch%args.save_every_epoch == 0:
            cprint('  Epoch {:2d} \tLoss: {:.2f}  Gradient Norm: {:.2f}'.format(epoch, loss_train, g_norm))
            print("              \tSTD: ", std)
            save_measurement_net(args, meanet)
        losses.append(loss_train.detach().cpu().numpy())
        g_norms.append(g_norm.detach().cpu().numpy())

    # plot the trainning to file
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.yscale('log')
    plt.title("train_loss")
    plt.subplot(1, 2, 2)
    plt.plot(g_norms)
    plt.yscale('log')
    plt.title("graident norm")
    plt.savefig(args.plot_path)

    cprint("Done.", 'green')
