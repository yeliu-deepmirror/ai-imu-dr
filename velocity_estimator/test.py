import sys
sys.path.insert(0, '..')

import os
import shutil
import numpy as np
from collections import namedtuple
import glob
import time
import datetime
import pickle
import torch
import matplotlib.pyplot as plt
from termcolor import cprint
from model import DmVelNet
from mesnet.utils_numpy_filter import NUMPYIEKF
from train_torch_filter import imu_mean, imu_std

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def plot_trajectory(velocity_delta, gyr_acc, dt = 1.0 / 100.0):
    rotation = np.eye(3)
    position = np.zeros(3)
    velocity_car = np.zeros(3)

    trajectory = []
    # compute the trajectory based on the car velocity
    for i in range(gyr_acc.shape[0]):
        rotation = rotation.dot(NUMPYIEKF.so3exp(gyr_acc[i, 0:3] * dt))

        # update car velocity
        velocity_car = velocity_car + velocity_delta[i]

        # keep only front velocity
        # velocity_car[0] = 0
        # velocity_car[2] = 0

        # update world position
        velocity_world = rotation.dot(velocity_car)
        # velocity_world[2] = 0

        position = position + velocity_world * dt
        trajectory.append(position)

    return np.array(trajectory)


def trajectory_range(trajectory):
    return (trajectory.max(axis=0) - trajectory.min(axis=0)).max()


cprint("prepare model")
class TestArgs():
    device = "cuda"
    training_data_path = "/ai-imu-dr/data_dm/ai_imu_trainning_data_test_phone.npz"
    save_path = "/ai-imu-dr/data_dm/ai_imu_measurement.npy"
    model_path = "/ai-imu-dr/temp/dmvelnet.p"

args = TestArgs()
# smaller dropout rate when testing
torch_meanet = DmVelNet(args.device, 0.1)
torch_meanet.load(args.model_path)
if args.device == "cuda":
    torch_meanet.cuda()

cprint("load imu data")

trainning_data = dict(np.load(args.training_data_path))
imu_data = (trainning_data["gyr_acc"] - imu_mean) / imu_std
imu_data = torch.from_numpy(imu_data).t().unsqueeze(0).to(args.device)

cprint("read data " + str(imu_data.shape))

measurements = torch_meanet.forward(imu_data)
measurements = measurements.detach().cpu().numpy()

cprint("result " + str(measurements.shape))
# print(measurements)

with open(args.save_path, 'wb') as f:
    np.save(f, measurements)

cprint("Done")

trajectory_estimate = plot_trajectory(measurements, trainning_data["gyr_acc"])

if trainning_data.get("car_vel") is not None:
    velocity_xyz = trainning_data["car_vel"]
    velocity_delta = velocity_xyz[1:, :] - velocity_xyz[:-1, :]
    trajectory_gt = plot_trajectory(velocity_delta, trainning_data["gyr_acc"][1:])

    rescale = trajectory_range(trajectory_gt) / trajectory_range(trajectory_estimate)
    if rescale > 10:
        trajectory_estimate = rescale * trajectory_estimate
        cprint("rescale factor : " + str(rescale), 'yellow')

ax = plt.figure().add_subplot(projection='3d')
ax.plot(trajectory_estimate[:, 0], trajectory_estimate[:, 1], trajectory_estimate[:, 2], label="estimate")
if trainning_data.get("car_vel") is not None:
    ax.plot(trajectory_gt[:, 0], trajectory_gt[:, 1], trajectory_gt[:, 2], label="gt")
ax.legend()

plt.show()
