import sys
sys.path.insert(0, '..')


import os
import argparse
import numpy as np
import math
import glob
import torch
import matplotlib.pyplot as plt
from termcolor import cprint
from model import DmVelNet
from dataset import InsDataset
from mesnet.utils_numpy_filter import NUMPYIEKF


import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class Trajectory:
    def __init__(self, dt = 1.0 / 100.0):
        self.dt_ = dt
        self.rotation_ = np.eye(3)
        self.position_ = np.zeros(3)
        self.trajectory_ = []

    def push_data(self, gyr_acc, velocity):
        self.rotation_ = self.rotation_.dot(NUMPYIEKF.so3exp(gyr_acc[0:3] * self.dt_))
        velocity_world = self.rotation_.dot(velocity.reshape([3]))
        self.position_ = self.position_ + velocity_world * self.dt_
        self.trajectory_.append(self.position_)

    def trajectory_array(self):
        return np.array(self.trajectory_)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test_data_path', type=str, default="/ai-imu-dr/data_dm/ai_imu_trainning_data_test_1.npz")
    parser.add_argument('--test_data_key', type=str, default="")
    parser.add_argument('--model_path', type=str, default="/ai-imu-dr/temp/dmvelnet.p")
    parser.add_argument('--device', type=str, default="cuda")

    # model parameters
    parser.add_argument('--block_size', type=int, default=50)

    args, _ = parser.parse_known_args()

    test_data_path = args.test_data_path
    if len(args.test_data_key) > 0:
        test_data_path = "/ai-imu-dr/data_dm/ai_imu_trainning_data_test_" + args.test_data_key + ".npz"
    cprint("process " + test_data_path, 'green')

    log_every_n = 5000
    cprint("Prepare model ...", 'green')
    torch_meanet = DmVelNet(args.device, args.block_size)
    torch_meanet.load(args.model_path)
    if args.device == "cuda":
        torch_meanet.cuda()
    torch_meanet.eval()

    cprint("Prepare data ...", 'green')
    test_set = InsDataset(test_data_path, False, args.block_size)

    trajectory = Trajectory()
    trajectory_gt = Trajectory()

    for i in range(len(test_set)):
        input_data = test_set.get_test_item(i).to(args.device)
        velocity_n = torch_meanet(input_data).detach().cpu().numpy()

        # compute the trajectory
        velocity = test_set.to_raw_velocity(velocity_n)
        gyr_acc = test_set.get_raw_gyr_acc(i)
        trajectory.push_data(gyr_acc, velocity)

        velocity_gt = test_set.get_raw_vel_gt(i)
        if velocity_gt is not None:
            trajectory_gt.push_data(gyr_acc, velocity_gt)

        # set velocity for next model input
        test_set.set_velocity(i, velocity_n)

        if i%log_every_n == 0:
            print(i, velocity)

    # plot the trajectory
    trajectory_array = trajectory.trajectory_array()
    trajectory_gt_array = trajectory_gt.trajectory_array()

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], trajectory_array[:, 2], label="estimate")
    if trajectory_gt_array.shape[0] > 0:
        ax.plot(trajectory_gt_array[:, 0], trajectory_gt_array[:, 1], trajectory_gt_array[:, 2], label="gt")
    ax.legend()

    plt.show()
