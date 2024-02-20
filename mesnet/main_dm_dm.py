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
from utils_torch_filter import DmMesNet

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

cprint("prepare model")
class TestArgs():
    device = "cuda"
    training_data_path = "/ai-imu-dr/data_dm/ai_imu_trainning_data_1.npz"
    save_path = "/ai-imu-dr/data_dm/ai_imu_measurement.npy"
    model_path = "/ai-imu-dr/temp/dmmesnet.p"
    imu_mean = np.array([0, 0, 0, 0, 0, 9.81], dtype=np.float64)
    imu_std = np.array([0.1, 0.1, 0.1, 1.0, 1.0, 1.0], dtype=np.float64)

args = TestArgs()
torch_meanet = DmMesNet(args.device)
torch_meanet.load(args.model_path)
if args.device == "cuda":
    torch_meanet.cuda()

cprint("load imu data")

trainning_data = dict(np.load(args.training_data_path))
imu_data = (trainning_data["gyr_acc"] - args.imu_mean) / args.imu_std
imu_data = torch.from_numpy(imu_data).t().unsqueeze(0).to(args.device)

cprint("read data " + str(imu_data.shape))

measurements = torch_meanet.forward(imu_data)
measurements = measurements.detach().cpu().numpy()

cprint("result " + str(measurements.shape))

with open(args.save_path, 'wb') as f:
    np.save(f, measurements)

cprint("Done")
