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
from navpy import lla2ned
from collections import OrderedDict
from dataset import BaseDataset
from utils_torch_filter import TORCHIEKF
from utils_numpy_filter import NUMPYIEKF as IEKF
from utils import prepare_data
from train_torch_filter import train_filter
from utils_plot import results_filter
from argparse import ArgumentParser, Namespace

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

cprint("prepare model")
device = "cuda"

torch_iekf = TORCHIEKF(device)
path_iekf = os.path.join("../temp", "iekfnets.p")
mondict = torch.load(path_iekf)
torch_iekf.load_state_dict(mondict)
torch_iekf.cuda()

torch_iekf.u_loc = torch.tensor([ 4.3230e-05,  2.2210e-04, -2.4795e-03, -5.1683e-02,  1.5524e-01, 9.7926e+00], dtype=torch.float64).to(device)
torch_iekf.u_std = torch.tensor([0.0213, 0.0248, 0.1365, 0.8894, 0.9457, 0.3641], dtype=torch.float64).to(device)
torch_iekf.cov0_measurement = torch.Tensor([1.0, 10.0]).double().to(device)

cprint("load imu data")

imu_data = torch.from_numpy(np.load(os.path.join("../temp", "imu_data.npy"))).to(device);

cprint("read data " + str(imu_data.shape))

imu_data_n = ((imu_data-torch_iekf.u_loc)/torch_iekf.u_std).t().unsqueeze(0)
imu_data_n = imu_data_n[:, :6]
measurements_covs = torch_iekf.mes_net.forward_tmp(imu_data_n, torch_iekf)
measurements_covs = measurements_covs.detach().cpu().numpy()

cprint("result " + str(measurements_covs.shape))

with open(os.path.join("../temp", "car_cov.npy"), 'wb') as f:
    np.save(f, measurements_covs)

cprint("Done")
