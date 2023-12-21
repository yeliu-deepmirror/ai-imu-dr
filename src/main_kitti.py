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

class KITTIParameters(IEKF.Parameters):
    # gravity vector
    g = np.array([0, 0, -9.80655])

    cov_omega = 2e-4
    cov_acc = 1e-3
    cov_b_omega = 1e-8
    cov_b_acc = 1e-6
    cov_Rot_c_i = 1e-8
    cov_t_c_i = 1e-8
    cov_Rot0 = 1e-6
    cov_v0 = 1e-1
    cov_b_omega0 = 1e-8
    cov_b_acc0 = 1e-3
    cov_Rot_c_i0 = 1e-5
    cov_t_c_i0 = 1e-2
    cov_lat = 1
    cov_up = 10

    def __init__(self, **kwargs):
        super(KITTIParameters, self).__init__(**kwargs)
        self.set_param_attr()

    def set_param_attr(self):
        attr_list = [a for a in dir(KITTIParameters) if
                     not a.startswith('__') and not callable(getattr(KITTIParameters, a))]
        for attr in attr_list:
            setattr(self, attr, getattr(KITTIParameters, attr))


class KITTIDataset(BaseDataset):
    OxtsPacket = namedtuple('OxtsPacket',
                            'lat, lon, alt, ' + 'roll, pitch, yaw, ' + 'vn, ve, vf, vl, vu, '
                                                                       '' + 'ax, ay, az, af, al, '
                                                                            'au, ' + 'wx, wy, wz, '
                                                                                     'wf, wl, wu, '
                                                                                     '' +
                            'pos_accuracy, vel_accuracy, ' + 'navstat, numsats, ' + 'posmode, '
                                                                                  'velmode, '
                                                                                  'orimode')

    # Bundle into an easy-to-access structure
    datasets_fake = ['2011_09_26_drive_0093_extract', '2011_09_28_drive_0039_extract',
                     '2011_09_28_drive_0002_extract']
    """
    '2011_09_30_drive_0028_extract' has trouble at N = [6000, 14000] -> test data
    '2011_10_03_drive_0027_extract' has trouble at N = 29481
    '2011_10_03_drive_0034_extract' has trouble at N = [33500, 34000]
    """

    # training set to the raw data of the KITTI dataset.
    # The following dict lists the name and end frame of each sequence that
    # has been used to extract the visual odometry / SLAM training set
    odometry_benchmark = OrderedDict()
    odometry_benchmark["2011_10_03_drive_0027_extract"] = [0, 45692]
    odometry_benchmark["2011_10_03_drive_0042_extract"] = [0, 12180]
    odometry_benchmark["2011_10_03_drive_0034_extract"] = [0, 47935]
    odometry_benchmark["2011_09_26_drive_0067_extract"] = [0, 8000]
    odometry_benchmark["2011_09_30_drive_0016_extract"] = [0, 2950]
    odometry_benchmark["2011_09_30_drive_0018_extract"] = [0, 28659]
    odometry_benchmark["2011_09_30_drive_0020_extract"] = [0, 11347]
    odometry_benchmark["2011_09_30_drive_0027_extract"] = [0, 11545]
    odometry_benchmark["2011_09_30_drive_0028_extract"] = [11231, 53650]
    odometry_benchmark["2011_09_30_drive_0033_extract"] = [0, 16589]
    odometry_benchmark["2011_09_30_drive_0034_extract"] = [0, 12744]

    odometry_benchmark_img = OrderedDict()
    odometry_benchmark_img["2011_10_03_drive_0027_extract"] = [0, 45400]
    odometry_benchmark_img["2011_10_03_drive_0042_extract"] = [0, 11000]
    odometry_benchmark_img["2011_10_03_drive_0034_extract"] = [0, 46600]
    odometry_benchmark_img["2011_09_26_drive_0067_extract"] = [0, 8000]
    odometry_benchmark_img["2011_09_30_drive_0016_extract"] = [0, 2700]
    odometry_benchmark_img["2011_09_30_drive_0018_extract"] = [0, 27600]
    odometry_benchmark_img["2011_09_30_drive_0020_extract"] = [0, 11000]
    odometry_benchmark_img["2011_09_30_drive_0027_extract"] = [0, 11000]
    odometry_benchmark_img["2011_09_30_drive_0028_extract"] = [11000, 51700]
    odometry_benchmark_img["2011_09_30_drive_0033_extract"] = [0, 15900]
    odometry_benchmark_img["2011_09_30_drive_0034_extract"] = [0, 12000]

    def __init__(self, args):
        super(KITTIDataset, self).__init__(args)

        self.datasets_validatation_filter['2011_09_30_drive_0028_extract'] = [11231, 53650]
        self.datasets_train_filter["2011_10_03_drive_0042_extract"] = [0, None]
        self.datasets_train_filter["2011_09_30_drive_0018_extract"] = [0, 15000]
        self.datasets_train_filter["2011_09_30_drive_0020_extract"] = [0, None]
        self.datasets_train_filter["2011_09_30_drive_0027_extract"] = [0, None]
        self.datasets_train_filter["2011_09_30_drive_0033_extract"] = [0, None]
        self.datasets_train_filter["2011_10_03_drive_0027_extract"] = [0, 18000]
        self.datasets_train_filter["2011_10_03_drive_0034_extract"] = [0, 31000]
        self.datasets_train_filter["2011_09_30_drive_0034_extract"] = [0, None]

        for dataset_fake in KITTIDataset.datasets_fake:
            if dataset_fake in self.datasets:
                self.datasets.remove(dataset_fake)
            if dataset_fake in self.datasets_train:
                self.datasets_train.remove(dataset_fake)

    # read data from kitti removed.
    # if needed, see the original repo https://github.com/mbrossar/ai-imu-dr


def test_filter(args, dataset):
    iekf = IEKF(args.device)
    torch_iekf = TORCHIEKF(args.device)

    # put Kitti parameters
    iekf.filter_parameters = KITTIParameters()
    iekf.set_param_attr()
    torch_iekf.filter_parameters = KITTIParameters()
    torch_iekf.set_param_attr()

    torch_iekf.load(args, dataset)
    iekf.set_learned_covariance(torch_iekf)

    # print the network structure
    from torchsummary import summary
    torch_iekf.mes_net.cov0_measurement = torch_iekf.cov0_measurement
    summary(torch_iekf.mes_net, (6, 224), 100, "cpu")

    for i in range(0, len(dataset.datasets)):
        dataset_name = dataset.dataset_name(i)
        if dataset_name not in dataset.odometry_benchmark.keys():
            continue
        print("  Test filter on sequence: " + dataset_name)
        t, ang_gt, p_gt, v_gt, u = prepare_data(args, dataset, dataset_name, i,
                                                       to_numpy=True)
        N = None
        u_t = torch.from_numpy(u).double()
        measurements_covs = torch_iekf.forward_nets(u_t)
        measurements_covs = measurements_covs.detach().numpy()
        start_time = time.time()
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = iekf.run(t, u, measurements_covs,
                                                                   v_gt, p_gt, N,
                                                                   ang_gt[0])
        diff_time = time.time() - start_time
        print("  Execution time: {:.2f} s (sequence time: {:.2f} s)".format(diff_time,
                                                                          t[-1] - t[0]))
        mondict = {
            't': t, 'Rot': Rot, 'v': v, 'p': p, 'b_omega': b_omega, 'b_acc': b_acc,
            'Rot_c_i': Rot_c_i, 't_c_i': t_c_i,
            'measurements_covs': measurements_covs,
            }
        dataset.dump(mondict, args.path_results, dataset_name + "_filter.p")


class KITTIArgs():
        path_data_save = "../data"
        path_results = "../results"
        path_temp = "../temp"
        device = "cpu"

        epochs = 400
        seq_dim = 6000

        # training, cross-validation and test dataset
        cross_validation_sequences = ['2011_09_30_drive_0028_extract']
        test_sequences = ['2011_09_30_drive_0028_extract']
        continue_training = True

        # choose what to do
        train_filter = 1
        test_filter = 1
        results_filter = 1
        parameter_class = KITTIParameters


if __name__ == '__main__':
    parser = ArgumentParser(description="Training/Testing script parameters")
    parser.add_argument('--train_filter', default=False, action="store_true")
    parser.add_argument('--test_filter', default=True, action="store_true")
    parser.add_argument('--results_filter', default=False, action="store_true")
    opt, _ = parser.parse_known_args()

    args = KITTIArgs()
    dataset = KITTIDataset(args)

    if opt.train_filter:
        cprint("Run Train...", 'green')
        train_filter(args, dataset)

    if opt.test_filter:
        cprint("Run Test...", 'green')
        test_filter(args, dataset)

    if opt.results_filter:
        cprint("Run Result...", 'green')
        results_filter(args, dataset)
