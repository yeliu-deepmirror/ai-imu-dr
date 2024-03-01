import torch
import numpy as np
import matplotlib.pyplot as plt

class InsDataset(torch.utils.data.Dataset):
    def __init__(self, training_data_path, trainning = True, block_size = 50):
        super(torch.utils.data.Dataset, self).__init__()
        print(" block_size", block_size)

        self.block_size_ = block_size
        self.imu_data_name_ = "gyr_acc"
        self.vel_data_name_ = "car_vel"
        self.trainning_ = trainning
        trainning_data = dict(np.load(training_data_path))

        self.imu_mean = np.array([0, 0, 0, 0, 0, 9.81], dtype=np.float32)
        self.imu_std = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], dtype=np.float32)
        self.vel_mean = np.array([0, 0, 0], dtype=np.float32)
        self.vel_std = np.array([1, 1, 1], dtype=np.float32)

        self.data_gyr_acc_ = (trainning_data[self.imu_data_name_].astype(np.float32) - self.imu_mean) / self.imu_std
        print(" load input shape", self.data_gyr_acc_.shape)
        if trainning:
            assert trainning_data.get(self.vel_data_name_) is not None
            self.data_velocity_xyz_ = (trainning_data[self.vel_data_name_].astype(np.float32) - self.vel_mean) / self.vel_std
            print(" load output shape", self.data_velocity_xyz_.shape)
        else :
            self.data_velocity_xyz_ = (np.zeros((self.data_gyr_acc_.shape[0], 3)) - self.vel_mean) / self.vel_std
            if trainning_data.get(self.vel_data_name_) is not None:
                self.data_velocity_xyz_gt_ = (trainning_data[self.vel_data_name_].astype(np.float32) - self.vel_mean) / self.vel_std
            else:
                self.data_velocity_xyz_gt_ = None

        self.raw_size_ = self.data_velocity_xyz_.shape[0] - self.block_size_ - 1
        self.samples_ = np.arange(0, self.raw_size_).tolist()

        if trainning and False:
            print(" resample the tranning set based on velocity change")
            velocity_delta = self.data_velocity_xyz_[1:, :] - self.data_velocity_xyz_[:-1, :]
            tmp = np.abs(velocity_delta).sum(axis=1)

            # tmp[tmp > 0.03] = 0.03
            # plt.hist(tmp, 50, (0, 0.03))
            # plt.title("delta velocity histogram")
            # plt.show()

            # double the sample which dv > 0.005
            sample_dv_threshold = 0.005
            for i in range(tmp.shape[0]):
                if tmp[i] > sample_dv_threshold:
                    self.samples_.append(i)


    def __len__(self):
        return len(self.samples_)


    def __getitem__(self, index):
        id_begin = self.samples_[index]
        id_end = id_begin + self.block_size_

        # normalize imu data and velocity data
        input_n = np.concatenate((self.data_gyr_acc_[id_begin:id_end], self.data_velocity_xyz_[id_begin:id_end]), axis=1)
        input_n = np.transpose(input_n)
        output_n = self.data_velocity_xyz_[id_end]

        return input_n, output_n


    # the following functions are used only for test

    def get_test_item(self, index):
        input_n, _ = self.__getitem__(index)
        input_n = np.expand_dims(input_n, axis=0).astype(np.float32)
        return torch.from_numpy(input_n)


    def set_velocity(self, index, velocity):
        self.data_velocity_xyz_[index + self.block_size_, :] = velocity


    def get_raw_gyr_acc(self, index):
        gyr_acc = self.data_gyr_acc_[index + self.block_size_, :]
        return gyr_acc * self.imu_std + self.imu_mean


    def get_raw_vel_gt(self, index):
        if self.trainning_:
            vel = self.data_velocity_xyz_[index + self.block_size_, :]
            return vel * self.vel_std + self.vel_mean

        if self.data_velocity_xyz_gt_ is None:
            return None
        vel = self.data_velocity_xyz_gt_[index + self.block_size_, :]
        return vel * self.vel_std + self.vel_mean


    def to_raw_velocity(self, velocity_model):
        return velocity_model * self.vel_std + self.vel_mean
