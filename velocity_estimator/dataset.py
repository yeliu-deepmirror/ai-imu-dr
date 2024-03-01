import torch
import numpy as np

class InsDataset(torch.utils.data.Dataset):
    def __init__(self, training_data_path, trainning = True, block_size = 50):
        super(torch.utils.data.Dataset, self).__init__()
        self.block_size_ = block_size
        self.imu_data_name_ = "gyr_acc"
        self.vel_data_name_ = "car_vel"
        self.trainning_ = trainning
        trainning_data = dict(np.load(training_data_path))

        self.imu_mean = np.array([0, 0, 0, 0, 0, 9.81], dtype=np.float64)
        self.imu_std = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], dtype=np.float64)
        self.vel_mean = np.array([0, 0, 0], dtype=np.float64)
        self.vel_std = np.array([1, 1, 1], dtype=np.float64)

        self.data_gyr_acc_ = (trainning_data[self.imu_data_name_] - self.imu_mean) / self.imu_std
        print(" load input shape", self.data_gyr_acc_.shape)
        if trainning:
            assert trainning_data.get(self.vel_data_name_) is not None
            self.data_velocity_xyz_ = (trainning_data[self.vel_data_name_] - self.vel_mean) / self.vel_std
            print(" load output shape", self.data_velocity_xyz_.shape)
        else :
            self.data_velocity_xyz_ = (np.zeros((self.data_gyr_acc_.shape[0], 3)) - self.vel_mean) / self.vel_std
            if trainning_data.get(self.vel_data_name_) is not None:
                self.data_velocity_xyz_gt_ = (trainning_data[self.vel_data_name_] - self.vel_mean) / self.vel_std
            else:
                self.data_velocity_xyz_gt_ = None

    def __len__(self):
        return self.data_velocity_xyz_.shape[0] - self.block_size_ - 1


    def __getitem__(self, index):
        id_begin = index
        id_end = index + self.block_size_

        # normalize imu data and velocity data
        input_n = np.concatenate((self.data_gyr_acc_[id_begin:id_end], self.data_velocity_xyz_[id_begin:id_end]), axis=1)
        input_n = np.transpose(input_n)
        output_n = self.data_velocity_xyz_[id_end]

        return input_n, output_n


    def get_test_item(self, index):
        input_n, _ = self.__getitem__(index)
        input_n = np.expand_dims(input_n, axis=0)
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
