import logging
import pickle

import numpy as np
from scipy import signal
from torch.utils.data import Dataset

from src.graph.graph import Graph


class NTUDataloader(Dataset):
    """
    Dataloader and preprocessor for NTU dataset.
    """
    DATA_ARGS = {
        'xsub60': {'class': 60, 'folder': 'ntu60', 'shape': [3, 6, 300, 25, 2]},
        'xview60': {'class': 60, 'folder': 'ntu60', 'shape': [3, 6, 300, 25, 2]},
        'xsub120': {'class': 120, 'folder': 'ntu120', 'shape': [3, 6, 300, 25, 2]},
        'xsetup120': {'class': 120, 'folder': 'ntu120', 'shape': [3, 6, 300, 25, 2]},
    }

    def __init__(self, phase, args):
        self.phase = phase
        self.args = args
        self.T = self.args.num_frame
        self.inputs = self.args.inputs
        self.debug = self.args.debug
        self.normalize = self.args.normalize
        self.mean_map = None
        self.std_map = None
        self.shape = self.DATA_ARGS[self.args.dataset]['shape']
        self.num_classes = self.DATA_ARGS[self.args.dataset]['class']
        self.sub_folder = self.DATA_ARGS[self.args.dataset]['folder']
        # generate graph data depending on dataset

        assert self.args.layout == "ntu"
        self.graph = Graph(self.args.layout)
        self.A = self.graph.A
        self.parts = self.graph.parts
        self.conn = self.graph.connect_joint

        if self.args.transform:
            logging.info("Loading transformed data.")
            dataset_path = '{}/transformed/{}/{}'.format(self.args.root_folder, self.sub_folder, self.args.dataset)
        else:
            logging.info("Loading processed data.")
            dataset_path = '{}/processed/{}/{}'.format(self.args.root_folder, self.sub_folder, self.args.dataset)

        self.data_path = '{}/{}_{}_data.npy'.format(dataset_path, self.args.dataset, self.phase)
        self.label_path = '{}/{}_{}_label.pkl'.format(dataset_path, self.args.dataset, self.phase)

        self.load_data()
        if self.normalize:
            logging.info("Normalizing data.")
            self.mean_std()

        if self.args.filter:
            self.butterworth = signal.butter(self.args.filter_order, 0.5, output='sos')

    def load_data(self):
        logging.info('Trying to load NTU data.')
        try:
            self.data = np.load(self.data_path, mmap_mode='r')
            with open(self.label_path, 'rb') as f:
                self.name, self.label, self.seq_len = pickle.load(f, encoding='latin1')
        except Exception:
            logging.info('')
            logging.error('Error: Loading data files: {} or {}!'.format(self.data_path, self.label_path))
            raise ValueError('Error: Loading data files: {} or {}!'.format(self.data_path, self.label_path))

        if self.args.debug:
            logging.info('Loading SMALL dataset!')
            self.data = self.data[:200]
            self.label = self.label[:200]
            self.name = self.name[:200]
            self.seq_len = self.seq_len[:200]

        logging.info('Done with loading data.')

    def mean_std(self):
        """
        C == channels,
        M == skeletons,
        N == number of skeletons,
        V == vertices,
        T == frame
        :return:
        """
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))
        logging.info('Mean value: {} \n Std value: {}'.format(self.mean_map, self.std_map))

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = np.array(self.data[idx])
        label = self.label[idx]
        name = self.name[idx]

        if self.normalize:
            data = (data - self.mean_map) / self.std_map

        # (C, max_frame, V, M) -> (I, C*2, T, V, M)
        joint, velocity, bone, accel = self.multi_input(data[:, :self.T, :, :])
        data_new = []
        if 'J' in self.inputs:
            data_new.append(joint)
        if 'V' in self.inputs:
            data_new.append(velocity)
        if 'B' in self.inputs:
            data_new.append(bone)
        if 'A' in self.inputs:
            data_new.append(accel)
        data_new = np.stack(data_new, axis=0)

        return data_new, label, name

    def multi_input(self, data):
        """
        Generate features from data.
        :param data:
        :return: features
        """
        C, T, V, M = data.shape
        joint = np.zeros((C * 2, T, V, M), dtype=np.float32)
        velocity = np.zeros((C * 2, T, V, M), dtype=np.float32)
        accel = np.zeros((C * 2, T, V, M), dtype=np.float32)
        bone = np.zeros((C * 2, T, V, M), dtype=np.float32)
        joint[:C, :, :, :] = data
        for i in range(V):
            joint[C:, :, i, :] = data[:, :, i, :] - data[:, :, 1, :]
        for i in range(T - 2):
            velocity[:C, i, :, :] = (data[:, i + 1, :, :] - data[:, i, :, :]) / 1
            velocity[C:, i, :, :] = (data[:, i + 2, :, :] - data[:, i, :, :]) / 2
        for i in range(len(self.conn)):
            bone[:C, :, i, :] = data[:, :, i, :] - data[:, :, self.conn[i], :]
        bone_length = 0
        for i in range(C):
            bone_length += bone[i, :, :, :] ** 2
        bone_length = np.sqrt(bone_length) + 0.0001
        for i in range(C):
            bone[C + i, :, :, :] = np.arccos(bone[i, :, :, :] / bone_length)
        # acceleration
        if self.args.filter:
            filtered_signal = self.butterworth_filter(velocity)
            for i in range(T - 2):
                accel[:C, i, :, :] = (filtered_signal[:C, i + 1, :, :] - filtered_signal[:C, i, :, :]) / 1
                accel[C:, i, :, :] = (filtered_signal[C:, i + 2, :, :] - filtered_signal[C:, i, :, :]) / 2
        else:
            for i in range(T - 2):
                accel[:C, i, :, :] = (velocity[:C, i + 1, :, :] - velocity[:C, i, :, :]) / 1
                accel[C:, i, :, :] = (velocity[C:, i + 2, :, :] - velocity[C:, i, :, :]) / 2

        return joint, velocity, bone, accel

    def butterworth_filter(self, data):
        """
        Filter the signal with the give filter order.
        :param data:
        :return:
        """
        C, T, V, M = data.shape
        filtered_signal = np.zeros((C, T, V, M))
        for v in range(V):
            for c in range(C):
                filtered_signal[c, :, v, 0] = signal.sosfiltfilt(self.butterworth, data[c, :, v, 0])
        return filtered_signal


class NTULocationFeeder:
    def __init__(self, data_shape):
        _, _, self.T, self.V, self.M = data_shape

    def load(self, names):
        location = np.zeros((len(names), 2, self.T, self.V, self.M))
        for i, name in enumerate(names):
            with open(name, 'r') as fr:
                frame_num = int(fr.readline())
                for frame in range(frame_num):
                    if frame >= self.T:
                        break
                    person_num = int(fr.readline())
                    for person in range(person_num):
                        fr.readline()
                        joint_num = int(fr.readline())
                        for joint in range(joint_num):
                            v = fr.readline().split(' ')
                            if joint < self.V and person < self.M:
                                location[i, 0, frame, joint, person] = float(v[5])
                                location[i, 1, frame, joint, person] = float(v[6])
        return location
