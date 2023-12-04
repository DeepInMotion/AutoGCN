import multiprocessing
import os
import pickle
import time

import tqdm
from tqdm import tqdm
import numpy as np
import os.path as osp
import logging

from transform import TransformNTU
from src.utils.visualization import vis


class NTUGendata:
    MAX_BODY = 2
    NUM_JOINT = 25
    MAX_CHANNEL = 3  # Joint xyz + ColorX ColorY -> 5
    NUM_CHANNEL = 3
    MAX_FRAME = 300
    channel_name = ['x', 'y', 'z', 'colorX', 'colorY']
    training_subjects_60 = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
    training_cameras_60 = [2, 3]
    training_subjects_120 = [
        1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38,
        45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82,
        83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
    ]
    training_setups_120 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

    def __init__(self, root_path, ntu60_path, ntu120_path, ignore_path, args):

        self.choice = args.choice
        self.transform = args.trans
        self.visualize = args.vis
        # self.raw_path = raw_path
        self.debug = args.debug
        self.skeletons_drop = dict()
        self.skeletons_alternate = 0
        self.args = args

        # alter the save path
        if self.transform:
            self.out_path = '{}/transformed/{}'.format(root_path, self.choice)
        else:
            self.out_path = '{}/processed/{}'.format(root_path, self.choice)

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        if self.transform:
            self.transformer = TransformNTU(self.out_path)
            self.transform_opt = args.trans_opt
            # self.transform_opt = ['pad', 'parallel_s', 'parallel_h', 'sub', 'view', 'scale']

        # logging - store log in save path
        self.data_corrupt_logger = logging.getLogger('ntu_corrupt_log')
        self.data_corrupt_logger.setLevel(logging.INFO)
        self.data_corrupt_logger.propagate = False
        self.data_corrupt_logger.addHandler(logging.FileHandler(osp.join(self.out_path, 'ntu_corrupt_log.log')))

        # open ignore.txt file
        try:
            if ignore_path is not None:
                with open(ignore_path, 'r') as f:
                    self.skeleton_ignore = [line.strip() + '.skeleton' for line in f.readlines()]
        except ValueError:
            print("Error reading [{}]".format(ignore_path))

        self.file_list = []
        for folder in [ntu60_path, ntu120_path]:
            for filename in os.listdir(folder):
                self.file_list.append((folder, filename))
            if '120' not in self.choice:  # for NTU 60, only one folder
                break

    def process(self, cores):

        # TODO make extra for ntu120
        if self.choice == 'ntu60':
            benchmark = ['xsub60'] #['xsub60', 'xview60']
        else:
            benchmark = ['xsetup120'] #['xsub120', 'xsetup120']

        part = ['train', 'eval'] # ['train', 'eval']

        for b in benchmark:
            for p in part:
                self.gendata_ntu(phase=p, bench=b)

        logging.info("Done!")


        """
        func_args = []
        for b in benchmark:
            for p in part:

                f_arg = (p, b)
                func_args.append(f_arg)

        cpu_available = multiprocessing.cpu_count()
        if cores > 1:
            num_args = len(func_args)
            cores = min(cpu_available, num_args, cores)
        print("Cores: {} Available, {} Chosen.".format(cpu_available, cores))

        start_t = time.time()
        pool = multiprocessing.Pool(cores)
        pool.starmap(self.gendata_ntu, func_args)

        pool.close()
        pool.join()

        end_t = time.time()
            
        print("Done processing in {}".format(end_t - start_t))
        """
        # print(f"@@@ DONE all processing in {end_t - start_t:.2f}s @@@", flush=True)

    def gendata_ntu(self, phase, bench):
        """
        Generate .npy binary from ntu skeleton data.
        :return: None
        """
        # crawl over data
        logging.info('Start processing {}-{}'.format(phase, bench))
        sample_data = []
        sample_label = []
        sample_path = []
        sample_length = []

        # iterizer = tqdm(sorted(self.file_list), dynamic_ncols=True)
        for folder, filename in self.file_list:
        # crawl over skeleton files to extract infos
        # for filename in crawl:
            if filename in self.skeleton_ignore or filename.endswith('.skeleton') is not True:
                continue

            # get values from filename
            file_path = os.path.join(folder, filename)
            setup_number = int(filename[filename.find('S') + 1:filename.find('S') + 4])
            action_class = int(filename[filename.find('A') + 1:filename.find('A') + 4])
            subject_id = int(filename[filename.find('P') + 1:filename.find('P') + 4])
            camera_id = int(filename[filename.find('C') + 1:filename.find('C') + 4])

            # determine which category sample belongs to
            if bench == 'xsub60':
                istraining = (subject_id in self.training_subjects_60)
            elif bench == 'xsub120':
                istraining = (subject_id in self.training_subjects_120)
            elif bench == 'xview60':
                istraining = (camera_id in self.training_cameras_60)
            elif bench == 'xsetup120':
                istraining = (setup_number in self.training_setups_120)
            else:
                raise ValueError('Invalid benchmark provided: {}'.format(phase))

            if phase == 'train':
                issample = istraining
            elif phase == 'eval':
                issample = not istraining
            else:
                raise ValueError('Invalid data part provided: {}'.format(phase))

            # update body dict
            if issample:
                sample_path.append(file_path)
                sample_label.append(action_class - 1)

        # skeleton_data = np.zeros((len(skeleton_label), self.MAX_CHANNEL, self.MAX_FRAME, self.NUM_JOINT, self.MAX_BODY),
        #                         dtype=np.float32)

        pop_list = []
        corrupt_count = 0
        # prog_bar = mmcv.ProgressBar(len(sample_path))
        for i, skel_path in enumerate(tqdm(sample_path)):
            # empty array to store skeleton
            skeleton_data = np.zeros((self.MAX_CHANNEL, self.MAX_FRAME, self.NUM_JOINT, self.MAX_BODY),
                                     dtype=np.float32)
            data, corrupt, motion = self.read_xyz(skel_path, skel_path[-15:])
            if corrupt > 0:
                corrupt_count += 1
            # if corrupt != 0:
            # data = self.transformer.trans_sub_center(data)
            # vis(data)
            # skeleton_data[i, :, 0:data.shape[1], :, :] = data
            skeleton_data[:, 0:data.shape[1], :, :] = data
            sample_data.append(skeleton_data)
            # append to list
            sample_length.append(data.shape[1])
            # prog_bar.update()

        if self.visualize:
            # num = random.randint(0, len(skeleton_label))
            num = 8
            # print('Showing skeleton'.format(skeleton_label[num]))
            print('Display RAW skeleton.')
            vis(sample_data[num, ...], is_3d=True, title='raw', pause=0.005, gif=True)

        sample_data = np.array(sample_data, dtype=np.float32)
        if self.transform:
            # disentangle transformation and reading of skeleton data
            print('Transform skeleton data with following options: {}'.format(self.transform_opt))
            sample_data = self.transformer.transform(sample_data, self.transform_opt)
        if self.visualize:
            print('Display PROCESSED skeleton.')
            vis(sample_data[num, ...], is_3d=True, title='processed', pause=0.005, gif=True)
        """
        if len(self.skeletons_drop) > 0 or len(pop_list) > 0:
            print('{} skeletons have NULL frames!'.format(len(self.skeletons_drop)))
            print('During the transformation process {} have been dropped'.format(len(pop_list)))
            # drop skeletons and
            fp = np.delete(fp, pop_list)
            for i in pop_list:
                del skeleton_name[i]
                del skeleton_label[i]

            # compare length
            if fp.shape[0] != len(skeleton_name):
                raise ValueError('Skeleton names and data array differ!')
        """
        # val-xsub60 (70256, 3, 300, 25, 2) float32 size 3161520000
        # Save data
        if not self.debug:
            save_path = '{}/{}'.format(self.out_path, bench)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            print('Save data to {}'.format(save_path))
            # np.save('{}/{}_{}_data.npy'.format(save_path, bench, phase), skeleton_data)
            np.save('{}/{}_{}_data.npy'.format(save_path, bench, phase), sample_data)
            with open('{}/{}_{}_label.pkl'.format(save_path, bench, phase), 'wb') as f:
                pickle.dump((sample_path, list(sample_label), list(sample_length)), f)
                # pickle.dump((skeleton_name, list(skeleton_label)), f)

        logging.info('{}-{} finished.'.format(bench, phase))

        logging.info('{} skeletons have NULL frames'.format(self.skeletons_drop))
        logging.info('{} skeletons have alternating number of bodies'.format(self.skeletons_alternate))

    def read_xyz(self, file, name):
        """
        Loop over extracted sequence info to get x, y and z coordinates.
        Log the skeletons which have missing frames and drop them.
        Log the files in which skeletons frequency is not the same over time.
        :param name: skeleton name
        :param file: path to skeleton file
        :return: ndarray [3, num_frames, num_joints, bodies],
        """
        seq_info = self.read_skeleton(file)
        frames_drop_idx = []
        body_num = []
        body_id = []
        skeleton = np.zeros((self.MAX_BODY, seq_info['numFrame'], self.NUM_JOINT, self.MAX_CHANNEL), dtype=np.float32)
        for n, f in enumerate(seq_info['frameInfo']):
            if f['numBody'] == 0:
                # indicates that no data in the current frame!
                body_num.append(0)
                frames_drop_idx.append(n)
            # check if bodies are changing
            elif f['numBody'] == 1:
                body_num.append(1)
            elif f['numBody'] == 2:
                body_num.append(2)
            else:
                body_num.append(-1)
            for m, b in enumerate(f['bodyInfo']):
                for j, v in enumerate(b['jointInfo']):
                    if m < self.MAX_BODY and j < self.NUM_JOINT:
                        # skip max bodies
                        skeleton[m, n, j, :] = [v['x'], v['y'], v['z']]
                    else:
                        pass

        # check list
        if all(x == body_num[0] for x in body_num) is False:
            # log alternation if no frames were dropped
            counts = {}
            for item in body_num:
                if item in counts:
                    counts[item] += 1
                else:
                    counts[item] = 1
            self.skeletons_alternate += 1
            self.data_corrupt_logger.info('{}: bodies alternation: {} \n'.format(name[:-9], body_num))

        body_set = set(body_num)
        motion = []
        num_frames_drop = len(frames_drop_idx)
        if 2 in body_set:
            # TODO filter out data which has two skeletons but only one is expected
            # 2 bodies in data
            # [num_frames, 25, 3]
            for body in skeleton:
                motion.append(np.sum(np.var(body, axis=0)))
            energy = np.array([self.get_nonzero_std(x) for x in skeleton])
            index = energy.argsort()[::-1][0:self.MAX_BODY]
            skeleton = skeleton[index]

            # filter out
            energy_min = max(energy) * 0.9
            del_list = np.where(np.array(energy < energy_min) is True)[0]

            """
            for i in del_list[::-1]:
                 if not self.xy_valid(data[i]):
                    # delete obj should be obj
                    vis(data, is_3d=False)
                    # TODO: seems like algorithm right now prefers noisy data
                    # TODO: missing data within body points!
                    # data = np.delete(data, [i, ...])
                    data = np.concatenate([data[:i], data[i + 1:]], 0)
                    # vis(data, is_3d=False)
            """
            energy = np.array([self.get_nonzero_std(x) for x in skeleton])
            index = energy.argsort()[::-1][0:self.MAX_BODY]
            skeleton = skeleton[index]

            # motion = motion[index]

        skeleton = skeleton.transpose(3, 1, 2, 0)
        return skeleton.astype(np.float32), num_frames_drop, motion
        """
        if False: #num_frames_drop > 0:
            # TODO: also check if bodys are different and one is dropped
            expected_frames = seq_info['numFrame']
            self.data_corrupt_logger.info('{}: {} frames of {} missed: {}\n'.format(name[:-9], num_frames_drop,
                                                                                    expected_frames, frames_drop_idx))
            self.skeletons_drop[name[:-9]] = frames_drop_idx
            # drop the frames
            if frames_drop_idx[0] == 0:
                start = frames_drop_idx[-1] + 1
                end = expected_frames - 1
            else:
                start = 0
                end = frames_drop_idx[0]
            data = data[:, start:end, :, :]
        """

    def statistics_ntu(self, data):
        """
        Calculate some statistics of generated data.
        ndarray [3, num_frames, num_joints, bodies]
        1) Histogram of frame length
        2) Histogram action classes
        3) Occurrences of frame drops
        4) Changing number of skeletons
        :return: None
        """
        # TODO
        import matplotlib.pyplot as plt
        import scipy.stats as st

        frame_length = []

        for x in data:
            for y in x:
                frame_length.append(y)

        plt.hist(frame_length, bins=50, label='Frame length')
        mn, mx = plt.xlim()
        plt.xlim(mn, mx)
        kde_xs = np.linspace(mn, mx, 300)
        kde = st.gaussian_kde(x)
        plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
        plt.ylabel("Probability")
        plt.xlabel("Frame length")
        plt.title("Frame Length")

    @staticmethod
    def get_nonzero_std(s):
        """
        Std deviation of three channels.
        :param s:
        :return:
        """
        # (T,V,C) not ctv
        T, V, C = s.shape
        assert C == 3 and V == 25
        index = s.sum(-1).sum(-1) != 0  # select valid frames
        s = s[index]
        if len(s) != 0:
            s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
        else:
            s = 0
        return s

    def xy_valid(self, s):
        """
        Body valid? [C, T, V]
        :param s:
        :return:
        """
        # TODO
        s = s.transpose(2, 0, 1)
        assert s.shape[0] == 3 and s.shape[2] == 25
        index = s[:self.NUM_CHANNEL].sum(0).sum(-1) != 0  # select valid frames
        if all(index) is False:
            return False
        s = s[:, index]
        x = s[0, 0].max() - s[0, 0].min()
        y = s[1, 0].max() - s[1, 0].min()
        return y * 0.8 > x

    @staticmethod
    def read_skeleton(file):
        """
        This function loops over the Frames and extracts the corresponding frame infos which are then
        stored in dict.
        Code adopted from https://github.com/shahroudy/NTURGB-D
        :param file: path to raw skeleton file.
        :return: dict
        """
        with open(file, 'r') as f:
            skeleton_sequence = {}
            skeleton_sequence['numFrame'] = int(f.readline())
            skeleton_sequence['frameInfo'] = []
            for t in range(skeleton_sequence['numFrame']):
                frame_info = {}
                frame_info['numBody'] = int(f.readline())
                frame_info['bodyInfo'] = []
                for m in range(frame_info['numBody']):
                    body_info = {}
                    body_info_key = [
                        'bodyID', 'clipedEdges', 'handLeftConfidence',
                        'handLeftState', 'handRightConfidence', 'handRightState',
                        'isResticted', 'leanX', 'leanY', 'trackingState'
                    ]
                    body_info = {
                        k: float(v)
                        for k, v in zip(body_info_key,
                                        f.readline().split())
                    }
                    body_info['numJoint'] = int(f.readline())
                    body_info['jointInfo'] = []
                    for v in range(body_info['numJoint']):
                        joint_info_key = [
                            'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                            'orientationW', 'orientationX', 'orientationY',
                            'orientationZ', 'trackingState'
                        ]
                        joint_info = {
                            k: float(v)
                            for k, v in zip(joint_info_key,
                                            f.readline().split())
                        }
                        body_info['jointInfo'].append(joint_info)
                    frame_info['bodyInfo'].append(body_info)
                skeleton_sequence['frameInfo'].append(frame_info)

        return skeleton_sequence
