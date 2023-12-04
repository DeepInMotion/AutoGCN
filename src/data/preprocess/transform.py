import numpy as np
import math
from tqdm import tqdm
import logging
import os.path as osp


class Transform:
    def __init__(self, sh_l, sh_r, hip_l, hip_r, spine, spine_m, spine_b):
        self.N, self.C, self.T, self.V, self.M = None, None, None, None, None
        self.shoulder_l_idx = sh_l
        self.shoulder_r_idx = sh_r
        self.hip_l_idx = hip_l
        self.hip_r_idx = hip_r
        self.spine_idx = spine
        self.spine_middle_idx = spine_m
        self.spine_base_idx = spine_b

        self.is_sorted = lambda a: np.all(a[:-1] <= a[1:])

        # Thresholds
        self.thresh_length = 15
        self.thresh_interpolate = 10

    def _trans_scale_normalization(self, s):
        """
        Scale normalization.
        j′_k(n) k − j_1′(n) / d_1,2 = ˆj(n) k − j′_1(n) / 1
        d1,2 = 1 / N ∑ √ (x ′ (n) 1 −x′(n) 2 )^2 +(y′(n) 1 −y′(n) 2 )^2 +(z′(n) 1 −z′(n) 2 )^2.
        ˆj(n) k=1 d1,2 j′(n) k + (1 − 1 d1,2 )j′(n) 1 .
        :param s:
        :return:
        """
        # prog_bar = mmcv.ProgressBar(len(s))
        data_new = np.zeros(s.shape)
        # overall mean
        average_dist_total = np.zeros([self.N, self.M])
        for i_s, skeleton in enumerate(tqdm(s)):
            for i_p, person in enumerate(skeleton):
                average_dist = np.zeros(self.T)
                for i_f, frame in enumerate(person):
                    x_ = (frame[self.spine_idx, 0] - frame[self.spine_base_idx, 0]) ** 2
                    y_ = (frame[self.spine_idx, 1] - frame[self.spine_base_idx, 1]) ** 2
                    z_ = (frame[self.spine_idx, 2] - frame[self.spine_base_idx, 2]) ** 2
                    # times 4 so values do not get too large
                    average_dist[i_f] = np.sqrt(x_ + y_ + z_) * 4
                average_dist_total[i_s, i_p] = (1 / self.T) * np.sum(average_dist)
            # prog_bar.update()
        # mean over all skeletons
        # throw out 0 values for empty skeletons
        average_dist_total[average_dist_total == 0] = np.nan
        average_dist_mean = np.mean(np.hstack(average_dist_total[~np.isnan(average_dist_total)]))

        # prog_bar = mmcv.ProgressBar(len(s))
        for i_s, skeleton in enumerate(tqdm(s)):
            for i_p, person in enumerate(skeleton):
                scale_alig_frame = np.zeros([25, 3])
                for i_f, frame in enumerate(person):
                    for i_j, joint in enumerate(frame):
                        xyz = (1 / average_dist_mean) * joint + \
                              (1 - (1 / average_dist_mean)) * frame[self.spine_base_idx]
                        scale_alig_frame[i_j] = xyz
                    data_new[i_s, i_p, i_f, ...] = scale_alig_frame
            # prog_bar.update()
        return data_new

    def _trans_view_normalization(self, s):
        """
        View alignment implementation.
        R= [ v1 ‖v1‖ ∣∣∣∣ v2 − Projv1 (v2) ‖v2 − Projv1 (v2)‖ ∣∣∣∣ v1 × v2 ‖v1 × v2‖]
        oR = (sH L t=0 + sH R t=0 )/2,
        :param s:
        :return:
        """
        # prog_bar = mmcv.ProgressBar(len(s))
        data_new = np.zeros(s.shape)
        null_skeleton = False
        for i_s, skeleton in enumerate(tqdm(s)):
            for i_p, person in enumerate(skeleton):
                # o_average = - 1.0 / person.shape[0] * np.sum(person[:, self.base_bone[1]], axis=0)
                # principal_M = 0
                view_alig_frame = np.zeros([25, 3])
                for i_f, frame in enumerate(person):
                    if i_f == 0:
                        if frame.any() == 0:
                            # Null frame -> break to avoid RunTimeWarning
                            null_skeleton = True
                            break
                        # origin
                        new_origin_d = frame[self.hip_r_idx] + frame[self.hip_l_idx]
                        # v1 = x_t0_spine - x_t0_root
                        # spine_base_idx == 0, middle_spine_idx == 1
                        v1 = frame[self.spine_middle_idx] - frame[self.spine_base_idx]
                        try:
                            v1 = v1 / np.linalg.norm(v1)
                        except RuntimeWarning:
                            print(i_p, v1)
                        # v2 = x_t0_hip^_left - x_t0_hip_right
                        # middle_spine_idx == 12, hip_right_idx == 16
                        v2_ = frame[self.hip_l_idx] - frame[self.hip_r_idx]
                        proj_v2_v1 = np.dot(v1.T, v2_) * v1 / np.linalg.norm(v1)
                        v2 = v2_ - np.squeeze(proj_v2_v1)
                        v2 = v2 / np.linalg.norm(v2)

                        v3 = np.cross(v2, v1) / np.linalg.norm(np.cross(v2, v1))

                        v1 = np.reshape(v1, (3, 1))
                        v2 = np.reshape(v2, (3, 1))
                        v3 = np.reshape(v3, (3, 1))

                        R = np.hstack([v2, v3, v1])
                        null_skeleton = False
                    if null_skeleton:
                        # leave skeleton as is
                        continue
                    for i_j, joint in enumerate(frame):
                        xyz_alig = np.squeeze(np.matmul(np.linalg.inv(R), np.reshape(joint - new_origin_d, (3, 1))))
                        view_alig_frame[i_j] = xyz_alig

                    data_new[i_s, i_p, i_f, ...] = view_alig_frame
            # prog_bar.update()
        return data_new

    def _trans_remove_null(self, s, total_frames):
        """
        Remove the missing frames and interpolate when data is under certain threshold.
        1) Check null frames location
            - beginning, end, start
        2) Check threshold
        3) Check if all data is missing
        4) pad or remove
        :param s:
        :return:
        """
        mask = dict()
        for i_s, skeleton in tqdm(enumerate(s)):
            for i_p, person in enumerate(skeleton):
                null_list = []
                for i_f, frame in enumerate(person):
                    # print(i_f, frame)
                    if np.all(frame == 0) and i_f < total_frames[i_s]:
                        # append index
                        null_list.append(i_f)
                    if i_f > total_frames[i_s]:
                        break

                if len(null_list) > 0:
                    # focus on valid data
                    # check location
                    if self.is_sorted(null_list):
                        # only beginning and end
                        if len(null_list) == total_frames[i_f]:
                            # remove whole skeleton
                            mask[i_s] = [i_p, [-1]]
                        if null_list[0] == 0:
                            # beginning

                            mask[i_f] = [i_p, ]
                            start = null_list[0]
                            end = null_list[-1]
                            mask = []
                            s = s[i_s, i_p, i_f]
                            pass
                        elif null_list[-1] == total_frames[i_s]:
                            # end
                            pass
                    else:
                        # values in between are missing
                        pass

                if len(null_list) > 0:
                    # delete frames bigger than total frames
                    null_list = null_list[0:total_frames[i_s]]
                    if self.is_sorted(null_list):
                        if len(null_list) < self.thresh_length:
                            # below threshold and sorted
                            for i_c in null_list:
                                s = np.delete(s, [i_s, i_p, i_c])
                            total_frames[i_s] -= len(null_list)

        return s, total_frames

    def _trans_noise(self, s):
        """
        Filter the noise from the skeleton sequence.
        Thresholds in init
        1) Denoise by frame length
        2) Denoise by spread of X and Y
        3) Denoise by motion
        :param s:
        :return:
        """
        # TODO
        pass

    def _trans_parallel_hip(self, s):
        """
        Parallel the bone between hip(jpt 0) and spine(jpt 1) of the first person to the z axis
        :param s: transposed data [N, M, T, V, C]
        :return:
        """
        orientation = [0, 1, 0]
        # prog_bar = mmcv.ProgressBar(len(s))
        for i_s, skeleton in enumerate(tqdm(s)):
            if skeleton.sum() == 0:
                continue
            joint_bottom = skeleton[0, 0, self.spine_base_idx]
            joint_top = skeleton[0, 0, self.spine_middle_idx]
            axis = np.cross(joint_top - joint_bottom, orientation)
            angle = self._angle_between(joint_top - joint_bottom, orientation)
            matrix = self._rotation_matrix(axis, angle)
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0:
                    continue
                for i_f, frame in enumerate(person):
                    if frame.sum() == 0:
                        continue
                    for i_j, joint in enumerate(frame):
                        s[i_s, i_p, i_f, i_j] = np.dot(matrix, joint)
            # prog_bar.update()
        return s

    def _trans_parallel_shoulder(self, s):
        """
        Parallel the bone between right shoulder(jpt 8) and left shoulder(jpt 4) of the first person to the x axis
        :param s:
        :return:
        """
        orientation = [1, 0, 0]
        # prog_bar = mmcv.ProgressBar(len(s))
        for i_s, skeleton in enumerate(tqdm(s)):
            if skeleton.sum() == 0:
                continue
            joint_rshoulder = skeleton[0, 0, self.shoulder_r_idx]
            joint_lshoulder = skeleton[0, 0, self.shoulder_l_idx]
            axis = np.cross(joint_rshoulder - joint_lshoulder, orientation)
            angle = self._angle_between(joint_rshoulder - joint_lshoulder, orientation)
            matrix = self._rotation_matrix(axis, angle)
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0:
                    continue
                for i_f, frame in enumerate(person):
                    if frame.sum() == 0:
                        continue
                    for i_j, joint in enumerate(frame):
                        s[i_s, i_p, i_f, i_j] = np.dot(matrix, joint)
            # prog_bar.update()
        return s

    def _trans_sub_center(self, s):
        """
        Sub the center joint #1 (spine joint in ntu and neck joint in kinetics)
        :param s: transposed data N M T V C
        :return: data
        """
        # prog_bar = mmcv.ProgressBar(len(s))
        for i_s, skeleton in enumerate(tqdm(s)):
            if skeleton.sum() == 0:
                continue
            # take first value to secure the orientation of the second skeleton
            main_body_center = skeleton[0][0, 1:2, :].copy()
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0:
                    continue
                mask = (person.sum(-1) != 0).reshape(self.T, self.V, 1)
                s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask
            # prog_bar.update()
        return s

    @staticmethod
    def _trans_pad_null(s):
        """
        Pad the null frames with the previous frames.
        Repeat if smaller than max_frame and move to beginning.
        :param s: transposed data [N, M, T, V, C]
        :return: data
        """
        # prog_bar = mmcv.ProgressBar(len(s))
        for i_s, skeleton in enumerate(tqdm(s)):
            if skeleton.sum() == 0:
                print(i_s, ' has no skeleton')
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0:
                    continue
                if person[0].sum() == 0:
                    index = (person.sum(-1).sum(-1) != 0)
                    tmp = person[index].copy()
                    person *= 0
                    person[:len(tmp)] = tmp

                for i_f, frame in enumerate(person):
                    if frame.sum() == 0:
                        if person[i_f:].sum() == 0:
                            rest = len(person) - i_f
                            num = int(np.ceil(rest / i_f))
                            pad = np.concatenate(
                                [person[0:i_f] for _ in range(num)], 0)[:rest]
                            s[i_s, i_p, i_f:] = pad
                            break
            # prog_bar.update()
        return s

    @staticmethod
    def _angle_between(v1, v2):
        """
        Returns the angle in radians between vectors 'v1' and 'v2'
        :param v1:
        :param v2:
        :return:
        """
        offset = 1e-6
        if np.abs(v1).sum() < offset or np.abs(v2).sum() < offset:
            return 0
        v1_u = v1 / np.linalg.norm(v1)  # unit_vector
        v2_u = v2 / np.linalg.norm(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    @staticmethod
    def _rotation_matrix(axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation
        about the given axis by theta radians
        :param axis:
        :param theta:
        :return:
        """
        offset = 1e-6
        if np.abs(axis).sum() < offset or np.abs(theta) < offset:
            return np.eye(3)
        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


class TransformKinetics(Transform):
    """
    Kinetics Transform class
    """
    def __init__(self, out_path):
        # C == channels, M == skeletons, N == number of skeletons, V == vertices, T == frame
        super().__init__(sh_l=2, sh_r=5, hip_l=8, hip_r=11, spine=None, spine_m=None, spine_b=None)
        self.N, self.C, self.T, self.V, self.M = None, None, None, None, None

        # logging
        self.transform_logger = logging.getLogger('kinetics_transform_log')
        self.transform_logger.setLevel(logging.INFO)
        self.transform_logger.addHandler(logging.FileHandler(osp.join(out_path, 'kinetics_transform_log.log')))

    def transform(self, data: np.array, options: list) -> np.array:
        """
        Transform the Skeleton data depending on the options which are set.
        :return: transformed data [N, C, T, V, M]
        """
        assert len(data.shape) == 5
        self.N, self.C, self.T, self.V, self.M = data.shape
        data = np.transpose(data, [0, 4, 2, 3, 1])
        # pop_list = []
        for opt in options:
            if opt == 'pad':
                print('Padding NULL values.')
                data = self._trans_pad_null(data)
            elif opt == 'sub':
                print('Subbing center.')
                data = self._trans_sub_center(data)
            elif opt == 'parallel_s':
                print('Paralleling shoulders.')
                data = self._trans_parallel_shoulder(data)
            elif opt == 'parallel_h':
                print('Paralleling hips.')
                data = self._trans_parallel_hip(data)
            elif opt == 'view':
                print('Normalizing view.')
                data = self._trans_view_normalization(data)
            elif opt == 'scale':
                print('Normalizing scale.')
                data = self._trans_scale_normalization(data)
            elif opt == '':
                raise Warning('No transform option specified!')
            else:
                raise ValueError('Transform option {} not known!'.format(opt))

        # data = self._trans_remove_null(data, total_frames)
        # data = self._trans_view_normalization(data)
        # eventually delete corrupt or missing data

        transformed_data = np.transpose(data, [0, 4, 2, 3, 1])

        return transformed_data


class TransformNTU(Transform):
    """
    NTU Transform class - in case of complete deletion give info to gendata class.
    """
    def __init__(self, out_path):
        # C == channels, M == skeletons, N == number of skeletons, V == vertices, T == frame
        super().__init__(sh_l=4, sh_r=8, hip_l=12, hip_r=16, spine=1, spine_m=20, spine_b=0)
        self.N, self.C, self.T, self.V, self.M = None, None, None, None, None

        # logging
        self.transform_logger = logging.getLogger('ntu_transform_log')
        self.transform_logger.setLevel(logging.INFO)
        self.transform_logger.addHandler(logging.FileHandler(osp.join(out_path, 'ntu_transform_log.log')))

    def transform(self, data: np.array, options: list) -> np.array:
        """
        Transform the Skeleton data depending on the options which are set.
        :return: transformed data [N, C, T, V, M]
        """
        assert len(data.shape) == 5
        self.N, self.C, self.T, self.V, self.M = data.shape
        data = np.transpose(data, [0, 4, 2, 3, 1])
        # pop_list = []
        for opt in options:
            if opt == 'pad':
                print('Padding NULL values.\n')
                data = self._trans_pad_null(data)
            elif opt == 'sub':
                print('Subbing center.\n')
                data = self._trans_sub_center(data)
            elif opt == 'parallel_s':
                print('Paralleling shoulders.\n')
                data = self._trans_parallel_shoulder(data)
            elif opt == 'parallel_h':
                print('Paralleling hips.\n')
                data = self._trans_parallel_hip(data)
            elif opt == 'view':
                print('Normalizing view.\n')
                data = self._trans_view_normalization(data)
            elif opt == 'scale':
                print('Normalizing scale.\n')
                data = self._trans_scale_normalization(data)
            elif opt == '':
                raise Warning('No transform option specified!')
            else:
                raise ValueError('Transform option {} not known!'.format(opt))

        # eventually delete corrupt or missing data
        data = np.array(data, dtype=np.float32)
        transformed_data = np.transpose(data, [0, 4, 2, 3, 1])

        return transformed_data
