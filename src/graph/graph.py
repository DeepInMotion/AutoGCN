import numpy as np
import torch
import torch.nn as nn
import logging


class Graph:
    """ The Graph to model the skeletons extracted by the openpose
    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).
        layout (string): must be one of the follow candidates
        - openpose: Consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D
        - in-motion: Consists of 19 joints ('head_top', 'nose', 'right_ear', 'left_ear', 'upper_neck', 'right_shoulder',
            'right_elbow', 'right_wrist', 'thorax', 'left_shoulder', 'left_elbow', 'left_wrist', 'pelvis', 'right_hip',
            'right_knee', 'right_ankle', 'left_hip', 'left_knee', and 'left_ankle').
        - in-motion-2022: Consists of 29 joints ('head_top', 'nose', 'right_ear', 'left_ear', 'upper_neck',
            'right_shoulder', 'right_elbow', 'right_wrist', 'thorax', 'left_shoulder', 'left_elbow', 'left_wrist',
            'pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle',
            'right_little_finger', 'right_index_finger', 'left_little_finger', 'left_index_finger',
            'right_heel', 'right_little_toe', 'right_big_toe', 'left_heel', 'left_little_toe' and 'left_big_toe').
        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points
        disentangled_num_scales (int): number of disentangled adjacency matrices,
        including hops distance from 0 to disentangled_num_scales-1
        use_mask (bool): If ``True``, adds a residual mask to the edges of the graph
    """

    def __init__(self,
                 layout='ntu',
                 strategy='spatial',
                 max_hop=10,
                 dilation=1,
                 disentangled_num_scales=7,
                 use_mask=True):

        self.max_hop = max_hop
        self.dilation = dilation
        self.layout = layout
        self.strategy = strategy
        self.disentangled_num_scales = disentangled_num_scales
        self.use_mask = use_mask

        logging.info("Generate Graph data in the [{}] layout with [{}]  strategy".format(self.layout, self.strategy))
        self.num_nodes, self.neighbor_link, self.center, self.edge, self.connect_joint, self.parts = self._get_edge()
        self.hop_dis = self.get_hop_distance(self.num_nodes, self.edge, max_hop=max_hop)
        # self.get_adjacency(strategy)
        self.A = self._get_adjacency()

    def __str__(self):
        return self.A

    def _get_edge(self):
        parts = None
        if self.layout == 'openpose':
            num_nodes = 18
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            center = 1
            connect_joint = np.array([1, 1, 1, 2, 3, 1, 5, 6, 2, 8, 9, 5, 11, 12, 0, 0, 14, 15])  # As in Pa-Res-GCN
        elif self.layout == 'ntu-rgbd':
            num_nodes = 25
            neighbor_link = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                             (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                             (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                             (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                             (22, 23), (23, 8), (24, 25), (25, 12)]
            center = 21 - 1
            connect_joint = np.array(
                [2, 2, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 23, 8, 25, 12]) - 1
            parts = [
                np.array([5, 6, 7, 8, 22, 23]) - 1,  # left_arm
                np.array([9, 10, 11, 12, 24, 25]) - 1,  # right_arm
                np.array([13, 14, 15, 16]) - 1,  # left_leg
                np.array([17, 18, 19, 20]) - 1,  # right_leg
                np.array([1, 2, 3, 4, 21]) - 1  # torso
            ]

        elif self.layout == 'ntu':
            num_nodes = 25
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            center = 21 - 1
            connect_joint = np.array([2,2,21,3,21,5,6,7,21,9,10,11,1,13,14,15,1,17,18,19,2,23,8,25,12]) - 1
            parts = [
                np.array([5, 6, 7, 8, 22, 23]) - 1,     # left_arm
                np.array([9, 10, 11, 12, 24, 25]) - 1,  # right_arm
                np.array([13, 14, 15, 16]) - 1,         # left_leg
                np.array([17, 18, 19, 20]) - 1,         # right_leg
                np.array([1, 2, 3, 4, 21]) - 1          # torso
            ]
        elif self.layout == 'ntu_edge':
            num_nodes = 24
            neighbor_link = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                             (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                             (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                             (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                             (23, 24), (24, 12)]
            center = 2
            # Need to verify connections
            connect_joint = np.array(
                [2, 2, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 23, 8, 25, 12]) - 1
            parts = [
                np.array([5, 6, 7, 8, 22, 23]) - 1,  # left_arm
                np.array([9, 10, 11, 12, 24, 25]) - 1,  # right_arm
                np.array([13, 14, 15, 16]) - 1,  # left_leg
                np.array([17, 18, 19, 20]) - 1,  # right_leg
                np.array([1, 2, 3, 4, 21]) - 1  # torso
            ]
        elif self.layout == 'in-motion' or self.layout == 'kinetics-in-motion' or self.layout == 'ntu-in-motion':
            '''
            0	HeadTop
            1	Nose
            2	REar
            3	LEar
            4	UpperNeck
            5	RShoulder
            6	RElbow
            7	RWrist
            8	Thorax
            9	LShoulder
            10	LElbow
            11	LWrist
            12	Pelvis
            13	RHip
            14	RKnee
            15	RAnkle
            16	LHip
            17	LKnee
            18	LAnkle
            '''
            num_nodes = 19
            neighbor_link = [(0, 1), (2, 1), (3, 1), (1, 4), (9, 8),
                             (10, 9), (11, 10), (5, 8), (6, 5), (7, 6),
                             (4, 8), (12, 8), (16, 12), (17, 16), (18, 17),
                             (13, 12), (14, 13), (15, 14)]
            center = 8  # thorax center
            connect_joint = np.array([1, 4, 1, 1, 8, 8, 5, 6, 8, 8, 9, 10, 8, 12, 13, 14, 12, 16,
                                   17])  # Bones with 8 (thorax) as center and shoulder connected to thorax
        elif self.layout == 'in-motion-2022':
            """
            29 - point
            0	HeadTop
            1	Nose
            2	REar
            3	LEar
            4	UpperNeck
            5	RShoulder
            6	RElbow
            7	RWrist
            8	Thorax
            9	LShoulder
            10	LElbow
            11	LWrist
            12	Pelvis
            13	RHip
            14	RKnee
            15	RAnkle
            16	LHip
            17	LKnee
            18	LAnkle
            19	RLittleFinger
            20	RIndexFinger
            21	LLittleFinger
            22	LIndexFinger
            23	RHeel
            24	RLittleToe
            25	RBigToe
            26	LHeel
            27	LLittleToe
            28	LBigToe
            """
            num_nodes = 29
            neighbor_link = [(0, 1), (2, 1), (3, 1), (1, 4), (9, 8),
                             (10, 9), (11, 10), (5, 8), (6, 5), (7, 6),
                             (4, 8), (12, 8), (16, 12), (17, 16), (18, 17),
                             (13, 12), (14, 13), (15, 14), (19, 7), (20, 7), (21, 11), (22, 11), (23, 15),
                             (24, 15), (25, 15), (26, 18), (27, 18), (28, 18)]
            center = 8  # thorax center
            connect_joint = np.array(
                [1, 4, 1, 1, 8, 8, 5, 6, 8, 8, 9, 10, 8, 12, 13, 14, 12, 16, 17, 7, 7, 11, 11, 15, 15, 15, 18, 18,
                 18])  # Bones with 8 (thorax) as center and shoulder connected to thorax
        # Intersection pose between openpose and in-motion, for transfer learning
        elif self.layout == 'kinetics':
            """
            0	Nose 1	Neck 2	RShoulder 3	RElbow 4	RWrist 5	LShoulder 6	LElbow 7	LWrist 8	RHip 9	RKnee 10
            RAnkle 11	LHip 12	LKnee 13 LAnkle
            {14, "REye"}, # {15, "LEye"}, # {16, "REar"}, # {17, "LEar"}
            """
            num_nodes = 18
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14), (8, 11)]
            connect_joint = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])
            parts = [
                np.array([5, 6, 7]),              # left_arm
                np.array([2, 3, 4]),              # right_arm
                np.array([11, 12, 13]),           # left_leg
                np.array([8, 9, 10]),             # right_leg
                np.array([0, 1, 14, 15, 16, 17])  # torso
            ]
            # num_nodes = 18
            # neighbor_link = [(0, 1), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9), (11, 5),
            #                  (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15)]
            # neighbor_link = [(14, 0), (15, 0), (0, 1), (5, 1), (6, 5),
            #                  (7, 6), (2, 1), (3, 2), (4, 3), (11, 1),
            #                  (12, 11), (13, 12), (8, 1), (9, 8), (10, 9)]
            center = 1  # neck/thorax
            # connect_joint = np.array([1, 1, 1, 2, 3, 1, 5, 6, 1, 8, 9, 1, 11, 12, 0, 0])
        elif self.layout == 'ntu_red':
            '''
            0	Spine base
            1	Neck
            2	Head
            3	Lshoulder
            4	Lelbow
            5	Lwrist
            6	Rshoulder
            7	Relbow
            8	Rwrist
            9	Lhip
            10	Lknee
            11	Lankle
            12	Rhip
            13	Rknee
            14	Rankle
            15	Spine
            '''
            num_nodes = 16
            neighbor_link = [(2, 1), (1, 15), (3, 15), (4, 3), (5, 4),
                             (6, 15), (7, 6), (8, 7), (0, 15), (9, 0),
                             (10, 9), (11, 10), (12, 0), (13, 12), (14, 13)]
            center = 15  # spine/thorax
            connect_joint = np.array([15, 15, 1, 15, 3, 4, 15, 6, 7, 0, 9, 10, 0, 12, 13])
        elif self.layout == 'ntu-in-motion-reduced':
            num_nodes = 16
            neighbor_link = [(2, 1), (1, 15), (6, 15), (7, 6), (8, 7), (3, 15), (4, 3), (5, 4), (0, 15), (12, 0),
                             (13, 12), (14, 13), (9, 0), (10, 9), (11, 10)]
            center = 15  # spine/thorax
            connect_joint = np.array([15, 15, 1, 15, 3, 4, 15, 6, 7, 0, 9, 10, 0, 12, 13, 15])
        elif self.layout == 'kinetics-in-motion-reduced':
            num_nodes = 16
            neighbor_link = [(1, 0), (0, 2), (0, 6), (3, 6), (4, 3), (5, 4), (6, 7), (7, 8), (8, 9), (6, 10),
                             (10, 11), (11, 12), (6, 13), (13, 14), (14, 15)]
            center = 6  # spine/thorax
            connect_joint = np.array([6, 0, 0, 6, 3, 4, 6, 6, 7, 8, 6, 10, 11, 6, 13, 14])
        else:
            logging.info("")
            raise ValueError("Layout {} is not implemented!".format(self.layout))

        self_link = [(i, i) for i in range(num_nodes)]
        edge = self_link + neighbor_link

        return num_nodes, neighbor_link, center, edge, connect_joint, parts

    def _get_hop_distance(self):
        A = np.zeros((self.num_nodes, self.num_nodes))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_nodes, self.num_nodes)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_adjacency(self):
        hop_dis = self._get_hop_distance()
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_nodes, self.num_nodes))
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1
        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_nodes, self.num_nodes))
        for i, hop in enumerate(valid_hop):
            A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
        return A

    def _normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_nodes, self.num_nodes))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self.normalize_digraph(adjacency)
        if strategy == 'uniform':
            A = np.zeros((1, self.num_nodes, self.num_nodes))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_nodes, self.num_nodes))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        # https://github.com/yysijie/st-gcn/blob/master/net/utils/graph.py
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_nodes, self.num_nodes))
                a_close = np.zeros((self.num_nodes, self.num_nodes))
                a_further = np.zeros((self.num_nodes, self.num_nodes))
                for i in range(self.num_nodes):
                    for j in range(self.num_nodes):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        # https://github.com/kenziyuliu/MS-G3D/blob/master/graph/tools.py
        elif strategy == 'disentangled':
            outward = [(j, i) for (i, j) in self.neighbor_link]
            edges = self.neighbor_link + outward
            A_binary = self.get_adjacency_matrix(edges, self.num_nodes)
            # A_binary_with_I = self.get_adjacency_matrix(edges + self.self_link, self.num_nodes)
            A_powers = [self.k_adjacency(A_binary, k, with_self=True) for k in range(self.disentangled_num_scales)]
            A_powers = np.concatenate([self.normalize_adjacency_matrix(g) for g in A_powers])
            self.A = torch.Tensor(A_powers)
            if self.use_mask:
                # NOTE: the inclusion of residual mask appears to slow down training noticeably
                self.A_res = nn.init.uniform_(nn.Parameter(torch.Tensor(self.A.shape)), -1e-6, 1e-6)
                self.A = self.A + self.A_res
            self.A = self.A.view(self.disentangled_num_scales, self.num_nodes, self.num_nodes)
        else:
            raise ValueError("The strategy {} is not implemented!".format(strategy))

    def get_neck_hip_indexes(self):
        """
        Used for finding the spine which can be used for rotating the body
        @return:
        """
        if self.layout == "in-motion" or self.layout == 'in-motion-2022' \
                or self.layout == 'kinetics-in-motion' or self.layout == 'ntu-in-motion':
            neck_index = 8
            # Since in-motion has the pelvis as a joint, we use this for a more precise center-point
            left_hip_index = 12
            right_hip_index = 12
        elif self.layout == "kinetics":
            neck_index = 1
            left_hip_index = 8
            right_hip_index = 11
        elif self.layout == 'ntu':
            neck_index = 15
            left_hip_index = 0
            right_hip_index = 0
        elif self.layout == 'ntu-in-motion-reduced':
            neck_index = 15
            left_hip_index = 0
            right_hip_index = 0
        elif self.layout == 'kinetics-in-motion-reduced':
            neck_index = 6
            left_hip_index = 13
            right_hip_index = 10
        else:
            raise ValueError("Layout {} is not implemented!".format(self.layout))
        return neck_index, left_hip_index, right_hip_index

    @staticmethod
    def get_hop_distance(num_nodes, edge, max_hop=1):
        A = np.zeros((num_nodes, num_nodes))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1
        # compute hop steps
        hop_dis = np.zeros((num_nodes, num_nodes)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    @staticmethod
    def normalize_digraph(A):
        Dl = np.sum(A, 0)
        num_nodes = A.shape[0]
        Dn = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)
        return AD

    @staticmethod
    def normalize_undigraph(A):
        Dl = np.sum(A, 0)
        num_nodes = A.shape[0]
        Dn = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-0.5)
        DAD = np.dot(np.dot(Dn, A), Dn)
        return DAD

    @staticmethod
    def get_adjacency_matrix(edges, num_nodes):
        A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for edge in edges:
            A[edge] = 1.
        return A

    @staticmethod
    def k_adjacency(A, k, with_self=False, self_factor=1):
        assert isinstance(A, np.ndarray)
        I = np.eye(len(A), dtype=A.dtype)
        if k == 0:
            return I
        Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
             - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
        if with_self:
            Ak += (self_factor * I)
        return Ak

    @staticmethod
    def normalize_adjacency_matrix(A):
        node_degrees = A.sum(-1)
        degs_inv_sqrt = np.power(node_degrees, -0.5)
        norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
        return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)
