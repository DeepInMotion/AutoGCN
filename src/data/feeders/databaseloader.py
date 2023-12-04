from abc import ABC, abstractmethod


class DataBaseLoader(ABC):

    def __init__(self, phase: str, data_path: str, inputs: dict, conn: list, frames: int, args,
                 debug=False, normalize=True, augment=False):
        self.args = args
        self.T = frames
        self.inputs = inputs
        self.conn = conn
        self.mean_map = None
        self.std_map = None
        self.normalize = normalize
        self.augment = augment
        self.debug = debug

        self.data_path = '{}/{}_data.npy'.format(data_path, phase)
        self.label_path = '{}/{}_label.pkl'.format(data_path, phase)

        self.load_data()
        if self.normalize:
            self.mean_std()

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def mean_std(self):
        pass

    @abstractmethod
    def top_k(self, score, top_k):
        pass




