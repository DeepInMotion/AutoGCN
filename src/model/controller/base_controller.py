from abc import ABC, abstractmethod


class BaseController(ABC):

    def __init__(self, **kwargs):
        super(BaseController, self).__init__(**kwargs)

    @abstractmethod
    def has_converged(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def sample(self, arch_computations, hyper_computations):
        raise NotImplementedError

    @abstractmethod
    def policy_argmax(self, arch_computations, hyper_computations):
        raise NotImplementedError

    @abstractmethod
    def update(self, reward_signal):
        raise NotImplementedError
