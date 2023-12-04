import random


class ReplayMemory:
    def __init__(self, capacity: int, threshold: float):
        self.capacity = capacity
        self.memory = []
        self.threshold = threshold

    def push(self, rollout: list) -> int:
        counter = 0
        for r in rollout:
            if r[2]['acc_top1'] > self.threshold:
                self.memory.append(r)
                counter += 1
            if self.memory.__len__() > self.capacity:
                self.memory.pop(0)
        return counter

    def sample(self, batch_size: int):
        if self.memory.__len__() <= batch_size:
            return random.sample(self.memory, self.memory.__len__())
        else:
            return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
