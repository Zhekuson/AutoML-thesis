from abc import ABC, abstractmethod


class ResearchDataset(ABC):
    def __init__(self, *args, **kwargs):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    @abstractmethod
    def load(self, *args, **kwargs):
        pass


