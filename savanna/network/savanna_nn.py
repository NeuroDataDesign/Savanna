import abc

class savanna_nn(metaclass=abc.ABCMETA):

    def __init__(self, pytorch_nn):
        self.pytorch_nn = pytorch_nn
        self.type = "savanna_nn"

    @abc.abstractmethod
    def fit(self, input):
        return

    @abc.abstractmethod
    def predict(self, input):
        return
