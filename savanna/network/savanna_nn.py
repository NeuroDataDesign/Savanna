import abc

class Savanna_nn(metaclass=abc.ABCMeta):

    def __init__(self, pytorch_nn):
        self.pytorch_nn = pytorch_nn
        self.type = "savanna_nn"

    @abc.abstractmethod
    def fit(self, input, labels):
        return


    def predict(self, input):
        outputs = self.pytorch_nn(input)
        return outputs

    @abc.abstractmethod
    def final_predict(self, input):
        return
