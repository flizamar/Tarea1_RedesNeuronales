import numpy as np
class Perceptron():
    def __init__(self, n):
        self.weights = 4*np.random.rand(n) - 2
        self.bias = 4*np.random.rand() - 2

    def feed(self, inputs):
        inputs = np.array(inputs)
        return 1 if inputs.dot(self.weights) + self.bias > 0 else 0
      
    def train(self, desiredOutput, inputs):
        inputs = np.array(inputs)
        diff = desiredOutput - self.feed(inputs)
        lr = 0.1
        self.weights = self.weights + (lr*diff*inputs)
        self.bias = self.bias + (lr*diff)
    
    def transferDerivative(self, out):
        return out*(1.0 - out)
      
class SigmoidNeuron(Perceptron):
      def feed(self, inputs):
        inputs = np.array(inputs)
        val = inputs.dot(self.weights) + self.bias
        self.output = np.exp(val)/(1 + np.exp(val))
        return self.output


