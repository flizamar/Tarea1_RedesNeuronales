class Red():
  def __init__(self, size_input, size_layers, size_output):
      #Primera capa
      self.first_layer = Layer(size_layers[0], size_input)
      self.layers = []
      #capas intermedias
      for i in range(1, len(size_layers)):
        self.layers.append(Layer(size_layers[i], size_layers[i - 1]))
      #capa de output
      self.output_layer = Layer(size_output, size_layers[-1])
  
  def feed(self, inputs):
      ans = np.zeros(len(self.first_layer.perceptrons))
      for ind, perceptron in enumerate(self.first_layer.perceptrons):
        ans[ind] = perceptron.feed(inputs)
      for i, layer in enumerate(self.layers):
        temp = np.zeros(len(layer.perceptrons))
        for j, perceptron in enumerate(layer.perceptrons):
          temp[j] = perceptron.feed(ans)
        ans = temp
      temp = np.zeros(len(self.output_layer.perceptrons))
      for ind, perceptron in enumerate(self.output_layer.perceptrons):
        temp[ind] = perceptron.feed(ans)
      ans = temp
      return ans
  
  def backpropagation(self, desiredOutput, output):
      assert len(desiredOutput) == len(output)
      error = desiredOutput - output
      for n, perceptron in enumerate(self.output_layer.perceptrons):
        perceptron.delta_green = error[n] * perceptron.transferDerivative(perceptron.output)
      if len(self.layers) > 0:
        for n, perceptron in enumerate(self.layers[-1].perceptrons):
          perceptron.delta_green = self.output_layer.error_purple(n) * perceptron.transferDerivative(perceptron.output)
        for i, capa in reversed(list(enumerate(self.layers[:-1]))):
          for n, perceptron in enumerate(capa.perceptrons):
            perceptron.delta_green = self.layers[i + 1].error_purple(n) * perceptron.transferDerivative(perceptron.output)
      for n, perceptron in enumerate(self.first_layer.perceptrons):
        if len(self.layers) > 0:
          perceptron.delta_green = self.layers[0].error_purple(n) * perceptron.transferDerivative(perceptron.output)
        else:
          perceptron.delta_green = self.output_layer.error_purple(n) * perceptron.transferDerivative(perceptron.output)
          
  def train(self, desiredOutput, inputs, lr): 
      self.backpropagation(desiredOutput, self.feed(inputs))
      for perc in self.first_layer.perceptrons:
        for i, w in enumerate(perc.weights):
          perc.weights[i] = w + (lr*perc.delta_green*inputs[i])
        perc.bias = perc.bias + (lr*perc.delta_green)
      if len(self.layers) > 0:
        for perc in self.layers[0].perceptrons:
          for i, w in enumerate(perc.weights):
            perc.weights[i] = w + (lr*perc.delta_green*self.first_layer.perceptrons[i].output)
          perc.bias = perc.bias + (lr*perc.delta_green)
        for i, capa in enumerate(self.layers[1:]):
          for perc in capa.perceptrons:
            for j, w in enumerate(perc.weights):
              perc.weights[j] = w + (lr*perc.delta_green*self.layers[i].perceptrons[j].output)
            perc.bias = perc.bias + (lr*perc.delta_green)
      for perc in self.output_layer.perceptrons:
        for i, w in enumerate(perc.weights):
          if len(self.layers) > 0:
            perc.weights[i] = w + (lr*perc.delta_green*self.layers[-1].perceptrons[i].output)
          else:
            perc.weights[i] = w + (lr*perc.delta_green*self.first_layer.perceptrons[i].output)
        perc.bias = perc.bias + (lr*perc.delta_green)
          
