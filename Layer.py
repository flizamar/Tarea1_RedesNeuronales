class Layer():
  def __init__(self, num_perceptrons, n_inputs_perceptron):
    self.perceptrons = [0]*num_perceptrons
    for i in range(num_perceptrons):
      self.perceptrons[i] = SigmoidNeuron(n_inputs_perceptron)
      
  def error_purple(self, n):
    suma = 0
    for perc in self.perceptrons:
      suma += perc.weights[n] * perc.delta_green
    return suma
