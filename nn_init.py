from loss_function import *

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_tensor_type(torch.DoubleTensor)

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_nodes, hidden_layers, output_nodes):
        super(MultiLayerPerceptron, self).__init__()
        self.layer_dim = [input_nodes] + hidden_layers + [output_nodes]
        self.nn_layers = []

        self._connect_neurons(self.layer_dim)

    #def _connect_neurons(self, layer_sequence):


    def _connect_neurons(self, layer_sequence):
        last_element = len(layer_sequence)-1
        tmp = []
        try:
            for element in range(len(layer_sequence)-1):
                if element == last_element:
                    tmp.append(
                        nn.Sequential(
                            nn.BatchNorm1d(num_features=layer_sequence[element], affine=False),
                            nn.Linear(layer_sequence[element], layer_sequence[element + 1]),
                        ))
                else:
                    tmp.append(
                        nn.Sequential(
                            nn.Linear(layer_sequence[element], layer_sequence[element+1]),
                            nn.Sigmoid(),
                            nn.BatchNorm1d(num_features=layer_sequence[element + 1], affine=False)
                    ))

        except Exception as CN:
            print("Error connecting neurons:\n" + str(CN))
        finally:
            self.nn_layers = nn.ModuleList(tmp)
            print(self.nn_layers)


    def forward(self, input):
        for layer in self.nn_layers:
            output = layer(input)
            input = output
        return output





class NeuralNetwork(nn.Module):
    def __init__(self, input_nodes, hidden_layers, output_nodes):
        super(NeuralNetwork, self).__init__()

        self.model1 = MultiLayerPerceptron(input_nodes, hidden_layers, output_nodes).double()
        self.model2 = MultiLayerPerceptron(input_nodes, hidden_layers, output_nodes).double()

        self.loss = LossFunction().loss

    def forward(self, x, y):
        output_x = self.model1(x)
        output_y = self.model2(y)

        return output_x, output_y


