import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import math

from torchBME import BME as BoundedME

# Define the model
class BanditLayer(nn.Module):
    def __init__(self, input_dim, output_dim, 
                 k=0.5, epsilon = 0.3, delta = 0.1,
                 weights_are_query = True):
        super(BanditLayer, self).__init__()
        
        self.k = k
        self.epsilon = epsilon
        self.delta = delta
        self.weights_are_query = weights_are_query
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if not self.training:
            return torch.matmul(input, self.weight.t()) + self.bias
        
        # input = input.view(input.size(0), -1)  # Flatten the input tensor
        input_size, weight_size = input.size(0), self.weight.size(0)

        # Initialize the output matrix
        output = torch.zeros((input_size, weight_size))
        ActiveSetSize = math.ceil(self.k * input_size)
        if self.weights_are_query: 
            for j in range(weight_size):
                query_vector = self.weight[j]
                ApproximateActiveSet,_ = \
                    BoundedME(input, query_vector, ActiveSetSize, self.epsilon, self.delta)
                for i in ApproximateActiveSet:
                    output[i, j] = torch.dot(input[i], self.weight[j]) + self.bias[j]
        else: # input is query

            # TODO rethink - this doesn't work.
            for i in range(input_size):
                query_vector = input[i]
                ApproximateActiveSet,_ = \
                    BoundedME(self.weight, query_vector, ActiveSetSize, self.epsilon, self.delta)
                for j in ApproximateActiveSet:
                    output[i, j] = torch.dot(input[i], self.weight[j]) + self.bias[j]
        return output
