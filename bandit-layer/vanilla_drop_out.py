import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import math
import random
# Define the model
class VanillaDropOutLayer(nn.Module):
    def __init__(self, input_dim, output_dim, p=1):
        super(VanillaDropOutLayer, self).__init__()
        torch.device("cpu") 
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.p = p
        
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Apply dropout to the input
        #input = nn.functional.dropout(input, p=self.p, training=self.training)
        
        m, n = input.size(0), self.weight.size(0)

        # Initialize the output matrix
        output = torch.zeros((m, n))
        for j in range(n):
            for i in range(m):
                if (self.training and random.random() < self.p):
                    output[i, j] = torch.dot(input[i], self.weight[j]) + self.bias[j]
                else:
                    output[i, j] = torch.dot(input[i], self.weight[j]) + self.bias[j] 
        return output
