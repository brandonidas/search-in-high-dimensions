import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import math

from torchBME import BME as BoundedME

# Define the model
class BanditLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BanditLayer, self).__init__()
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
        m, n = input.size(0), self.weight.size(0)
        p = input.size(1)

        # Initialize the output matrix
        output = torch.zeros((m, n))
        for j in range(n):
          query_vector = self.weight[j]
          k = math.ceil(m/3)
          ApproximateActiveSet,_ = BoundedME(input, query_vector, k, 0.3,0.1)
          for i in ApproximateActiveSet:
            output[i, j] = torch.dot(input[i], self.weight[j]) + self.bias[j]

        return output

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn2d = nn.BatchNorm2d(16,affine=True)
        # batch norm is needed to reduce vanishing gradient. 
        self.relu = nn.ReLU()
        self.fc1 = BanditLayer(16 * 28 * 28, 10)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn2d(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Load the FashionMNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create an instance of the model
model = MyModel()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print the loss every 100 mini-batches
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100}')
            running_loss = 0.0

# Evaluation on the test set
model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

accuracy = total_correct / total_samples
print(f'Test Accuracy: {accuracy * 100}%')
