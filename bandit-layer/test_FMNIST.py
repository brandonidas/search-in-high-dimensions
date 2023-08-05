import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import math

from bandit_layer import BanditLayer
from basic_linear_layer import BasicLinearLayer

class MyModel(nn.Module):
    def __init__(self, layer_for_comparison):
        super(MyModel, self).__init__()
        torch.device("cpu")

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.bn2d = nn.BatchNorm2d(16,affine=True)
        # batch norm is needed to reduce vanishing gradient. 
        self.relu = nn.LeakyReLU(0.02)
        self.fc1 = layer_for_comparison
        self.fc2 = nn.Linear(
            in_features=layer_for_comparison.output_dim, 
            out_features=10)
        
        self.loss_function = nn.CrossEntropyLoss()
        self.optimiser = torch.optim.Adam(self.parameters(), lr = 0.005) # optimize lr later

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn2d(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        return x

# Load the FashionMNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def training_and_testing_loop(layer_for_comparison, epochs = 1):
    results_dict = dict()

    # Create an instance of the model
    device = torch.device("cpu")
    torch.set_default_tensor_type(torch.FloatTensor)

    model = MyModel(layer_for_comparison)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epoch_prints = True
    epoch_print_interval = 1500
    early_prototyping_stop_flag = False
    early_prototyping_stop = 1000
    
    num_epochs = 1 # epochs
    # Record the starting time
    start_time = time.process_time()
    total_forwardprop_time = 0
    total_backprop_time = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            start_forward_prop_time = time.process_time()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            end_forward_prop_time = time.process_time()
            total_forwardprop_time += end_forward_prop_time - start_forward_prop_time


            # Backward pass and optimization
            start_backprop_time = time.process_time()
            loss.backward()
            optimizer.step()
            end_backprop_time = time.process_time()
            total_backprop_time += end_backprop_time - start_backprop_time

            running_loss += loss.item()

            # Print the loss every 100 mini-batches
            if epoch_prints and (i + 1) % epoch_print_interval == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100}')
                running_loss = 0.0
            if i == early_prototyping_stop and early_prototyping_stop_flag == True:
                print("----- early-prototyping stopping")
                break

    # Record the ending time
    end_time = time.process_time()
    elapsed_time = end_time - start_time

    print(f"Elapsed training time: {elapsed_time:.6f} process time")
    print(f"Forward-prop training time: {total_forwardprop_time:.6f} process time")
    print(f"Back-prop training time: {total_backprop_time:.6f} process time")
    
    results_dict["elapsed_time"] = elapsed_time
    results_dict["total_forwardprop_time"] = total_forwardprop_time
    results_dict["total_backprop_time"] = total_backprop_time


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
    print("chance is 6.75 %")
    results_dict["test_accuracy"] = accuracy

    return results_dict


comparison_layer_output_dim = 64
print("")
print("weights are query on bandit layer")
training_and_testing_loop(BanditLayer(10816, comparison_layer_output_dim, 0.1))
print("")
print("inputs are query on bandit layers")
training_and_testing_loop(BanditLayer(10816, comparison_layer_output_dim, 0.1, weights_are_query=False))
print("")
print("basic dot product layer")
training_and_testing_loop(BasicLinearLayer(10816,comparison_layer_output_dim))

'''
Experiment design

Model parameters

'''