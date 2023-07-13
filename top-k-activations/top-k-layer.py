import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Custom layer definition
class TopKActivationLayer(nn.Module):
    def __init__(self, k):
        super(TopKActivationLayer, self).__init__()
        self.k = k

    def forward(self, input):
        flattened_input = input.view(input.size(0), -1)
        _, indices = torch.topk(flattened_input.abs(), int(self.k * flattened_input.size(1)), dim=1)
        threshold = torch.gather(flattened_input, 1, indices).min(dim=1, keepdim=True)[0]
        activated_input = torch.where(input >= threshold.view(-1, 1, 1, 1), input, torch.zeros_like(input))
        return activated_input

# Create a sample model using the TopKActivationLayer
class MyModel(nn.Module):
    def __init__(self, k):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.topk = TopKActivationLayer(k=k)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 28 * 28, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.topk(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.topk(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load FashionMNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform)


# Set hyperparameters
batch_size = 32
k = 1
learning_rate = 0.001
num_epochs = 5


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the model
model = MyModel(k=k).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item()}")

# Testing
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Accuracy on test images: {100 * correct / total}%")