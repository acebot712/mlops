import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Step 1: Load and preprocess the dataset (Example: CIFAR-10)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

# Step 2: Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 128)  # Input size: 3 channels, 32x32 image
        self.fc2 = nn.Linear(128, 10)  # Output size: 10 classes

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()

# Step 3: Define a loss function and an optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Step 4: Train the model
for epoch in range(5):  # You can adjust the number of training epochs
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}")

print("Training finished.")
