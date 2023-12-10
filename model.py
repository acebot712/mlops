import torch
import torch.nn as nn

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