import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

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
    
def train_cifar10():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    with mlflow.start_run():
        mlflow.log_params({
            "batch_size": 4,
            "learning_rate": 0.001,
            "epochs": 5
        })

        for epoch in range(5):
            running_loss = 0.0  # Initialize loss for each epoch
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()  # Update running_loss for each batch

            mlflow.log_metric("epoch", epoch + 1)
            mlflow.log_metric("loss", running_loss / len(trainloader))
            print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}")

        print("Training finished.")
    torch.save(net.state_dict(), 'model_checkpoint.pth')

if __name__ == '__main__':
    train_cifar10()
