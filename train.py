import os
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import Net
from tqdm import tqdm

def train_cifar10():
    # Define a transform for both training and validation data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(trainset, [45000, 5000])  # Split into train and validation sets

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=4, shuffle=False, num_workers=2)  # Validation loader

    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Check if checkpoint exists and load it
    checkpoint_path = 'model_checkpoint.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded checkpoint and continuing training from last saved epoch.")
    else:
        print("No checkpoint found. Training a new model.")

    criterion = nn.CrossEntropyLoss()

    with mlflow.start_run():
        mlflow.log_params({
            "batch_size": 4,
            "learning_rate": 0.001,
            "epochs": 5
        })

        for epoch in range(5):
            running_loss = 0.0
            for data in tqdm(trainloader, desc=f"Epoch {epoch+1}/5 - Training"):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Validation
            net.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for data in tqdm(valloader, desc=f"Epoch {epoch+1}/5 - Validation"):
                    images, labels = data
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            mlflow.log_metric("epoch", epoch + 1)
            mlflow.log_metric("training_loss", running_loss / len(trainloader))
            mlflow.log_metric("validation_loss", val_loss / len(valloader))
            mlflow.log_metric("validation_accuracy", accuracy)

            print(f"Epoch {epoch+1}, Training Loss: {running_loss / len(trainloader)}, Validation Loss: {val_loss / len(valloader)}, Validation Accuracy: {accuracy}%")

            net.train()  # Switch back to training mode
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)

        print("Training finished.")

if __name__ == '__main__':
    train_cifar10()