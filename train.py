import os

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from model import Net

EPOCHS = 8
BATCH_SIZE = 128
MAX_LR = 0.05
CHECKPOINT = "model_checkpoint.pth"


def pick_device():
    """Use the accelerator if there is one.

    The earlier version of this file never called .to() at all, so it trained on
    the CPU on a machine with a GPU sitting idle. On this model that is about an
    order of magnitude: roughly 620 images/second on the CPU against 13,000
    through MPS at this batch size.

    The margin depends heavily on batch size. At batch 4 it is closer to 2x,
    because each step finishes fast enough that the loop spends its time waiting
    on the data pipeline rather than computing. Batching is what makes the
    accelerator worth having.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def loaders():
    # Augmentation on train only. Random crop plus horizontal flip is the
    # standard pair for CIFAR-10 and is worth several points on its own.
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_full = torchvision.datasets.CIFAR10(root="./data", train=True,
                                              download=True, transform=train_tf)
    eval_full = torchvision.datasets.CIFAR10(root="./data", train=True,
                                             download=True, transform=eval_tf)

    g = torch.Generator().manual_seed(0)
    train_set, _ = torch.utils.data.random_split(train_full, [45000, 5000], generator=g)
    g = torch.Generator().manual_seed(0)
    _, val_set = torch.utils.data.random_split(eval_full, [45000, 5000], generator=g)
    # Two splits over two datasets with the same seed, so the validation images
    # are the same images but without augmentation. Evaluating on randomly
    # cropped images understates accuracy and makes the curve noisy -- the
    # previous version of this file did exactly that.

    return (
        torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                    num_workers=2, drop_last=True, persistent_workers=True),
        torch.utils.data.DataLoader(val_set, batch_size=512, shuffle=False,
                                    num_workers=2, persistent_workers=True),
    )


def train_cifar10():
    device = pick_device()
    print(f"device: {device}")

    trainloader, valloader = loaders()
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=MAX_LR, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    # One-cycle: warm up to MAX_LR, then anneal towards zero. Expect the
    # validation curve to dip around the peak -- that is the schedule working,
    # not the run failing, and it is why you plot every epoch rather than only
    # the last one.
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, epochs=EPOCHS, steps_per_epoch=len(trainloader)
    )

    if os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location=device)
        net.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print("Loaded checkpoint and continuing training.")
    else:
        print("No checkpoint found. Training a new model.")

    with mlflow.start_run():
        mlflow.log_params({
            "batch_size": BATCH_SIZE, "max_lr": MAX_LR, "epochs": EPOCHS,
            "scheduler": "one_cycle", "device": str(device),
            "parameters": sum(p.numel() for p in net.parameters()),
        })

        for epoch in range(EPOCHS):
            net.train()
            running = 0.0
            for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(net(inputs), labels)
                loss.backward()
                optimizer.step()
                scheduler.step()          # per step, which is what one-cycle expects
                running += loss.item()

            net.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in tqdm(valloader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation"):
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    val_loss += criterion(outputs, labels).item()
                    correct += (outputs.argmax(1) == labels).sum().item()
                    total += labels.size(0)

            accuracy = 100 * correct / total
            mlflow.log_metrics({
                "training_loss": running / len(trainloader),
                "validation_loss": val_loss / len(valloader),
                "validation_accuracy": accuracy,
                "learning_rate": scheduler.get_last_lr()[0],
            }, step=epoch + 1)

            print(f"Epoch {epoch+1}, Training Loss: {running/len(trainloader):.4f}, "
                  f"Validation Loss: {val_loss/len(valloader):.4f}, "
                  f"Validation Accuracy: {accuracy:.2f}%")

            torch.save({
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, CHECKPOINT)

        print("Training finished.")


if __name__ == "__main__":
    train_cifar10()
