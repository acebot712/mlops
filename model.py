import torch.nn as nn


def block(in_ch, out_ch):
    """Two 3x3 convolutions at the same width, then halve the resolution.

    Batch norm after each convolution and before the activation. It is what lets
    this train at a learning rate two orders of magnitude above the previous
    version of this file without diverging, and it is most of why the network
    reaches a usable accuracy in eight epochs.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
        nn.MaxPool2d(2),
    )


class Net(nn.Module):
    """Small CIFAR-10 CNN: three blocks, one linear head. 308,394 parameters.

    Deliberately boring. Nothing here tries to squeeze accuracy out of CIFAR-10;
    that problem was solved a decade ago. It is sized so training is cheap enough
    to rerun while reading about it, and good enough that serving it is a real
    problem.

    Worth knowing if you are comparing against the previous version of this file:
    that network had 1,147,466 parameters and no batch norm, and it was about
    three times FASTER on a CPU despite being nearly four times larger. Batch
    norm is memory-bandwidth bound and does not vectorise the way a large matrix
    multiply does. Parameter count is a fair proxy for memory and a poor one for
    time, and which way it misleads you depends on the device.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            block(3, 32),      # 32x32 -> 16x16
            block(32, 64),     # 16x16 -> 8x8
            block(64, 128),    # 8x8   -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128 * 4 * 4, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
