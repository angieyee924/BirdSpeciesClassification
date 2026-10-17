import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights
)


class CustomCNN(nn.Module):
    """
    Baseline convolutional neural network for
    200-class bird species classification.
    """

    def __init__(self, num_classes=200):
        super(CustomCNN, self).__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 4
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Convert feature map to 1 x 1
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(
                in_features=256,
                out_features=num_classes
            )
        )

    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = self.features(x)
        x = self.classifier(x)

        return x


def create_resnet18(num_classes=200):
    """
    Create an ImageNet-pretrained ResNet18 model
    for 200-class bird classification.
    """

    weights = ResNet18_Weights.DEFAULT

    model = resnet18(
        weights=weights
    )

    # Replace the ImageNet classifier
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(
            model.fc.in_features,
            num_classes
        )
    )

    return model


def create_efficientnet_b0(num_classes=200):
    """
    Create an ImageNet-pretrained EfficientNet-B0
    for 200-class bird classification.
    """

    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(
        weights=weights
    )

    in_features = (
        model.classifier[1].in_features
    )

    # Replace the ImageNet classifier
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(
            in_features,
            num_classes
        )
    )

    return model