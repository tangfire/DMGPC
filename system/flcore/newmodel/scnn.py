import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return self.relu(x)


class StandardCNN(nn.Module):
    def __init__(self, model_type, num_classes=10, input_size=(32, 32)):
        super().__init__()
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.conv_layers = self._build_conv_layers(model_type)
        conv_output_dim = self._get_conv_output_dim(input_size)

        self.classifier = nn.Sequential(
            nn.Linear(conv_output_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def _build_conv_layers(self, model_type):
        layers = nn.Sequential()

        if model_type == "cnn1":
            layers.add_module("conv1", nn.Conv2d(3, 16, kernel_size=3, padding=1))
            layers.add_module("bn1", nn.BatchNorm2d(16))
            layers.add_module("relu1", nn.ReLU())
            layers.add_module("pool1", nn.MaxPool2d(2))

            layers.add_module("conv2", nn.Conv2d(16, 32, kernel_size=3, padding=1))
            layers.add_module("bn2", nn.BatchNorm2d(32))
            layers.add_module("relu2", nn.ReLU())
            layers.add_module("pool2", nn.MaxPool2d(2))

        elif model_type == "cnn2":
            layers.add_module("conv1", nn.Conv2d(3, 16, kernel_size=3, padding=1))
            layers.add_module("bn1", nn.BatchNorm2d(16))
            layers.add_module("relu1", nn.ReLU())
            layers.add_module("pool1", nn.MaxPool2d(2))

            layers.add_module("conv2", nn.Conv2d(16, 32, kernel_size=3, padding=1))
            layers.add_module("bn2", nn.BatchNorm2d(32))
            layers.add_module("relu2", nn.ReLU())

            layers.add_module("resblock", ResidualBlock(in_channels=32, out_channels=64))
            layers.add_module("pool2", nn.MaxPool2d(2))

        elif model_type == "cnn3":
            layers.add_module("conv1", nn.Conv2d(3, 32, kernel_size=3, padding=1))
            layers.add_module("bn1", nn.BatchNorm2d(32))
            layers.add_module("relu1", nn.ReLU())
            layers.add_module("pool1", nn.MaxPool2d(2))

            layers.add_module("conv2", nn.Conv2d(32, 64, kernel_size=3, padding=1))
            layers.add_module("bn2", nn.BatchNorm2d(64))
            layers.add_module("relu2", nn.ReLU())
            layers.add_module("pool2", nn.MaxPool2d(2))

            layers.add_module("conv3", nn.Conv2d(64, 128, kernel_size=3, padding=1))
            layers.add_module("bn3", nn.BatchNorm2d(128))
            layers.add_module("relu3", nn.ReLU())

            layers.add_module("conv4", nn.Conv2d(128, 256, kernel_size=3, padding=1))
            layers.add_module("bn4", nn.BatchNorm2d(256))
            layers.add_module("relu4", nn.ReLU())
            layers.add_module("pool3", nn.MaxPool2d(2))

            layers.add_module("conv5", nn.Conv2d(256, 512, kernel_size=3, padding=1))
            layers.add_module("bn5", nn.BatchNorm2d(512))
            layers.add_module("relu5", nn.ReLU())

        elif model_type == "cnn4":
            layers.add_module("conv1", nn.Conv2d(3, 32, kernel_size=3, padding=1))
            layers.add_module("bn1", nn.BatchNorm2d(32))
            layers.add_module("relu1", nn.ReLU())
            layers.add_module("pool1", nn.MaxPool2d(2))

            layers.add_module("conv2", nn.Conv2d(32, 64, kernel_size=3, padding=1))
            layers.add_module("bn2", nn.BatchNorm2d(64))
            layers.add_module("relu2", nn.ReLU())
            layers.add_module("pool2", nn.MaxPool2d(2))

        return layers.to(self.device)

    def _get_conv_output_dim(self, input_size=(32, 32)):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_size[0], input_size[1]).to(self.device)
            output = self.conv_layers(dummy_input)
            return output.view(1, -1).shape[1]

    def forward(self, x):
        x = x.to(self.device)
        conv_features = self.conv_layers(x)
        conv_features = torch.flatten(conv_features, 1)
        output = self.classifier(conv_features)
        return output


def scnn1(**kwargs):
    return StandardCNN(model_type="cnn1", **kwargs)


def scnn2(**kwargs):
    return StandardCNN(model_type="cnn2", **kwargs)


def scnn3(**kwargs):
    return StandardCNN(model_type="cnn3", **kwargs)


def scnn4(**kwargs):
    return StandardCNN(model_type="cnn4", **kwargs)