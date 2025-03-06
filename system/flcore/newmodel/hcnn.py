import torch
import torch.nn as nn
import torch.nn.functional as F  # 添加这一行



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 捷径
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


class HeteroCNN(nn.Module):
    def __init__(self, model_type, num_classes=10, feat_dims=(512, 512), input_size=(32, 32), ortho_weight=0.01):
        super().__init__()
        self.ortho_weight = ortho_weight
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.coarse_dim, self.fine_dim = feat_dims

        # 独立的卷积路径
        self.coarse_conv = self._build_coarse_conv(model_type)
        self.fine_conv = self._build_fine_conv(model_type)

        # 计算独立卷积路径的输出维度
        coarse_output_dim = self._get_conv_output_dim(self.coarse_conv, input_size)
        fine_output_dim = self._get_conv_output_dim(self.fine_conv, input_size)

        # 独立的全连接层
        self.coarse_fc = nn.Sequential(
            nn.Linear(coarse_output_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.coarse_dim)
        )

        self.fine_fc = nn.Sequential(
            nn.Linear(fine_output_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.fine_dim)
        )

        # 分类头（兼容 StandardCNN 的接口）
        self.classifier = nn.Sequential(
            nn.Linear(self.coarse_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def _build_coarse_conv(self, model_type):
        """粗粒度路径：更大的感受野"""
        layers = nn.Sequential()
        if model_type == "cnn1":
            # 之前的 cnn1 配置
            layers.add_module("conv1", nn.Conv2d(3, 32, kernel_size=5, padding=2))
            layers.add_module("bn1", nn.BatchNorm2d(32))
            layers.add_module("relu1", nn.ReLU())
            layers.add_module("pool1", nn.MaxPool2d(3, stride=2))

            layers.add_module("conv2", nn.Conv2d(32, 64, kernel_size=5, padding=2))
            layers.add_module("bn2", nn.BatchNorm2d(64))
            layers.add_module("relu2", nn.ReLU())
            layers.add_module("pool2", nn.MaxPool2d(3, stride=2))

        elif model_type == "cnn2":
            # 之前的 cnn2 配置
            layers.add_module("conv1", nn.Conv2d(3, 32, kernel_size=5, padding=2))
            layers.add_module("bn1", nn.BatchNorm2d(32))
            layers.add_module("relu1", nn.ReLU())
            layers.add_module("pool1", nn.MaxPool2d(3, stride=2))

            layers.add_module("conv2", nn.Conv2d(32, 64, kernel_size=5, padding=2))
            layers.add_module("bn2", nn.BatchNorm2d(64))
            layers.add_module("relu2", nn.ReLU())
            layers.add_module("resblock", ResidualBlock(in_channels=64, out_channels=128))
            layers.add_module("pool2", nn.MaxPool2d(3, stride=2))

        elif model_type == "cnn3":
            # 新增 cnn3 的粗粒度路径
            layers.add_module("conv1", nn.Conv2d(3, 64, kernel_size=5, padding=2))
            layers.add_module("bn1", nn.BatchNorm2d(64))
            layers.add_module("relu1", nn.ReLU())
            layers.add_module("pool1", nn.MaxPool2d(3, stride=2))

            layers.add_module("conv2", nn.Conv2d(64, 128, kernel_size=5, padding=2))
            layers.add_module("bn2", nn.BatchNorm2d(128))
            layers.add_module("relu2", nn.ReLU())
            layers.add_module("pool2", nn.MaxPool2d(3, stride=2))

            layers.add_module("conv3", nn.Conv2d(128, 256, kernel_size=5, padding=2))
            layers.add_module("bn3", nn.BatchNorm2d(256))
            layers.add_module("relu3", nn.ReLU())
            layers.add_module("pool3", nn.MaxPool2d(3, stride=2))

        elif model_type == "cnn4":
            # 新增 cnn4 的粗粒度路径（轻量级，但感受野更大）
            layers.add_module("conv1", nn.Conv2d(3, 32, kernel_size=5, padding=2))
            layers.add_module("bn1", nn.BatchNorm2d(32))
            layers.add_module("relu1", nn.ReLU())
            layers.add_module("pool1", nn.MaxPool2d(3, stride=2))

            layers.add_module("conv2", nn.Conv2d(32, 64, kernel_size=5, padding=2))
            layers.add_module("bn2", nn.BatchNorm2d(64))
            layers.add_module("relu2", nn.ReLU())
            layers.add_module("pool2", nn.MaxPool2d(3, stride=2))

        return layers.to(self.device)

    def _build_fine_conv(self, model_type):
        """细粒度路径：保留细节"""
        layers = nn.Sequential()
        if model_type == "cnn1":
            # 之前的 cnn1 配置
            layers.add_module("conv1", nn.Conv2d(3, 16, kernel_size=3, padding=1))
            layers.add_module("bn1", nn.BatchNorm2d(16))
            layers.add_module("relu1", nn.ReLU())
            layers.add_module("pool1", nn.MaxPool2d(2))

            layers.add_module("conv2", nn.Conv2d(16, 32, kernel_size=3, padding=1))
            layers.add_module("bn2", nn.BatchNorm2d(32))
            layers.add_module("relu2", nn.ReLU())
            layers.add_module("pool2", nn.MaxPool2d(2))

        elif model_type == "cnn2":
            # 之前的 cnn2 配置
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
            # 新增 cnn3 的细粒度路径
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
            layers.add_module("pool3", nn.MaxPool2d(2))

        elif model_type == "cnn4":
            # 新增 cnn4 的细粒度路径（轻量级，更细致的特征提取）
            layers.add_module("conv1", nn.Conv2d(3, 16, kernel_size=3, padding=1))
            layers.add_module("bn1", nn.BatchNorm2d(16))
            layers.add_module("relu1", nn.ReLU())
            layers.add_module("pool1", nn.MaxPool2d(2))

            layers.add_module("conv2", nn.Conv2d(16, 32, kernel_size=3, padding=1))
            layers.add_module("bn2", nn.BatchNorm2d(32))
            layers.add_module("relu2", nn.ReLU())
            layers.add_module("pool2", nn.MaxPool2d(2))

        return layers.to(self.device)

    def _get_conv_output_dim(self, conv_layers, input_size=(32, 32)):
        """计算特定卷积路径的输出维度"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_size[0], input_size[1]).to(self.device)
            output = conv_layers(dummy_input)
            return output.view(1, -1).shape[1]

    def forward(self, x):
        x = x.to(self.device)

        # 独立路径特征提取
        coarse_conv = torch.flatten(self.coarse_conv(x), 1)
        fine_conv = torch.flatten(self.fine_conv(x), 1)

        # 特征变换
        fine_feat = self.fine_fc(fine_conv)
        coarse_feat = self.coarse_fc(coarse_conv)

        # 正交性损失计算
        ortho_loss = self._calculate_ortho_loss(coarse_feat, fine_feat)

        # 分类输出（使用粗粒度特征）
        output = self.classifier(coarse_feat)

        return {
            'output': output,
            'coarse': coarse_feat,
            'fine': fine_feat,
            'ortho_loss': ortho_loss
        }

    def _calculate_ortho_loss(self, coarse, fine):
        """改进的正交性损失计算"""
        # 标准化特征
        coarse_norm = F.normalize(coarse, dim=1)
        fine_norm = F.normalize(fine, dim=1)

        # 计算余弦相似度
        cos_sim = torch.sum(coarse_norm * fine_norm, dim=1)

        # 温和的正交约束
        loss = torch.clamp(torch.abs(cos_sim), max=0.3).mean()

        return self.ortho_weight * loss


def hcnn1(**kwargs):
    return HeteroCNN(model_type="cnn1", **kwargs)


def hcnn2(**kwargs):
    return HeteroCNN(model_type="cnn2", **kwargs)


def hcnn3(**kwargs):
    return HeteroCNN(model_type="cnn3", **kwargs)


def hcnn4(**kwargs):
    return HeteroCNN(model_type="cnn4", **kwargs)