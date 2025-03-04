import torch
import torch.nn as nn


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
    def __init__(self, model_type, num_classes=10, feat_dims=(256, 512), input_size=(32, 32)):
        super().__init__()
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.coarse_dim, self.fine_dim = feat_dims

        ##################################################
        # Step 1: 根据 model_type 动态配置卷积层结构
        ##################################################
        # 公共卷积层配置
        self.conv_layers = self._build_conv_layers(model_type)

        conv_output_dim = self._get_conv_output_dim(input_size)

        ##################################################
        # Step 2: 独立粗/细粒度特征分支
        ##################################################
        # 细粒度路径
        self.fine_fc = nn.Sequential(
            nn.Linear(conv_output_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.fine_dim))

        # 粗粒度路径
        self.coarse_fc = nn.Sequential(
            nn.Linear(conv_output_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.coarse_dim))

        # 分类头（仅用粗粒度特征）
        self.classifier = nn.Linear(self.coarse_dim, num_classes)

    def _build_conv_layers(self, model_type):
        """根据 model_type 配置不同的卷积结构"""
        layers = nn.Sequential()

        if model_type == "cnn1":
            # 简单结构：2层卷积
            layers.add_module("conv1", nn.Conv2d(3, 16, kernel_size=3, padding=1))
            layers.add_module("bn1", nn.BatchNorm2d(16))
            layers.add_module("relu1", nn.ReLU())
            layers.add_module("pool1", nn.MaxPool2d(2))

            layers.add_module("conv2", nn.Conv2d(16, 32, kernel_size=3, padding=1))
            layers.add_module("bn2", nn.BatchNorm2d(32))
            layers.add_module("relu2", nn.ReLU())
            layers.add_module("pool2", nn.MaxPool2d(2))


        elif model_type == "cnn2":

            # 中等结构：3层卷积 + 残差连接

            layers.add_module("conv1", nn.Conv2d(3, 16, kernel_size=3, padding=1))

            layers.add_module("bn1", nn.BatchNorm2d(16))

            layers.add_module("relu1", nn.ReLU())

            layers.add_module("pool1", nn.MaxPool2d(2))

            layers.add_module("conv2", nn.Conv2d(16, 32, kernel_size=3, padding=1))

            layers.add_module("bn2", nn.BatchNorm2d(32))

            layers.add_module("relu2", nn.ReLU())

            # 修正残差连接

            layers.add_module("resblock", ResidualBlock(in_channels=32, out_channels=64))  # 自定义残差块

            layers.add_module("pool2", nn.MaxPool2d(2))

        elif model_type == "cnn3":
            # 深度结构：5层卷积 + SE注意力机制
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
            # 轻量级结构：深度可分离卷积
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
        """计算卷积层输出维度"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_size[0], input_size[1]).to(self.device)
            output = self.conv_layers(dummy_input)
            return output.view(1, -1).shape[1]

    def forward(self, x):
        # 公共特征提取
        x = x.to(self.device)
        conv_features = self.conv_layers(x)
        conv_features = torch.flatten(conv_features, 1)

        # 独立分支提取多粒度特征
        fine_feat = self.fine_fc(conv_features)  # [B, fine_dim]
        coarse_feat = self.coarse_fc(conv_features)  # [B, coarse_dim]

        # 分类输出
        out = self.classifier(coarse_feat)
        return {
            'output': out,
            'coarse': coarse_feat,
            'fine': fine_feat
        }


# 模型工厂函数（支持不同异构模型）
def hcnn1(**kwargs):
    return HeteroCNN(model_type="cnn1", **kwargs)

def hcnn2(**kwargs):
    return HeteroCNN(model_type="cnn2", **kwargs)

def hcnn3(**kwargs):
    return HeteroCNN(model_type="cnn3", **kwargs)

def hcnn4(**kwargs):
    return HeteroCNN(model_type="cnn4", **kwargs)