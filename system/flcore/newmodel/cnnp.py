import torch
import torch.nn as nn


class HeteroCNNP(nn.Module):
    def __init__(self, model_type, num_classes=10):
        super().__init__()
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 公共卷积层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2).to(self.device)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = self._build_conv2().to(self.device)
        self.pool2 = nn.MaxPool2d(2)

        # 自适应池化解决尺寸问题
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

        # 全连接层配置
        fc1_dims = {
            'CNN-1': 2000,
            'CNN-2': 2000,
            'CNN-3': 1000,
            'CNN-4': 800,
            'CNN-5': 500
        }
        self.fc1 = nn.Linear(self._get_conv2_out_channels(), fc1_dims[model_type]).to(self.device)
        self.fc2 = nn.Linear(fc1_dims[model_type], 500).to(self.device)
        self.fc3 = nn.Linear(500, num_classes).to(self.device)

        # 定义分类头
        self.head = self.fc3  # 将 fc3 层作为分类头

    def _build_conv2(self):
        """根据模型类型构建第二个卷积层"""
        conv2_channels = {
            'CNN-1': 32,
            'CNN-2': 16,
            'CNN-3': 32,
            'CNN-4': 32,
            'CNN-5': 32
        }
        return nn.Conv2d(16, conv2_channels[self.model_type], kernel_size=5, padding=2)

    def _get_conv2_out_channels(self):
        return self.conv2.out_channels

    def forward_features(self, x):
        # 确保输入在正确设备
        x = x.to(self.device)

        # 公共特征提取路径
        x = self.pool1(torch.relu(self.conv1(x)))  # [B,16,16,16]
        x = self.pool2(torch.relu(self.conv2(x)))  # [B,C,8,8]
        x = self.adaptive_pool(x)  # [B,C,1,1]
        x = torch.flatten(x, 1)  # [B, C]
        return x

    def forward(self, x):
        # 确保输入在正确设备
        x = x.to(self.device)

        # 提取公共特征
        features = self.forward_features(x)  # [B, C]

        # 完整的前向计算
        x = torch.relu(self.fc1(features))  # [B, fc1_dim]
        x = torch.relu(self.fc2(x))  # [B, 500]
        out = self.head(x)  # [B, num_classes]，输出最终的预测值（例如分类得分）

        return out  # 只返回输出，不需要细粒度或粗粒度的中间结果


# 生成不同模型类型的函数
def cnnp1(**kwargs) -> HeteroCNNP:
    return HeteroCNNP('CNN-1', **kwargs)


def cnnp2(**kwargs) -> HeteroCNNP:
    return HeteroCNNP('CNN-2', **kwargs)


def cnnp3(**kwargs) -> HeteroCNNP:
    return HeteroCNNP('CNN-3', **kwargs)


def cnnp4(**kwargs) -> HeteroCNNP:
    return HeteroCNNP('CNN-4', **kwargs)


def cnnp5(**kwargs) -> HeteroCNNP:
    return HeteroCNNP('CNN-5', **kwargs)
