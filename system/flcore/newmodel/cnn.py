import torch
import torch.nn as nn


class HeteroCNN(nn.Module):
    def __init__(self, model_type, num_classes=10, feat_dims=(256, 512)):
        super().__init__()
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.coarse_dim, self.fine_dim = feat_dims

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

        # 细粒度特征投影层（统一维度）
        self.fine_proj = nn.Linear(fc1_dims[model_type], self.fine_dim).to(self.device)

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

    # 模型特征提取
    # 检查HeteroCNN特征提取层次：
    # def forward(self, x):
    #     features = self.forward_features(x)  # 基础特征
    #     fine_feat_x = torch.relu(self.fc1(features))  # 中间层 → 细粒度
    #     fine_feat = self.fine_proj(fine_feat_x)  # 细粒度特征
    #     coarse_feat = torch.relu(self.fc2(fine_feat_x))  # 更高层 → 粗粒度
    #     # ? 符合"低维粗粒度，高维细粒度"的设计

    def forward(self, x):
        # 确保输入在正确设备
        x = x.to(self.device)

        # 细粒度特征（FC1层投影）
        features = self.forward_features(x)  # [B, C]
        fine_feat_x = torch.relu(self.fc1(features))  # [B, fc1_dim]
        fine_feat = self.fine_proj(fine_feat_x)  # [B, fine_dim]

        # 粗粒度特征（FC2层）
        coarse_feat = torch.relu(self.fc2(fine_feat_x))  # [B, 500]
        out = self.fc3(coarse_feat)  # [B, num_classes]
        return {
            'output': out,
            'coarse': coarse_feat,  # 粗粒度特征 (500维)
            'fine': fine_feat  # 细粒度特征 (统一为512维)
        }


def cnn1(**kwargs) -> HeteroCNN:
    return HeteroCNN('CNN-1', **kwargs)


def cnn2(**kwargs) -> HeteroCNN:
    return HeteroCNN('CNN-2', **kwargs)


def cnn3(**kwargs):
    return HeteroCNN('CNN-3', **kwargs)


def cnn4(**kwargs):
    return HeteroCNN('CNN-4', **kwargs)


def cnn5(**kwargs):
    return HeteroCNN('CNN-5', **kwargs)