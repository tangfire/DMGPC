import torch
import torch.nn as nn

class DynamicProjector(nn.Module):
    def __init__(self, d_coarse=500, d_fine=512, hidden_dim=64):
        super().__init__()
        # 维度对齐层
        self.align_layer = nn.Linear(d_fine, d_coarse)

        # 注意力权重生成网络
        self.attention_net = nn.Sequential(
            nn.Linear(d_coarse * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出权重在[0,1]之间
        )

        # 在初始化时确定模型的设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 将所有网络层移动到指定设备
        self.to(self.device)

    def forward(self, P_coarse, P_fine):
        """
        输入:
            P_coarse: [B, d_coarse] 粗粒度原型（B为类别数）
            P_fine: [B, d_fine] 细粒度原型
        输出:
            P_fused: [B, d_coarse] 融合后的原型（与粗粒度维度一致）
        """

        # 确保输入在同一设备，并与模型在同一设备
        P_coarse = P_coarse.to(self.device)
        P_fine = P_fine.to(self.device)

        # 将细粒度原型对齐到粗粒度原型的维度
        P_fine_aligned = self.align_layer(P_fine)

        # 拼接特征（使用对齐后的细粒度原型）
        combined = torch.cat([P_coarse, P_fine_aligned], dim=1)  # [B, d_coarse + d_fine]


        alpha = self.attention_net(combined)  # [B, 1]
        P_fused = alpha * P_coarse + (1 - alpha) * P_fine_aligned
        return P_fused