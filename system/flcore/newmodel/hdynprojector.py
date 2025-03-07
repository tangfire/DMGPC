import torch
import torch.nn as nn

# class HDynamicProjector(nn.Module):
#     def __init__(self, d_coarse=512, d_fine=512, hidden_dim=512, fused_dim=512):
#         super().__init__()
#
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         self.fused_dim = fused_dim
#
#
#
#         # 细粒度 → 统一维度
#         self.fine_proj = nn.Sequential(
#             nn.Linear(d_fine, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, fused_dim)
#         ).to(self.device)
#
#         # 粗粒度 → 统一维度
#         self.coarse_proj = nn.Sequential(
#             nn.Linear(d_coarse, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, fused_dim)
#         ).to(self.device)
#
#         # 动态门控（输入为原始维度之和）
#         self.gate_net = nn.Sequential(
#             nn.Linear(d_coarse + d_fine, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         ).to(self.device)
#
#     def forward(self, P_coarse, P_fine):
#         # 映射到统一维度
#         aligned_coarse = self.coarse_proj(P_coarse).to(self.device)  # [B, fused_dim]
#         aligned_fine = self.fine_proj(P_fine).to(self.device)      # [B, fused_dim]
#
#         # 动态权重
#         gate_input = torch.cat([P_coarse, P_fine], dim=1)
#         alpha = self.gate_net(gate_input)  # [B, 1]
#
#         # 加权融合
#         fused_proto = alpha * aligned_coarse + (1 - alpha) * aligned_fine
#         return fused_proto  # [B, fused_dim]

# class HDynamicProjector(nn.Module):
#     def __init__(self, d_coarse=512, d_fine=512):
#         super().__init__()
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         # 轻量级适配器（客户端本地）
#         self.adapter = nn.Sequential(
#             nn.Linear(d_coarse + d_fine, 256),
#             nn.ReLU(),
#             nn.Linear(256, d_coarse)
#         ).to(self.device)
#
#     def forward(self, P_coarse, P_fine):
#         # 动态融合本地多粒度原型
#         fused = self.adapter(torch.cat([P_coarse, P_fine], dim=1)).to(self.device)
#         return fused


# class HDynamicProjector(nn.Module):
#     def __init__(self, d_coarse=512, d_fine=512):
#         super().__init__()
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.attention = nn.Sequential(
#             nn.Linear(d_coarse + d_fine, 128),
#             nn.ReLU(),
#             nn.Linear(128, 2),
#             nn.Softmax(dim=1)
#         ).to(self.device)
#
#     def forward(self, P_coarse, P_fine):
#         # P_coarse: [B, D1], P_fine: [B, D2]
#         feat = torch.cat([P_coarse, P_fine], dim=1)
#         alpha = self.attention(feat)  # [B, 2]
#         fused = alpha[:, 0].unsqueeze(1) * P_coarse + alpha[:, 1].unsqueeze(1) * P_fine
#         return fused


class HDynamicProjector(nn.Module):
    def __init__(self, d_coarse=512, d_fine=512, hidden_dim=128):
        super().__init__()
        # 元网络：学习如何调整权重
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.meta_net = nn.LSTM(
            input_size=d_coarse + d_fine,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True
        ).to(self.device)
        self.alpha_generator = nn.Sequential(
            nn.Linear(2 * hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        ).to(self.device)

    def forward(self, P_coarse, P_fine):
        # 输入形状：[B, d_coarse], [B, d_fine]
        combined = torch.cat([P_coarse, P_fine], dim=1)

        # 通过LSTM学习序列关系
        out, _ = self.meta_net(combined.unsqueeze(1))  # 添加序列维度
        alpha = self.alpha_generator(out.squeeze(1))

        # 动态融合
        fused = alpha[:, 0].unsqueeze(1) * P_coarse + alpha[:, 1].unsqueeze(1) * P_fine
        return fused
