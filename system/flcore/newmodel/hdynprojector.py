import torch
import torch.nn as nn

class HDynamicProjector(nn.Module):
    def __init__(self, d_coarse=512, d_fine=512, hidden_dim=512, fused_dim=512):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.fused_dim = fused_dim

        # 细粒度 → 统一维度
        self.fine_proj = nn.Sequential(
            nn.Linear(d_fine, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fused_dim)
        ).to(self.device)

        # 粗粒度 → 统一维度
        self.coarse_proj = nn.Sequential(
            nn.Linear(d_coarse, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fused_dim)
        ).to(self.device)

        # 动态门控（输入为原始维度之和）
        self.gate_net = nn.Sequential(
            nn.Linear(d_coarse + d_fine, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(self.device)

    def forward(self, P_coarse, P_fine):
        # 映射到统一维度
        aligned_coarse = self.coarse_proj(P_coarse).to(self.device)  # [B, fused_dim]
        aligned_fine = self.fine_proj(P_fine).to(self.device)      # [B, fused_dim]

        # 动态权重
        gate_input = torch.cat([P_coarse, P_fine], dim=1)
        alpha = self.gate_net(gate_input)  # [B, 1]

        # 加权融合
        fused_proto = alpha * aligned_coarse + (1 - alpha) * aligned_fine
        return fused_proto  # [B, fused_dim]