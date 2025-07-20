import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SelectiveGate(nn.Module):
    """选择门网络 G_φ(u_k)，生成时变参数 {Δ_k, B_k, C_k}"""

    def __init__(self, d_input: int, d_state: int, dt_rank: int = 16):
        super().__init__()
        self.d_input = d_input
        self.d_state = d_state
        self.dt_rank = dt_rank

        # 时间步长参数生成
        self.dt_proj = nn.Linear(d_input, dt_rank, bias=True)
        self.dt_rank_proj = nn.Linear(dt_rank, 1, bias=False)

        # 状态矩阵B和C生成
        self.B_proj = nn.Linear(d_input, d_state, bias=False)
        self.C_proj = nn.Linear(d_input, d_state, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入 [batch, seq_len, d_input]
        Returns:
            delta, B, C: 时变参数
        """
        batch, seq_len, _ = x.shape

        # 生成时间步长 Δ_k
        dt = self.dt_proj(x)  # [batch, seq_len, dt_rank]
        dt = self.dt_rank_proj(dt).squeeze(-1)  # [batch, seq_len]
        dt = F.softplus(dt)  # 确保正值

        # 生成状态矩阵 B_k 和 C_k
        B = self.B_proj(x)  # [batch, seq_len, d_state]
        C = self.C_proj(x)  # [batch, seq_len, d_state]

        return dt, B, C


class MinimalityRegularizer(nn.Module):
    """最小性正则化模块，估计 I(U_{1:k}; h_k)"""

    def __init__(self, d_state: int, d_input: int, hidden_dim: int = 128):
        super().__init__()
        self.d_state = d_state
        self.d_input = d_input

        # 辅助解码器 p_θ(u_k|h_k)
        self.decoder = nn.Sequential(
            nn.Linear(d_state, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_input)
        )

    def forward(self, h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        估计互信息 I(U_{1:k}; h_k) 的变分上界

        Args:
            h: 隐藏状态 [batch, seq_len, d_state]
            u: 输入序列 [batch, seq_len, d_input]
        Returns:
            mutual_info_estimate: 互信息估计值
        """
        batch, seq_len, _ = h.shape

        # 重构损失作为互信息的代理
        u_reconstructed = self.decoder(h)  # [batch, seq_len, d_input]

        # 计算重构误差（负对数似然）
        reconstruction_loss = F.mse_loss(u_reconstructed, u, reduction='none')
        reconstruction_loss = reconstruction_loss.sum(dim=-1)  # [batch, seq_len]

        # 互信息估计（简化版本）
        mutual_info = reconstruction_loss.mean()

        return mutual_info


class SSMKernel(nn.Module):
    """SSM核心计算模块"""

    def __init__(self, d_state: int):
        super().__init__()
        self.d_state = d_state

        # 状态矩阵A（固定的对角结构，类似S4）
        self.A = nn.Parameter(torch.randn(d_state))

    def forward(self, x: torch.Tensor, dt: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        离散化SSM前向传播

        Args:
            x: 输入 [batch, seq_len, d_input]
            dt: 时间步长 [batch, seq_len]
            B: 输入矩阵 [batch, seq_len, d_state]  
            C: 输出矩阵 [batch, seq_len, d_state]
        Returns:
            y: 输出 [batch, seq_len, d_state]
        """
        batch, seq_len, d_input = x.shape
        device = x.device

        # 离散化参数
        A_discrete = torch.exp(dt.unsqueeze(-1) * self.A.unsqueeze(0).unsqueeze(0))  # [batch, seq_len, d_state]
        B_discrete = dt.unsqueeze(-1) * B  # [batch, seq_len, d_state]

        # 初始化隐藏状态
        h = torch.zeros(batch, self.d_state, device=device)
        outputs = []

        # 序列处理
        for i in range(seq_len):
            # 状态更新: h_k = A_discrete * h_{k-1} + B_discrete * x_k
            h = A_discrete[:, i] * h + B_discrete[:, i] * x[:, i].unsqueeze(-1)

            # 输出: y_k = C_k * h_k
            y = (C[:, i] * h).sum(dim=-1, keepdim=True)
            outputs.append(y)

        outputs = torch.stack(outputs, dim=1)  # [batch, seq_len, 1]

        return outputs, h


class MPSSSMLayer(nn.Module):
    """MPS-SSM单层实现"""

    def __init__(self, d_input: int, d_state: int = 16, dt_rank: int = 16):
        super().__init__()
        self.d_input = d_input
        self.d_state = d_state

        # 核心组件
        self.selective_gate = SelectiveGate(d_input, d_state, dt_rank)
        self.ssm_kernel = SSMKernel(d_state)
        self.minimality_regularizer = MinimalityRegularizer(d_state, d_input)

        # 输入/输出投影
        self.input_proj = nn.Linear(d_input, d_input)
        self.output_proj = nn.Linear(1, d_input)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入 [batch, seq_len, d_input]
        Returns:
            output: 层输出 [batch, seq_len, d_input]
            mi_loss: 互信息损失
        """
        # 输入投影
        x_proj = self.input_proj(x)

        # 生成选择性参数
        dt, B, C = self.selective_gate(x_proj)

        # SSM前向传播
        ssm_out, final_state = self.ssm_kernel(x_proj, dt, B, C)

        # 输出投影
        output = self.output_proj(ssm_out)

        # 计算互信息正则化项
        # 这里需要构造累积历史，简化处理
        mi_loss = self.minimality_regularizer(
            final_state.unsqueeze(1).expand(-1, x.size(1), -1),
            x
        )

        return output, mi_loss


class MPSSSM(nn.Module):
    """完整的MPS-SSM模型"""

    def __init__(self,
                 d_input: int,
                 d_output: int,
                 d_state: int = 16,
                 n_layers: int = 4,
                 lambda_mi: float = 0.01):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.n_layers = n_layers
        self.lambda_mi = lambda_mi

        # 多层MPS-SSM
        self.layers = nn.ModuleList([
            MPSSSMLayer(d_input, d_state) for _ in range(n_layers)
        ])

        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(d_input, d_input // 2),
            nn.ReLU(),
            nn.Linear(d_input // 2, d_output)
        )

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> dict:
        """
        Args:
            x: 输入序列 [batch, seq_len, d_input]
            targets: 目标值 [batch, seq_len, d_output] (训练时)
        Returns:
            结果字典包含predictions, losses等
        """
        batch, seq_len, _ = x.shape

        # 通过多层MPS-SSM
        hidden = x
        total_mi_loss = 0.0

        for layer in self.layers:
            hidden, mi_loss = layer(hidden)
            total_mi_loss += mi_loss

        # 预测
        predictions = self.prediction_head(hidden)

        result = {
            'predictions': predictions,
            'mi_loss': total_mi_loss / self.n_layers
        }

        # 计算总损失
        if targets is not None:
            # 预测损失
            pred_loss = F.mse_loss(predictions, targets)

            # 总损失 = 预测损失 + λ * 互信息损失
            total_loss = pred_loss + self.lambda_mi * total_mi_loss / self.n_layers

            result.update({
                'pred_loss': pred_loss,
                'total_loss': total_loss
            })

        return result


# 使用示例
def example_usage():
    """使用示例"""
    batch_size = 32
    seq_len = 512
    d_input = 64
    d_output = 1

    # 创建模型
    model = MPSSSM(
        d_input=d_input,
        d_output=d_output,
        d_state=16,
        n_layers=4,
        lambda_mi=0.01
    )

    # 模拟数据
    x = torch.randn(batch_size, seq_len, d_input)
    targets = torch.randn(batch_size, seq_len, d_output)

    # 前向传播
    result = model(x, targets)

    print(f"Predictions shape: {result['predictions'].shape}")
    print(f"Prediction loss: {result['pred_loss'].item():.4f}")
    print(f"MI loss: {result['mi_loss'].item():.4f}")
    print(f"Total loss: {result['total_loss'].item():.4f}")

    return model, result


if __name__ == "__main__":
    model, result = example_usage()