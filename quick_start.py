#!/usr/bin/env python3
"""
MPS-SSM 快速启动脚本
运行: python quick_start.py
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# 确保能导入我们的模块
sys.path.append('.')


# ============================================================================
# 从之前的artifacts复制核心代码 (简化版本)
# ============================================================================

class MPSSSM(nn.Module):
    """简化的MPS-SSM用于快速测试"""

    def __init__(self, d_input, d_output, d_state=16, lambda_mi=0.01):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.d_state = d_state
        self.lambda_mi = lambda_mi

        # 简化的SSM层
        self.input_proj = nn.Linear(d_input, d_state)
        self.ssm_layer = nn.LSTM(d_state, d_state, batch_first=True)
        self.output_proj = nn.Linear(d_state, d_output)

        # 互信息估计器
        self.mi_estimator = nn.Sequential(
            nn.Linear(d_state, d_state),
            nn.ReLU(),
            nn.Linear(d_state, d_input)
        )

    def forward(self, x, targets=None):
        batch_size, seq_len, _ = x.shape

        # 简化的前向传播
        x_proj = self.input_proj(x)
        hidden, _ = self.ssm_layer(x_proj)
        predictions = self.output_proj(hidden)

        # 互信息估计
        x_reconstructed = self.mi_estimator(hidden)
        mi_loss = nn.functional.mse_loss(x_reconstructed, x)

        result = {
            'predictions': predictions,
            'mi_loss': mi_loss
        }

        if targets is not None:
            pred_loss = nn.functional.mse_loss(predictions, targets)
            total_loss = pred_loss + self.lambda_mi * mi_loss
            result.update({
                'pred_loss': pred_loss,
                'total_loss': total_loss
            })

        return result


def generate_test_data(n_samples=200, seq_len=64, d_input=2):
    """生成测试数据"""
    print("📊 生成测试数据...")

    data = []
    targets = []

    for i in range(n_samples):
        # 生成简单的正弦波
        t = torch.linspace(0, 4 * np.pi, seq_len)
        freq1 = 0.5 + torch.rand(1)
        freq2 = 2.0 + torch.rand(1)

        signal1 = torch.sin(freq1 * t)
        signal2 = torch.cos(freq2 * t)

        # 添加噪声
        noise = 0.1 * torch.randn(seq_len)

        if d_input == 1:
            input_seq = (signal1 + noise).unsqueeze(-1)
        else:
            input_seq = torch.stack([signal1 + noise, signal2 + noise], dim=-1)

        # 目标：预测下一个时刻的干净信号
        target_seq = torch.roll(signal1, -1).unsqueeze(-1)
        target_seq[-1] = target_seq[-2]

        data.append(input_seq)
        targets.append(target_seq)

    return torch.stack(data), torch.stack(targets)


def quick_test():
    """快速测试"""
    print("🚀 MPS-SSM 快速测试开始...")
    print("=" * 50)

    # 1. 检查环境
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 2. 生成数据
    train_data, train_targets = generate_test_data(n_samples=100, seq_len=32, d_input=2)
    test_data, test_targets = generate_test_data(n_samples=20, seq_len=32, d_input=2)

    print(f"训练数据形状: {train_data.shape}")
    print(f"目标数据形状: {train_targets.shape}")

    # 3. 创建模型
    print("\n🧠 创建模型...")
    model = MPSSSM(d_input=2, d_output=1, d_state=16, lambda_mi=0.01)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {total_params:,}")

    # 4. 测试前向传播
    print("\n⚡ 测试前向传播...")
    train_data, train_targets = train_data.to(device), train_targets.to(device)
    test_data, test_targets = test_data.to(device), test_targets.to(device)

    with torch.no_grad():
        result = model(test_data[:4], test_targets[:4])
        print(f"✅ 前向传播成功!")
        print(f"   预测形状: {result['predictions'].shape}")
        print(f"   预测损失: {result['pred_loss'].item():.4f}")
        print(f"   互信息损失: {result['mi_loss'].item():.4f}")
        print(f"   总损失: {result['total_loss'].item():.4f}")

    # 5. 简单训练测试
    print("\n🏋️ 训练测试...")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(
        TensorDataset(train_data, train_targets),
        batch_size=16, shuffle=True
    )

    model.train()
    losses = []

    for epoch in range(10):
        epoch_loss = 0
        n_batches = 0

        for batch_data, batch_targets in train_loader:
            optimizer.zero_grad()

            result = model(batch_data, batch_targets)
            loss = result['total_loss']

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if epoch % 2 == 0:
            print(f"   Epoch {epoch + 1:2d}: Loss = {avg_loss:.4f}")

    print(f"✅ 训练测试完成! 最终损失: {losses[-1]:.4f}")

    # 6. 鲁棒性测试
    print("\n🛡️ 鲁棒性测试...")

    model.eval()

    # 干净数据测试
    with torch.no_grad():
        clean_result = model(test_data, test_targets)
        clean_mse = clean_result['pred_loss'].item()

    # 添加噪声
    noise_strength = 0.5
    noisy_data = test_data + noise_strength * torch.randn_like(test_data)

    with torch.no_grad():
        noisy_result = model(noisy_data, test_targets)
        noisy_mse = noisy_result['pred_loss'].item()

    degradation = (noisy_mse - clean_mse) / clean_mse

    print(f"   干净数据 MSE: {clean_mse:.6f}")
    print(f"   噪声数据 MSE: {noisy_mse:.6f}")
    print(f"   性能下降率: {degradation:.2%}")

    # 7. 可视化结果
    print("\n📈 生成可视化...")

    # 创建输出目录
    os.makedirs('quick_test_results', exist_ok=True)

    # 训练损失
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(losses, 'b-', linewidth=2)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    # 预测vs真实值
    plt.subplot(1, 2, 2)
    with torch.no_grad():
        sample_pred = model(test_data[:1])['predictions'][0, :, 0].cpu()
        sample_true = test_targets[0, :, 0].cpu()

    plt.plot(sample_true.numpy(), 'g-', label='True', linewidth=2)
    plt.plot(sample_pred.numpy(), 'r--', label='Predicted', linewidth=2)
    plt.title('Sample Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quick_test_results/quick_test_results.png', dpi=150, bbox_inches='tight')
    print("   结果保存到: quick_test_results/quick_test_results.png")

    # 8. 总结
    print("\n" + "=" * 50)
    print("📋 测试总结:")
    print(f"   ✅ 模型创建: 成功")
    print(f"   ✅ 前向传播: 成功")
    print(f"   ✅ 训练过程: 成功 (损失从 {losses[0]:.4f} 降到 {losses[-1]:.4f})")
    print(f"   ✅ 鲁棒性测试: 性能下降 {degradation:.2%}")
    print(f"   ✅ 可视化: 保存到 quick_test_results/")

    if degradation < 0.5:  # 如果性能下降小于50%
        print("   🎉 鲁棒性测试通过!")
    else:
        print("   ⚠️  鲁棒性需要改进")

    print("\n🎯 下一步建议:")
    print("   1. 运行完整训练: python full_training.py")
    print("   2. 调整超参数: lambda_mi, learning_rate等")
    print("   3. 在真实数据上测试")

    return {
        'model': model,
        'train_losses': losses,
        'clean_mse': clean_mse,
        'noisy_mse': noisy_mse,
        'degradation': degradation
    }


def main():
    """主函数"""
    try:
        results = quick_test()
        print("\n✅ 快速测试完成!")
        return results
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 运行测试
    results = main()

    # 如果测试成功，显示图表
    if results is not None:
        try:
            plt.show()
        except:
            print("无法显示图表，但已保存到文件")