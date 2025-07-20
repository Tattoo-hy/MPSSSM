import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import wandb
from tqdm import tqdm
import os
import json


class TimeSeriesDataset:
    """时间序列数据集生成器"""

    @staticmethod
    def generate_synthetic_data(n_samples: int = 1000, seq_len: int = 512,
                                d_input: int = 1, noise_std: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成合成时间序列数据"""
        t = torch.linspace(0, 4 * np.pi, seq_len)

        data = []
        targets = []

        for i in range(n_samples):
            # 生成多种频率的正弦波组合
            freq1 = 0.5 + 0.5 * torch.rand(1)
            freq2 = 2.0 + 2.0 * torch.rand(1)
            phase1 = 2 * np.pi * torch.rand(1)
            phase2 = 2 * np.pi * torch.rand(1)

            signal = (torch.sin(freq1 * t + phase1) +
                      0.5 * torch.sin(freq2 * t + phase2))

            # 添加噪声
            noise = noise_std * torch.randn_like(signal)
            noisy_signal = signal + noise

            if d_input > 1:
                # 多维输入：添加额外特征
                extra_features = torch.randn(seq_len, d_input - 1) * 0.1
                input_seq = torch.cat([noisy_signal.unsqueeze(-1), extra_features], dim=-1)
            else:
                input_seq = noisy_signal.unsqueeze(-1)

            # 预测目标：下一个时刻的干净信号
            target_seq = torch.roll(signal, -1).unsqueeze(-1)
            target_seq[-1] = target_seq[-2]  # 最后一个值复制

            data.append(input_seq)
            targets.append(target_seq)

        return torch.stack(data), torch.stack(targets)

    @staticmethod
    def generate_selective_copying_task(n_samples: int = 1000, seq_len: int = 100,
                                        vocab_size: int = 10, n_targets: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成选择性复制任务数据"""
        data = []
        targets = []

        for _ in range(n_samples):
            # 生成随机序列
            seq = torch.randint(0, vocab_size, (seq_len,))

            # 随机选择目标位置
            target_positions = torch.randperm(seq_len // 2)[:n_targets]
            target_positions = target_positions.sort()[0]

            # 标记目标（使用特殊token vocab_size）
            marked_seq = seq.clone()
            for pos in target_positions:
                marked_seq[pos + seq_len // 2] = vocab_size  # 标记符

            # 目标是按顺序复制标记的数字
            target_seq = torch.zeros(seq_len)
            for i, pos in enumerate(target_positions):
                if seq_len - n_targets + i < seq_len:
                    target_seq[seq_len - n_targets + i] = seq[pos]

            # 转换为one-hot或embedding format
            input_seq = torch.nn.functional.one_hot(marked_seq, vocab_size + 1).float()

            data.append(input_seq)
            targets.append(target_seq.unsqueeze(-1))

        return torch.stack(data), torch.stack(targets)

    @staticmethod
    def add_non_causal_noise(data: torch.Tensor, noise_type: str = "impulse",
                             noise_ratio: float = 0.05, noise_strength: float = 5.0) -> torch.Tensor:
        """添加非因果扰动"""
        noisy_data = data.clone()
        batch_size, seq_len, d_input = data.shape

        if noise_type == "impulse":
            # 脉冲噪声
            n_noise_points = int(seq_len * noise_ratio)
            for b in range(batch_size):
                noise_positions = torch.randperm(seq_len)[:n_noise_points]
                for pos in noise_positions:
                    # 添加脉冲
                    impulse = noise_strength * data[b].std() * torch.randn(d_input)
                    noisy_data[b, pos] += impulse

        elif noise_type == "spurious":
            # 伪相关噪声（高频正弦波）
            t = torch.linspace(0, 4 * np.pi, seq_len)
            for b in range(batch_size):
                freq = 10.0 + 5.0 * torch.rand(1)  # 高频
                phase = 2 * np.pi * torch.rand(1)
                spurious_signal = noise_strength * 0.1 * torch.sin(freq * t + phase)
                noisy_data[b, :, 0] += spurious_signal

        return noisy_data


class MPSSSMTrainer:
    """MPS-SSM训练器"""

    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-5)
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('max_epochs', 100),
            eta_min=config.get('min_lr', 1e-6)
        )

        # 训练历史
        self.train_history = {
            'train_loss': [], 'train_pred_loss': [], 'train_mi_loss': [],
            'val_loss': [], 'val_pred_loss': [], 'val_mi_loss': []
        }

        # 最佳模型保存
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_pred_loss = 0.0
        total_mi_loss = 0.0
        n_batches = 0

        for batch_idx, (data, targets) in enumerate(tqdm(train_loader, desc="Training")):
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # 前向传播
            result = self.model(data, targets)
            total_loss_batch = result['total_loss']

            # 反向传播
            total_loss_batch.backward()

            # 梯度裁剪
            if self.config.get('grad_clip', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )

            self.optimizer.step()

            # 统计
            total_loss += total_loss_batch.item()
            total_pred_loss += result['pred_loss'].item()
            total_mi_loss += result['mi_loss'].item()
            n_batches += 1

        return {
            'train_loss': total_loss / n_batches,
            'train_pred_loss': total_pred_loss / n_batches,
            'train_mi_loss': total_mi_loss / n_batches
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        total_pred_loss = 0.0
        total_mi_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for data, targets in tqdm(val_loader, desc="Validation"):
                data, targets = data.to(self.device), targets.to(self.device)

                result = self.model(data, targets)

                total_loss += result['total_loss'].item()
                total_pred_loss += result['pred_loss'].item()
                total_mi_loss += result['mi_loss'].item()
                n_batches += 1

        return {
            'val_loss': total_loss / n_batches,
            'val_pred_loss': total_pred_loss / n_batches,
            'val_mi_loss': total_mi_loss / n_batches
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """完整训练流程"""
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.config['max_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['max_epochs']}")

            # 训练
            train_metrics = self.train_epoch(train_loader)

            # 验证
            val_metrics = self.validate(val_loader)

            # 学习率调度
            self.scheduler.step()

            # 记录历史
            for key, value in train_metrics.items():
                self.train_history[key].append(value)
            for key, value in val_metrics.items():
                self.train_history[key].append(value)

            # 保存最佳模型
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.best_model_state = self.model.state_dict().copy()
                print(f"🎉 New best model! Val loss: {self.best_val_loss:.6f}")

            # 打印指标
            print(f"Train - Total: {train_metrics['train_loss']:.6f}, "
                  f"Pred: {train_metrics['train_pred_loss']:.6f}, "
                  f"MI: {train_metrics['train_mi_loss']:.6f}")
            print(f"Val - Total: {val_metrics['val_loss']:.6f}, "
                  f"Pred: {val_metrics['val_pred_loss']:.6f}, "
                  f"MI: {val_metrics['val_mi_loss']:.6f}")

            # Wandb记录（如果启用）
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch + 1,
                    'lr': self.scheduler.get_last_lr()[0],
                    **train_metrics,
                    **val_metrics
                })

        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model with val loss: {self.best_val_loss:.6f}")

        return self.train_history

    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'train_history': self.train_history,
            'best_val_loss': self.best_val_loss
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_history = checkpoint.get('train_history', {})
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Model loaded from {filepath}")


class RobustnessEvaluator:
    """鲁棒性评估器"""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def evaluate_robustness(self, test_data: torch.Tensor, test_targets: torch.Tensor,
                            noise_types: List[str] = ["impulse", "spurious"],
                            noise_ratios: List[float] = [0.01, 0.05, 0.1]) -> Dict:
        """评估对非因果扰动的鲁棒性"""
        self.model.eval()
        results = {}

        # 干净数据性能
        with torch.no_grad():
            clean_result = self.model(test_data.to(self.device),
                                      test_targets.to(self.device))
            clean_mse = clean_result['pred_loss'].item()

        results['clean_mse'] = clean_mse

        # 不同噪声类型和强度的性能
        for noise_type in noise_types:
            results[noise_type] = {}
            for noise_ratio in noise_ratios:
                # 添加噪声
                noisy_data = TimeSeriesDataset.add_non_causal_noise(
                    test_data, noise_type=noise_type, noise_ratio=noise_ratio
                )

                with torch.no_grad():
                    noisy_result = self.model(noisy_data.to(self.device),
                                              test_targets.to(self.device))
                    noisy_mse = noisy_result['pred_loss'].item()

                # 计算性能下降率
                degradation_rate = (noisy_mse - clean_mse) / clean_mse
                results[noise_type][noise_ratio] = {
                    'mse': noisy_mse,
                    'degradation_rate': degradation_rate
                }

        return results


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 总损失
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 预测损失
    axes[0, 1].plot(history['train_pred_loss'], label='Train')
    axes[0, 1].plot(history['val_pred_loss'], label='Validation')
    axes[0, 1].set_title('Prediction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 互信息损失
    axes[1, 0].plot(history['train_mi_loss'], label='Train')
    axes[1, 0].plot(history['val_mi_loss'], label='Validation')
    axes[1, 0].set_title('Mutual Information Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 损失比例
    train_ratio = np.array(history['train_mi_loss']) / np.array(history['train_pred_loss'])
    val_ratio = np.array(history['val_mi_loss']) / np.array(history['val_pred_loss'])
    axes[1, 1].plot(train_ratio, label='Train MI/Pred Ratio')
    axes[1, 1].plot(val_ratio, label='Val MI/Pred Ratio')
    axes[1, 1].set_title('MI/Prediction Loss Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主训练函数"""
    # 配置
    config = {
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'max_epochs': 50,
        'grad_clip': 1.0,
        'lambda_mi': 0.01,
        'use_wandb': False,  # 设置为True使用wandb记录
    }

    # 数据参数
    data_config = {
        'n_train': 5000,
        'n_val': 1000,
        'n_test': 500,
        'seq_len': 256,
        'd_input': 4,
        'd_output': 1,
        'noise_std': 0.1
    }

    print("🚀 Starting MPS-SSM training...")

    # 1. 生成数据
    print("📊 Generating data...")
    train_data, train_targets = TimeSeriesDataset.generate_synthetic_data(
        n_samples=data_config['n_train'],
        seq_len=data_config['seq_len'],
        d_input=data_config['d_input'],
        noise_std=data_config['noise_std']
    )

    val_data, val_targets = TimeSeriesDataset.generate_synthetic_data(
        n_samples=data_config['n_val'],
        seq_len=data_config['seq_len'],
        d_input=data_config['d_input'],
        noise_std=data_config['noise_std']
    )

    test_data, test_targets = TimeSeriesDataset.generate_synthetic_data(
        n_samples=data_config['n_test'],
        seq_len=data_config['seq_len'],
        d_input=data_config['d_input'],
        noise_std=data_config['noise_std']
    )

    # 2. 创建数据加载器
    train_dataset = TensorDataset(train_data, train_targets)
    val_dataset = TensorDataset(val_data, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=2)

    # 3. 创建模型
    from mps_ssm_implementation import MPSSSM  # 假设之前的代码在这个文件中

    model = MPSSSM(
        d_input=data_config['d_input'],
        d_output=data_config['d_output'],
        d_state=16,
        n_layers=4,
        lambda_mi=config['lambda_mi']
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # 4. 初始化wandb（可选）
    if config.get('use_wandb', False):
        wandb.init(
            project="mps-ssm-experiments",
            config={**config, **data_config}
        )

    # 5. 训练
    trainer = MPSSSMTrainer(model, config)
    history = trainer.train(train_loader, val_loader)

    # 6. 保存模型
    os.makedirs('models', exist_ok=True)
    trainer.save_model('models/mps_ssm_best.pth')

    # 7. 绘制训练历史
    os.makedirs('figures', exist_ok=True)
    plot_training_history(history, 'figures/training_history.png')

    # 8. 鲁棒性评估
    print("\n🧪 Evaluating robustness...")
    evaluator = RobustnessEvaluator(model, trainer.device)
    robustness_results = evaluator.evaluate_robustness(test_data, test_targets)

    print("\n📈 Robustness Results:")
    print(f"Clean MSE: {robustness_results['clean_mse']:.6f}")

    for noise_type in ['impulse', 'spurious']:
        print(f"\n{noise_type.title()} Noise:")
        for ratio, metrics in robustness_results[noise_type].items():
            print(f"  Ratio {ratio}: MSE={metrics['mse']:.6f}, "
                  f"Degradation={metrics['degradation_rate']:.2%}")

    # 9. 保存结果
    with open('results/robustness_results.json', 'w') as f:
        json.dump(robustness_results, f, indent=2)

    print("\n✅ Training completed!")

    if config.get('use_wandb', False):
        wandb.finish()


if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    main()