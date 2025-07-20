# 文件：experiments/experiment_1_selectivity.py
import torch
import numpy as np
from mps_ssm_implementation import MPSSSM
from mps_ssm_training import TimeSeriesDataset, MPSSSMTrainer
from torch.utils.data import DataLoader, TensorDataset


def selective_copying_experiment():
    """选择性复制任务实验"""
    print("🧪 实验一：选择性机制验证")

    # 生成选择性复制任务数据
    train_data, train_targets = TimeSeriesDataset.generate_selective_copying_task(
        n_samples=2000, seq_len=100, vocab_size=10, n_targets=3
    )

    test_data, test_targets = TimeSeriesDataset.generate_selective_copying_task(
        n_samples=500, seq_len=100, vocab_size=10, n_targets=3
    )

    # 模型配置
    models = {
        'MPS-SSM': MPSSSM(d_input=11, d_output=1, d_state=16, n_layers=3, lambda_mi=0.01),
        'MPS-SSM (no MI)': MPSSSM(d_input=11, d_output=1, d_state=16, n_layers=3, lambda_mi=0.0),
    }

    results = {}

    for name, model in models.items():
        print(f"\n训练 {name}...")

        config = {
            'batch_size': 32,
            'learning_rate': 1e-3,
            'max_epochs': 20,
            'grad_clip': 1.0
        }

        trainer = MPSSSMTrainer(model, config)

        # 数据加载器
        train_loader = DataLoader(
            TensorDataset(train_data, train_targets),
            batch_size=32, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(test_data, test_targets),
            batch_size=32, shuffle=False
        )

        # 训练
        history = trainer.train(train_loader, val_loader)

        # 评估准确率
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in val_loader:
                outputs = model(data)['predictions']
                predicted = torch.round(outputs)
                total += targets.numel()
                correct += (predicted == targets).sum().item()

        accuracy = correct / total
        results[name] = {
            'accuracy': accuracy,
            'final_loss': history['val_loss'][-1]
        }

        print(f"{name} 准确率: {accuracy:.2%}")

    return results


if __name__ == "__main__":
    results = selective_copying_experiment()
    print("\n📊 实验一结果:")
    for name, metrics in results.items():
        print(f"{name}: 准确率 {metrics['accuracy']:.2%}")