# 文件：experiments/experiment_2_robustness.py
import torch
import matplotlib.pyplot as plt
import pandas as pd
from mps_ssm_implementation import MPSSSM
from mps_ssm_training import TimeSeriesDataset, MPSSSMTrainer, RobustnessEvaluator
from torch.utils.data import DataLoader, TensorDataset


def robustness_experiment():
    """鲁棒性实验 - 验证定理2"""
    print("🧪 实验二：鲁棒性验证")

    # 1. 生成数据
    train_data, train_targets = TimeSeriesDataset.generate_synthetic_data(
        n_samples=3000, seq_len=256, d_input=4, noise_std=0.1
    )

    test_data, test_targets = TimeSeriesDataset.generate_synthetic_data(
        n_samples=1000, seq_len=256, d_input=4, noise_std=0.1
    )

    # 2. 训练模型
    models = {
        'MPS-SSM (λ=0.01)': MPSSSM(d_input=4, d_output=1, lambda_mi=0.01),
        'MPS-SSM (λ=0.005)': MPSSSM(d_input=4, d_output=1, lambda_mi=0.005),
        'MPS-SSM (λ=0)': MPSSSM(d_input=4, d_output=1, lambda_mi=0.0),  # 等价于Mamba
    }

    robustness_results = {}

    for name, model in models.items():
        print(f"\n训练 {name}...")

        config = {
            'batch_size': 32,
            'learning_rate': 1e-3,
            'max_epochs': 30,
            'grad_clip': 1.0
        }

        trainer = MPSSSMTrainer(model, config)

        train_loader = DataLoader(
            TensorDataset(train_data, train_targets),
            batch_size=32, shuffle=True
        )

        # 简化验证集
        val_loader = DataLoader(
            TensorDataset(test_data[:500], test_targets[:500]),
            batch_size=32, shuffle=False
        )

        # 训练
        trainer.train(train_loader, val_loader)

        # 鲁棒性评估
        evaluator = RobustnessEvaluator(model, trainer.device)
        results = evaluator.evaluate_robustness(
            test_data, test_targets,
            noise_types=['impulse', 'spurious'],
            noise_ratios=[0.01, 0.05, 0.1, 0.2]
        )

        robustness_results[name] = results

        print(f"{name} 干净数据 MSE: {results['clean_mse']:.6f}")

    # 3. 可视化结果
    plot_robustness_results(robustness_results)

    return robustness_results


def plot_robustness_results(results):
    """绘制鲁棒性结果"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    noise_types = ['impulse', 'spurious']
    noise_ratios = [0.01, 0.05, 0.1, 0.2]

    for i, noise_type in enumerate(noise_types):
        ax = axes[i]

        for model_name, model_results in results.items():
            degradation_rates = [
                model_results[noise_type][ratio]['degradation_rate']
                for ratio in noise_ratios
            ]

            ax.plot(noise_ratios, degradation_rates,
                    marker='o', label=model_name, linewidth=2)

        ax.set_xlabel('Noise Ratio')
        ax.set_ylabel('Performance Degradation Rate')
        ax.set_title(f'{noise_type.title()} Noise Robustness')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/robustness_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    results = robustness_experiment()