#!/usr/bin/env python3
"""
Standalone script to test a pre-trained MPS-SSM model on a given dataset.
Now robustly handles both old (incomplete) and new (complete) model checkpoints.
"""

import os
import sys
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, Any

from models.mps_ssm import MPSSSM
from data_provider.data_loader import get_dataloader
from core.metrics import calculate_metrics

def test_model(args: argparse.Namespace):
    """
    Loads a model and evaluates its performance on the test set.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"--- 使用设备: {device} ---")

    try:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件未找到于 '{args.config}'"); sys.exit(1)

    print(f"\n--- 准备数据集: {args.dataset} ---")
    data_config = {
        'dataset': args.dataset, 'pred_len': args.pred_len,
        'batch_size': yaml_config['training']['batch_size'],
        'seq_len': yaml_config['data']['seq_len'],
        'data_path': yaml_config['data']['data_path'],
    }
    try:
        test_loader = get_dataloader(data_config, mode='test')
        print("测试数据加载成功。")
    except Exception as e:
        print(f"错误: 加载数据时出错: {e}"); sys.exit(1)

    print(f"\n--- 加载模型: {args.model_path} ---")
    try:
        # Set weights_only=False to load older pickles, but be aware of the security risk.
        checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
        
        # Smartly merge configurations for robust loading
        final_model_config = yaml_config['model'].copy()
        config_from_checkpoint = checkpoint.get('model_config', {})
        final_model_config.update(config_from_checkpoint)

        # Ensure essential parameters from command line/data are set.
        final_model_config['pred_len'] = args.pred_len
        # Determine enc_in based on dataset name as a fallback
        if args.dataset.startswith('ETT'):
             final_model_config['enc_in'] = 7
        elif args.dataset == 'weather':
             final_model_config['enc_in'] = 21
        elif args.dataset == 'traffic':
             final_model_config['enc_in'] = 862
        else:
            # If not a known dataset, it must be in the config
            if 'enc_in' not in final_model_config:
                 raise ValueError("Could not determine 'enc_in' for the model.")

        print("使用最终模型配置:", final_model_config)
        model = MPSSSM(**final_model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("模型加载并设置成功。")
    except FileNotFoundError:
        print(f"错误: 模型文件未找到于 '{args.model_path}'"); sys.exit(1)
    except Exception as e:
        print(f"错误: 加载模型时出错: {e}"); sys.exit(1)

    print("\n--- 开始在测试集上评估 ---")
    all_predictions, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="正在测试"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_predictions.append(outputs['prediction'].cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    print("评估完成。")

    print("\n--- 最终性能指标 ---")
    final_metrics = calculate_metrics(all_predictions, all_targets)
    print(f"  均方误差 (MSE): {final_metrics['mse']:.4f}")
    print(f"  平均绝对误差 (MAE): {final_metrics['mae']:.4f}")
    print(f"  均方根误差 (RMSE): {final_metrics['rmse']:.4f}")
    print(f"  平均绝对百分比误差 (MAPE): {final_metrics['mape']:.2f}%")
    print(f"  相关系数 (Correlation): {final_metrics['correlation']:.4f}")
    print("--------------------------\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="独立测试 MPS-SSM 模型性能的脚本")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--pred_len', type=int, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    test_model(args)
