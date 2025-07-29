#!/usr/bin/env python3
"""
MPS-SSM Project: Single Experiment Runner
Final version with robust GPU selection, model loading, and dynamic feature dimensioning.
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

from data_provider.data_loader import get_dataloader
from models.mps_ssm import MPSSSM
from core.engine import train_one_epoch, evaluate
from core.utils import EarlyStopping, set_random_seed, save_model
from core.metrics import calculate_metrics

from functools import partial
from data_provider.data_loader import TimeSeriesDataset  # 需要直接使用Dataset类
from torch.utils.data import DataLoader
# 从 robustness.py 导入噪声函数
from data_provider.robustness import add_impulse_noise, add_spurious_correlation

class ExperimentRunner:
    """Manages single experiment execution"""
    
    def __init__(self, args):
        self.args = args
        
        # --- FIX: Robust GPU selection ---
        # Set the visible device based on the --gpu_id argument.
        # This ensures the script uses the correct GPU whether run directly or by a scheduler.
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        # Now, 'cuda:0' will always refer to the correct, isolated GPU.
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # ---------------------------------
        
        with open(args.config, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.setup_paths()
        set_random_seed(self.config['training']['seed'])
        
    def setup_paths(self):
        if self.args.mode == 'lambda_search':
            self.result_dir = Path("results/lambda_search")
            self.result_file = self.result_dir / f"{self.args.dataset}_{self.args.pred_len}_{self.args.lambda_val}.json"
            self.model_dir = Path("results/lambda_search_models")
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.model_path = self.model_dir / f"{self.args.dataset}_{self.args.pred_len}_{self.args.lambda_val}.pth"
        else: # test_only
            self.result_dir = Path("results/final_runs")
            self.model_dir = Path("results/lambda_search_models")
            self.log_file = self.result_dir / "logs" / f"{self.args.dataset}_{self.args.pred_len}.json"
            
    def load_data(self, mode):
        data_config = {
            'dataset': self.args.dataset, 'pred_len': self.args.pred_len,
            'batch_size': self.config['training']['batch_size'], 'seq_len': self.config['data']['seq_len'],
            'data_path': self.config['data']['data_path'],
        }
        if mode == 'train':
            train_loader = get_dataloader(data_config, mode='train')
            val_loader = get_dataloader(data_config, mode='val')
            return train_loader, val_loader
        else: # test
            test_loader = get_dataloader(data_config, mode='test')
            return test_loader
        
    def create_model(self, model_config: Dict) -> MPSSSM:
        return MPSSSM(**model_config).to(self.device)
        
    def train_model(self, model, train_loader, val_loader):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['training']['learning_rate'])
        early_stopping = EarlyStopping(
            patience=self.config['training']['patience'],
            save_path=str(self.model_path)
        )
        for epoch in range(self.config['training']['max_epochs']):
            train_loss = train_one_epoch(model, train_loader, optimizer, self.device, self.args.lambda_val)
            val_loss, _ = evaluate(model, val_loader, self.device)
            print(f"Epoch {epoch+1}/{self.config['training']['max_epochs']}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break
        
        if early_stopping.best_model_state:
            print("\nTraining finished. Restoring best model state.")
            model.load_state_dict(early_stopping.best_model_state)
        else:
            print("\nTraining finished. Warning: No improvement found.")
                
        return abs(early_stopping.best_score) if early_stopping.best_score is not None else float('inf')
    
    # 在 run_experiment.py 文件中
    # 用这个修正后的版本替换掉旧的 _evaluate_robustness 函数

    def _evaluate_robustness(self, model: MPSSSM, base_mse: float) -> Dict[str, float]:
        """
        在多种噪声条件下评估模型，并计算性能下降率。
        """
        print("\n--- Starting Robustness Evaluation ---")
        robustness_metrics = {}
        
        # 准备数据加载器的基础配置
        data_config = {
            'dataset': self.args.dataset, 'pred_len': self.args.pred_len,
            'batch_size': self.config['training']['batch_size'], 'seq_len': self.config['data']['seq_len'],
            'data_path': self.config['data']['data_path'],
        }
        
        # --- 修正：正确构造文件路径 ---
        data_file = os.path.join(data_config['data_path'], f"{self.args.dataset}.csv")
        
        # 首先，创建一个干净的数据集以获取scaler的参数
        clean_dataset = TimeSeriesDataset(
            data_path=data_file, # <-- 使用修正后的路径
            mode='test', seq_len=data_config['seq_len'], pred_len=data_config['pred_len'],
            dataset_type=self.args.dataset.split('_')[0]
        )

        # 定义要测试的噪声类型
        noise_scenarios = {
            "impulse": partial(add_impulse_noise, scale=clean_dataset.scaler.scale_),
            "spurious": partial(add_spurious_correlation, scale=clean_dataset.scaler.scale_)
        }

        for noise_name, noise_fn in noise_scenarios.items():
            print(f"Evaluating with {noise_name} noise...")
            
            # 创建带有噪声函数的数据集和加载器
            noisy_dataset = TimeSeriesDataset(
                data_path=data_file, # <-- 使用修正后的路径
                mode='test', seq_len=data_config['seq_len'], pred_len=data_config['pred_len'],
                dataset_type=self.args.dataset.split('_')[0],
                noise_fn=noise_fn
            )
            noisy_loader = DataLoader(
                noisy_dataset, batch_size=data_config['batch_size'], shuffle=False, num_workers=8
            )
            
            # 在噪声数据上评估模型
            noisy_mse, _ = evaluate(model, noisy_loader, self.device)
            degradation = (noisy_mse - base_mse) / base_mse if base_mse > 0 else 0
            
            # 记录结果
            robustness_metrics[f"{noise_name}_mse"] = float(noisy_mse)
            robustness_metrics[f"{noise_name}_degradation"] = float(degradation)
            
            print(f"  -> {noise_name.capitalize()} MSE: {noisy_mse:.4f}, Degradation: {degradation:.2%}")

        return robustness_metrics
    
    def run(self):
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file_path = log_dir / f"gpu_{self.args.gpu_id}.log"

        original_stdout, original_stderr = sys.stdout, sys.stderr
        with open(log_file_path, 'w') as log_file:
            sys.stdout = log_file; sys.stderr = log_file
            try:
                print(f"--- Starting experiment at {datetime.now().isoformat()} ---")
                print(f"Dataset: {self.args.dataset}, Pred_Len: {self.args.pred_len}, Lambda: {self.args.lambda_val}, Mode: {self.args.mode}\n")
                
                if self.args.dataset.startswith('ETT'): enc_in = 7
                elif self.args.dataset == 'weather': enc_in = 21
                elif self.args.dataset == 'traffic': enc_in = 862
                else: raise ValueError(f"Unknown dataset: {self.args.dataset}")

                if self.args.mode == 'lambda_search':
                    train_loader, val_loader = self.load_data(mode='train')
                    model_config = self.config['model'].copy()
                    model_config.update({'enc_in': enc_in, 'pred_len': self.args.pred_len, 'lambda_val': self.args.lambda_val})
                    model = self.create_model(model_config)
                    best_val_loss = self.train_model(model, train_loader, val_loader)
                    test_loader = self.load_data(mode='test')
                    test_mse, test_mae = evaluate(model, test_loader, self.device)
                    results = {
                        'dataset': self.args.dataset, 'pred_len': self.args.pred_len,
                        'lambda_val': self.args.lambda_val, 'val_loss': float(best_val_loss),
                        'test_mse': float(test_mse), 'test_mae': float(test_mae),
                        'model_path': str(self.model_path), 'timestamp': datetime.now().isoformat()
                    }
                    with open(self.result_file, 'w') as f: json.dump(results, f, indent=2)
                    print(f"Lambda search complete. Results JSON saved to {self.result_file}")
                else: # test_only
                    print("--- Starting Test-Only Mode ---")
    
                    # 1. 找到此数据集/预测长度组合的最佳lambda
                    search_results_dir = Path("results/lambda_search")
                    best_lambda = None
                    best_val_loss = float('inf')

                    # 搜索匹配的lambda结果文件
                    for f in search_results_dir.glob(f"{self.args.dataset}_{self.args.pred_len}_*.json"):
                        with open(f, 'r') as res_file:
                            data = json.load(res_file)
                            if data['val_loss'] < best_val_loss:
                                best_val_loss = data['val_loss']
                                best_lambda = data['lambda_val']

                    if best_lambda is None:
                        raise FileNotFoundError(f"No lambda search results found for {self.args.dataset} with pred_len {self.args.pred_len}")

                    print(f"Found best lambda: {best_lambda} with validation loss: {best_val_loss:.4f}")

                    # 2. 构建模型路径并加载模型
                    model_path_to_load = self.model_dir / f"{self.args.dataset}_{self.args.pred_len}_{best_lambda}.pth"
                    if not model_path_to_load.exists():
                        raise FileNotFoundError(f"Best model not found at: {model_path_to_load}")

                    model_config = self.config['model'].copy()
                    model_config.update({'enc_in': enc_in, 'pred_len': self.args.pred_len, 'lambda_val': best_lambda})

                    model = self.create_model(model_config)
                    print(f"Loading model state from {model_path_to_load}")
                    
                    # 先加载整个文件（我们称之为 checkpoint）
                    checkpoint = torch.load(model_path_to_load, map_location=self.device)
                    # 然后从 checkpoint 中提取出 key 为 'model_state_dict' 的值来加载
                    model.load_state_dict(checkpoint['model_state_dict'])

                    # 3. 加载测试数据并评估
                    test_loader = self.load_data(mode='test')
                    print("Evaluating model on test data...")
                    test_mse, test_mae = evaluate(model, test_loader, self.device)
                    print(f"Test Results -> MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")
                    
                    # --- 新增：调用鲁棒性评估 ---
                    robustness_results = self._evaluate_robustness(model, base_mse=test_mse)
                    # ---------------------------
                    
                    # 4. 保存最终结果
                    self.result_dir.mkdir(parents=True, exist_ok=True)
                    (self.result_dir / "logs").mkdir(parents=True, exist_ok=True)

                    final_results = {
                        'dataset': self.args.dataset,
                        'pred_len': self.args.pred_len,
                        'best_lambda': best_lambda,  # 确保键名与summarizer脚本一致
                        'test_mse': float(test_mse),
                        'test_mae': float(test_mae),
                        'robustness': robustness_results, # <-- 将鲁棒性结果字典加入
                        'best_model_path': str(model_path_to_load),
                        'timestamp': datetime.now().isoformat()
                    }
                    with open(self.log_file, 'w') as f:
                        json.dump(final_results, f, indent=2)
                    print(f"Final test results saved to {self.log_file}")

                print(f"\n--- Experiment finished successfully at {datetime.now().isoformat()} ---")
            except Exception as e:
                print(f"\n--- AN ERROR OCCURRED: {datetime.now().isoformat()} ---")
                import traceback
                print(traceback.format_exc())
            finally:
                sys.stdout, sys.stderr = original_stdout, original_stderr

def main():
    parser = argparse.ArgumentParser(description="Run single MPS-SSM experiment")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--pred_len', type=int, required=True)
    parser.add_argument('--lambda_val', type=float, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=['lambda_search', 'test_only'])
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    runner = ExperimentRunner(args)
    runner.run()

if __name__ == "__main__":
    main()
