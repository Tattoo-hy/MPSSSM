#!/usr/bin/env python3
"""
检查ETT数据集结构的脚本
帮助验证数据格式和列索引
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path


def check_dataset(file_path):
    """检查数据集的结构和基本信息"""
    print(f"\n检查数据集: {file_path}")
    print("=" * 60)
    
    # 读取CSV文件
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"错误: 无法读取文件 - {e}")
        return
    
    # 基本信息
    print(f"\n1. 基本信息:")
    print(f"   - 总行数: {len(df)}")
    print(f"   - 总列数: {len(df.columns)}")
    print(f"   - 文件大小: {Path(file_path).stat().st_size / 1024 / 1024:.2f} MB")
    
    # 列信息
    print(f"\n2. 列信息:")
    for i, col in enumerate(df.columns):
        print(f"   - 列 {i}: {col}")
    
    # 数据预览
    print(f"\n3. 数据预览 (前5行):")
    print(df.head())
    
    # 检查日期列
    print(f"\n4. 日期列分析:")
    date_col = df.columns[0]
    print(f"   - 日期列名: {date_col}")
    print(f"   - 第一个日期: {df[date_col].iloc[0]}")
    print(f"   - 最后一个日期: {df[date_col].iloc[-1]}")
    
    # 数据列分析（跳过日期列）
    print(f"\n5. 数据列分析 (跳过日期列):")
    data_cols = df.columns[1:]
    print(f"   - 数据列数: {len(data_cols)}")
    print(f"   - 数据列索引范围: 0 到 {len(data_cols)-1} (去除日期列后)")
    
    # 目标列（OT）
    if 'OT' in data_cols:
        ot_index = list(data_cols).index('OT')
        print(f"   - OT列索引: {ot_index} (去除日期列后)")
    else:
        print("   - 警告: 未找到OT列!")
    
    # 数据统计
    print(f"\n6. 数据统计 (数值列):")
    numeric_data = df[data_cols]
    print(numeric_data.describe())
    
    # 检查缺失值
    print(f"\n7. 缺失值检查:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("   发现缺失值:")
        for col, count in missing[missing > 0].items():
            print(f"   - {col}: {count} ({count/len(df)*100:.2f}%)")
    else:
        print("   - 没有缺失值")
    
    # 数据类型
    print(f"\n8. 数据类型:")
    for col in df.columns:
        print(f"   - {col}: {df[col].dtype}")
    
    # 数据集划分信息
    print(f"\n9. 数据集划分 (70%/10%/20%):")
    total_len = len(df)
    train_len = int(0.7 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len
    
    print(f"   - 训练集: {train_len} 行 (70%)")
    print(f"   - 验证集: {val_len} 行 (10%)")
    print(f"   - 测试集: {test_len} 行 (20%)")
    
    # 序列长度检查
    print(f"\n10. 序列长度可行性:")
    seq_len = 96
    pred_lens = [96, 192, 336, 720]
    
    for pred_len in pred_lens:
        min_required = seq_len + pred_len
        train_samples = train_len - min_required + 1
        val_samples = val_len - min_required + 1
        test_samples = test_len - min_required + 1
        
        print(f"\n   预测长度 {pred_len}:")
        print(f"   - 最小所需长度: {min_required}")
        print(f"   - 训练样本数: {max(0, train_samples)}")
        print(f"   - 验证样本数: {max(0, val_samples)}")
        print(f"   - 测试样本数: {max(0, test_samples)}")
        
        if train_samples <= 0 or val_samples <= 0 or test_samples <= 0:
            print(f"   - 警告: 数据不足以支持此预测长度!")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        # 默认检查所有常见数据集
        datasets = ['ETTh1.csv', 'ETTm1.csv', 'Weather.csv']
        data_dir = Path('data')
        
        print("ETT数据集结构检查工具")
        print("=" * 60)
        
        for dataset in datasets:
            file_path = data_dir / dataset
            if file_path.exists():
                check_dataset(str(file_path))
            else:
                print(f"\n跳过 {dataset} (文件不存在)")
    else:
        # 检查指定的文件
        file_path = sys.argv[1]
        check_dataset(file_path)


if __name__ == "__main__":
    main()