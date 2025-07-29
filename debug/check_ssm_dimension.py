#!/usr/bin/env python3
"""
调试MPS-SSM模型的维度问题
"""

import torch
from models import MPSSSM


def test_model_dimensions():
    """测试模型各层的维度"""
    print("MPS-SSM 维度测试")
    print("=" * 60)
    
    # 模型参数
    config = {
        'enc_in': 7,
        'pred_len': 96,
        'd_model': 512,
        'n_layers': 4,
        'd_state': 16,
        'expand_factor': 2,
        'decoder_hidden_dim': 256,
        'decoder_layers': 2,
        'lambda_val': 0.01
    }
    
    print("\n模型配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # 计算关键维度
    d_inner = config['d_model'] * config['expand_factor']
    print(f"\n计算的维度:")
    print(f"  d_inner (d_model * expand_factor): {d_inner}")
    
    # 创建模型
    model = MPSSSM(**config)
    
    # 创建测试输入
    batch_size = 4
    seq_len = 96
    dummy_input = torch.randn(batch_size, seq_len, config['enc_in'])
    
    print(f"\n输入维度: {dummy_input.shape}")
    
    # 手动追踪维度
    print("\n追踪各层维度:")
    
    # 1. Input embedding
    x_embed = model.embed_dropout(model.input_embed(dummy_input))
    print(f"1. 嵌入后: {x_embed.shape}")
    
    # 2. 通过SSM blocks
    hidden = x_embed
    for i, (ssm_block, layer_norm) in enumerate(zip(model.ssm_blocks, model.layer_norms)):
        print(f"\n第 {i+1} 层 SSM:")
        
        # 获取SSM输出
        with torch.no_grad():
            ssm_out, ssm_hidden = ssm_block(hidden)
        
        print(f"  - SSM输入: {hidden.shape}")
        print(f"  - SSM内部状态 (d_inner): {ssm_hidden.shape}")
        print(f"  - SSM输出 (d_model): {ssm_out.shape}")
        
        # Residual + LayerNorm
        hidden = layer_norm(hidden + ssm_out)
        print(f"  - 层输出 (after residual + norm): {hidden.shape}")
    
    # 3. 最小性正则化器的输入
    middle_layer_idx = len(model.ssm_blocks) // 2
    print(f"\n最小性正则化器:")
    print(f"  - 使用第 {middle_layer_idx + 1} 层的输出")
    print(f"  - 期望输入维度: (batch, seq_len, {config['d_model']})")
    
    # 4. 预测头
    final_hidden = hidden[:, -1, :]
    print(f"\n预测头:")
    print(f"  - 输入 (最后时刻): {final_hidden.shape}")
    
    # 完整前向传播测试
    print("\n\n完整前向传播测试:")
    try:
        with torch.no_grad():
            outputs = model(dummy_input)
        print("✓ 前向传播成功!")
        print(f"  - 预测输出: {outputs['prediction'].shape}")
        print(f"  - 重构输入: {outputs['reconstructed_input'].shape}")
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()


def test_minimality_regularizer():
    """单独测试MinimalityRegularizer"""
    from models.mps_ssm import MinimalityRegularizer
    
    print("\n\nMinimalityRegularizer 测试")
    print("=" * 60)
    
    # 创建正则化器
    hidden_dim = 512
    input_dim = 7
    regularizer = MinimalityRegularizer(
        hidden_dim=hidden_dim,
        input_dim=input_dim,
        decoder_hidden_dim=256,
        decoder_layers=2
    )
    
    # 测试输入
    batch_size = 4
    seq_len = 96
    test_input = torch.randn(batch_size, seq_len, hidden_dim)
    
    print(f"输入形状: {test_input.shape}")
    
    try:
        output = regularizer(test_input)
        print(f"✓ 输出形状: {output.shape}")
        print(f"  期望形状: ({batch_size}, {seq_len}, {input_dim})")
    except Exception as e:
        print(f"✗ 错误: {e}")


def check_parameter_count():
    """检查模型参数数量"""
    print("\n\n模型参数统计")
    print("=" * 60)
    
    model = MPSSSM(
        enc_in=7,
        pred_len=96,
        d_model=512,
        n_layers=4,
        lambda_val=0.01
    )
    
    total_params = 0
    trainable_params = 0
    
    print("\n各模块参数数量:")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name}: {module_params:,} (可训练: {module_trainable:,})")
        total_params += module_params
        trainable_params += module_trainable
    
    print(f"\n总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (假设float32)")


def main():
    """运行所有测试"""
    test_model_dimensions()
    test_minimality_regularizer()
    check_parameter_count()


if __name__ == "__main__":
    main()