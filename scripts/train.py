"""
Training script for Transformer model
位于: scripts/train.py
"""
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_utils import TranslationCorpus
from models.transformer import Transformer
from config.config import TransformerConfig
import matplotlib.pyplot as plt


def train():
    """训练函数"""
    # 定义训练数据
    sentences = [
        # 基础操作
        ['我 在 挖 方块', 'I am mining blocks'],
        ['苦力怕 爆炸 了', 'Creeper exploded'],
        ['末影人 传送 走 了', 'Enderman teleported away'],
        
        # 资源和工具
        ['我 需要 钻石 镐', 'I need a diamond pickaxe'],
        ['附魔台 需要 黑曜石', 'Enchanting table needs obsidian'],
        ['我 爱 你', 'I hate you'],
        
    ]
    
    # 创建语料库类实例
    print("创建语料库...")
    corpus = TranslationCorpus(sentences)
    
    # 更新配置
    TransformerConfig.update_from_corpus(corpus)
    
    # 创建模型实例
    print("初始化模型...")
    model = Transformer(**TransformerConfig.get_model_params())
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=TransformerConfig.learning_rate)
    
    # 记录epoch和loss
    epochs_list = []
    losses_list = []
    
    # 训练循环
    print("开始训练...")
    print(f"训练轮数: {TransformerConfig.epochs}")
    print(f"批次大小: {TransformerConfig.batch_size}")
    print(f"学习率: {TransformerConfig.learning_rate}")
    print("-" * 50)
    
    for epoch in range(TransformerConfig.epochs):
        optimizer.zero_grad()
        
        # 创建训练数据
        enc_inputs, dec_inputs, target_batch = corpus.make_batch(
            TransformerConfig.batch_size
        )
        
        # 获取模型输出
        outputs, _, _, _ = model(enc_inputs, dec_inputs)
        
        # 计算损失
        loss = criterion(
            outputs.view(-1, len(corpus.tgt_vocab)), 
            target_batch.view(-1)
        )
        
        # 记录epoch和loss
        epochs_list.append(epoch + 1)
        losses_list.append(loss.item())
        
        # 打印损失
        print(f"Epoch: {epoch + 1:04d} | Loss: {loss:.6f}")
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
    
    print("-" * 50)
    print("训练完成!")
    
    # 画图并保存
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, losses_list, marker='o', linestyle='-', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss over Epochs', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    print("训练曲线已保存为 'training_loss.png'")
    plt.close()
    
    return model, corpus


if __name__ == "__main__":
    # 训练模型
    model, corpus = train()
    
    # 保存模型（可选）
    torch.save(model.state_dict(), 'transformer_model.pth')
    print("\n模型已准备就绪，可以开始测试。")