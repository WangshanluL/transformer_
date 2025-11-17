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


def train():
    """训练函数"""
    # 定义训练数据
    sentences = [
        ['咖哥 喜欢 小冰', 'KaGe likes XiaoBing'],
        ['我 爱 学习 人工智能', 'I love studying AI'],
        ['深度学习 改变 世界', ' DL changed the world'],
        ['自然语言处理 很 强大', 'NLP is powerful'],
        ['神经网络 非常 复杂', 'Neural-networks are complex']
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
        
        # 打印损失
        print(f"Epoch: {epoch + 1:04d} | Loss: {loss:.6f}")
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
    
    print("-" * 50)
    print("训练完成!")
    
    return model, corpus


if __name__ == "__main__":
    # 训练模型
    model, corpus = train()
    
    # 保存模型（可选）
    # torch.save(model.state_dict(), 'transformer_model.pth')
    print("\n模型已准备就绪，可以开始测试。")