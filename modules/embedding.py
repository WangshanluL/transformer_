"""
Positional Encoding for Transformer model
位于: modules/embedding.py
"""
import numpy as np
import torch


def get_sin_enc_table(n_position, embedding_dim):
    """
    生成正弦位置编码表的函数，用于在 Transformer 中引入位置信息
    
    Args:
        n_position: 输入序列的最大长度
        embedding_dim: 词嵌入向量的维度
    Returns:
        sinusoid_table: [n_position, embedding_dim] 正弦位置编码表
    """
    # 根据位置和维度信息，初始化正弦位置编码表
    sinusoid_table = np.zeros((n_position, embedding_dim))
    
    # 遍历所有位置和维度，计算角度值
    for pos_i in range(n_position):
        for hid_j in range(embedding_dim):
            angle = pos_i / np.power(10000, 2 * (hid_j // 2) / embedding_dim)
            sinusoid_table[pos_i, hid_j] = angle
    
    # 计算正弦和余弦值
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i 偶数维
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1 奇数维
    
    return torch.FloatTensor(sinusoid_table)