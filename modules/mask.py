"""
Mask generation functions for Transformer model
位于: modules/mask.py
"""
import numpy as np
import torch


def get_attn_pad_mask(seq_q, seq_k):
    """
    定义填充注意力掩码函数
    
    Args:
        seq_q: [batch_size, len_q]
        seq_k: [batch_size, len_k]
    Returns:
        pad_attn_mask: [batch_size, len_q, len_k]
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    
    # 生成布尔类型张量，<PAD> token 的编码值为 0
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    
    # 变形为与注意力分数相同形状的张量
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)
    
    return pad_attn_mask


def get_attn_subsequent_mask(seq):
    """
    生成后续注意力掩码的函数，用于在多头自注意力计算中忽略未来信息
    
    Args:
        seq: [batch_size, seq_len(Q)=seq_len(K)]
    Returns:
        subsequent_mask: [batch_size, seq_len(Q), seq_len(K)]
    """
    # 获取输入序列的形状
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    
    # 使用 numpy 创建一个上三角矩阵（triu = triangle upper）
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    
    # 将 numpy 数组转换为 PyTorch 张量，并将数据类型设置为 byte（布尔值）
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    
    return subsequent_mask