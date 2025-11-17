"""
Attention mechanisms for Transformer model
位于: modules/attention.py
"""
import numpy as np
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""
    
    def __init__(self, d_k=64):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        
    def forward(self, Q, K, V, attn_mask):
        """
        Args:
            Q: [batch_size, n_heads, len_q, dim_q]
            K: [batch_size, n_heads, len_k, dim_k]
            V: [batch_size, n_heads, len_v, dim_v]
            attn_mask: [batch_size, n_heads, len_q, len_k]
        Returns:
            context: [batch_size, n_heads, len_q, dim_v]
            weights: [batch_size, n_heads, len_q, len_k]
        """
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        
        # 使用注意力掩码，将 attn_mask 中值为 1 的位置的权重替换为极小值
        scores.masked_fill_(attn_mask, -1e9)
        
        # 对注意力分数进行 softmax 归一化
        weights = nn.Softmax(dim=-1)(scores)
        
        # 计算上下文向量（也就是注意力的输出）
        context = torch.matmul(weights, V)
        
        return context, weights


class MultiHeadAttention(nn.Module):
    """多头自注意力"""
    
    def __init__(self, d_embedding=512, n_heads=8, d_k=64, d_v=64):
        super(MultiHeadAttention, self).__init__()
        self.d_embedding = d_embedding
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        
        # Q, K, V 的线性变换层
        self.W_Q = nn.Linear(d_embedding, d_k * n_heads)
        self.W_K = nn.Linear(d_embedding, d_k * n_heads)
        self.W_V = nn.Linear(d_embedding, d_v * n_heads)
        
        self.linear = nn.Linear(n_heads * d_v, d_embedding)
        self.layer_norm = nn.LayerNorm(d_embedding)
        
        self.attention = ScaledDotProductAttention(d_k)
        
    def forward(self, Q, K, V, attn_mask):
        """
        Args:
            Q, K, V: [batch_size, len_q/k/v, embedding_dim]
            attn_mask: [batch_size, len_q, len_k]
        Returns:
            output: [batch_size, len_q, embedding_dim]
            weights: [batch_size, n_heads, len_q, len_k]
        """
        residual, batch_size = Q, Q.size(0)
        
        # 将输入进行线性变换和重塑，以便后续处理
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        # 将注意力掩码复制到多头
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        
        # 使用缩放点积注意力计算上下文和注意力权重
        context, weights = self.attention(q_s, k_s, v_s, attn_mask)
        
        # 通过调整维度将多个头的上下文向量连接在一起
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_v
        )
        
        # 用一个线性层把连接后的多头自注意力结果转换，原始地嵌入维度
        output = self.linear(context)
        
        # 与输入 (Q) 进行残差链接，并进行层归一化后输出
        output = self.layer_norm(output + residual)
        
        return output, weights