"""
Encoder module for Transformer model
位于: models/encoder.py
"""
import torch
import torch.nn as nn
from modules.attention import MultiHeadAttention
from modules.feedforward import PoswiseFeedForwardNet
from modules.embedding import get_sin_enc_table
from modules.mask import get_attn_pad_mask


class EncoderLayer(nn.Module):
    """编码器层"""
    
    def __init__(self, d_embedding=512, n_heads=8, d_k=64, d_v=64, d_ff=2048):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_embedding, n_heads, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_embedding, d_ff)
        
    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        Args:
            enc_inputs: [batch_size, seq_len, embedding_dim]
            enc_self_attn_mask: [batch_size, seq_len, seq_len]
        Returns:
            enc_outputs: [batch_size, seq_len, embedding_dim]
            attn_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        # 将相同的 Q，K，V 输入多头自注意力层
        enc_outputs, attn_weights = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask
        )
        
        # 将多头自注意力 outputs 输入位置前馈神经网络层
        enc_outputs = self.pos_ffn(enc_outputs)
        
        return enc_outputs, attn_weights


class Encoder(nn.Module):
    """编码器"""
    
    def __init__(self, src_vocab_size, src_len, d_embedding=512, n_layers=6, 
                 n_heads=8, d_k=64, d_v=64, d_ff=2048):
        super(Encoder, self).__init__()
        self.d_embedding = d_embedding
        self.n_layers = n_layers
        
        # 词嵌入层
        self.src_emb = nn.Embedding(src_vocab_size, d_embedding)
        # 位置嵌入层
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sin_enc_table(src_len + 1, d_embedding), freeze=True
        )
        # 编码器层堆叠
        self.layers = nn.ModuleList([
            EncoderLayer(d_embedding, n_heads, d_k, d_v, d_ff) 
            for _ in range(n_layers)
        ])
        
    def forward(self, enc_inputs):
        """
        Args:
            enc_inputs: [batch_size, source_len]
        Returns:
            enc_outputs: [batch_size, seq_len, embedding_dim]
            enc_self_attn_weights: list of [batch_size, n_heads, seq_len, seq_len]
        """
        # 创建一个从 1 到 source_len 的位置索引序列
        pos_indices = torch.arange(1, enc_inputs.size(1) + 1).unsqueeze(0).to(enc_inputs)
        
        # 对输入进行词嵌入和位置嵌入相加
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(pos_indices)
        
        # 生成自注意力掩码
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        
        enc_self_attn_weights = []
        # 通过编码器层
        for layer in self.layers:
            enc_outputs, enc_self_attn_weight = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attn_weights.append(enc_self_attn_weight)
        
        return enc_outputs, enc_self_attn_weights