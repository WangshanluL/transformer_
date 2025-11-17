"""
Decoder module for Transformer model
位于: models/decoder.py
"""
import torch
import torch.nn as nn
from modules.attention import MultiHeadAttention
from modules.feedforward import PoswiseFeedForwardNet
from modules.embedding import get_sin_enc_table
from modules.mask import get_attn_pad_mask, get_attn_subsequent_mask


class DecoderLayer(nn.Module):
    """解码器层"""
    
    def __init__(self, d_embedding=512, n_heads=8, d_k=64, d_v=64, d_ff=2048):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_embedding, n_heads, d_k, d_v)
        self.dec_enc_attn = MultiHeadAttention(d_embedding, n_heads, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_embedding, d_ff)
        
    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        Args:
            dec_inputs: [batch_size, target_len, embedding_dim]
            enc_outputs: [batch_size, source_len, embedding_dim]
            dec_self_attn_mask: [batch_size, target_len, target_len]
            dec_enc_attn_mask: [batch_size, target_len, source_len]
        Returns:
            dec_outputs: [batch_size, target_len, embedding_dim]
            dec_self_attn: [batch_size, n_heads, target_len, target_len]
            dec_enc_attn: [batch_size, n_heads, target_len, source_len]
        """
        # 将相同的 Q，K，V 输入多头自注意力层
        dec_outputs, dec_self_attn = self.dec_self_attn(
            dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask
        )
        
        # 将解码器输出和编码器输出输入多头自注意力层
        dec_outputs, dec_enc_attn = self.dec_enc_attn(
            dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask
        )
        
        # 输入位置前馈神经网络层
        dec_outputs = self.pos_ffn(dec_outputs)
        
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    """解码器"""
    
    def __init__(self, tgt_vocab_size, tgt_len, d_embedding=512, n_layers=6, 
                 n_heads=8, d_k=64, d_v=64, d_ff=2048):
        super(Decoder, self).__init__()
        self.d_embedding = d_embedding
        self.n_layers = n_layers
        
        # 词嵌入层
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_embedding)
        # 位置嵌入层
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sin_enc_table(tgt_len + 1, d_embedding), freeze=True
        )
        # 解码器层堆叠
        self.layers = nn.ModuleList([
            DecoderLayer(d_embedding, n_heads, d_k, d_v, d_ff) 
            for _ in range(n_layers)
        ])
        
    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        Args:
            dec_inputs: [batch_size, target_len]
            enc_inputs: [batch_size, source_len]
            enc_outputs: [batch_size, source_len, embedding_dim]
        Returns:
            dec_outputs: [batch_size, target_len, embedding_dim]
            dec_self_attns: list of [batch_size, n_heads, target_len, target_len]
            dec_enc_attns: list of [batch_size, n_heads, target_len, source_len]
        """
        # 创建一个从 1 到 target_len 的位置索引序列
        pos_indices = torch.arange(1, dec_inputs.size(1) + 1).unsqueeze(0).to(dec_inputs)
        
        # 对输入进行词嵌入和位置嵌入相加
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos_indices)
        
        # 生成解码器自注意力掩码和解码器-编码器注意力掩码
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt(
            (dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0
        )
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        
        dec_self_attns, dec_enc_attns = [], []
        # 通过解码器层
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(
                dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask
            )
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        
        return dec_outputs, dec_self_attns, dec_enc_attns