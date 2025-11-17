"""
Transformer model
位于: models/transformer.py
"""
import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder


class Transformer(nn.Module):
    """Transformer 模型"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, src_len, tgt_len,
                 d_embedding=512, n_layers=6, n_heads=8, d_k=64, d_v=64, d_ff=2048):
        """
        Args:
            src_vocab_size: 源语言词汇表大小
            tgt_vocab_size: 目标语言词汇表大小
            src_len: 源序列最大长度
            tgt_len: 目标序列最大长度
            d_embedding: 嵌入维度
            n_layers: 编码器/解码器层数
            n_heads: 多头注意力头数
            d_k: Q, K 的维度
            d_v: V 的维度
            d_ff: 前馈网络隐藏层维度
        """
        super(Transformer, self).__init__()
        
        # 初始化编码器实例
        self.encoder = Encoder(src_vocab_size, src_len, d_embedding, n_layers, 
                              n_heads, d_k, d_v, d_ff)
        
        # 初始化解码器实例
        self.decoder = Decoder(tgt_vocab_size, tgt_len, d_embedding, n_layers, 
                              n_heads, d_k, d_v, d_ff)
        
        # 定义线性投影层，将解码器输出转换为目标词汇表大小的概率分布
        self.projection = nn.Linear(d_embedding, tgt_vocab_size, bias=False)
        
    def forward(self, enc_inputs, dec_inputs):
        """
        Args:
            enc_inputs: [batch_size, source_seq_len]
            dec_inputs: [batch_size, target_seq_len]
        Returns:
            dec_logits: [batch_size, tgt_seq_len, tgt_vocab_size]
            enc_self_attns: list of [batch_size, n_heads, src_seq_len, src_seq_len]
            dec_self_attns: list of [batch_size, n_heads, tgt_seq_len, tgt_seq_len]
            dec_enc_attns: list of [batch_size, n_heads, tgt_seq_len, src_seq_len]
        """
        # 将输入传递给编码器，并获取编码器输出和自注意力权重
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        
        # 将编码器输出、解码器输入和编码器输入传递给解码器
        # 获取解码器输出、解码器自注意力权重和编码器-解码器注意力权重
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(
            dec_inputs, enc_inputs, enc_outputs
        )
        
        # 将解码器输出传递给投影层，生成目标词汇表大小的概率分布
        dec_logits = self.projection(dec_outputs)
        
        # 返回逻辑值(原始预测结果), 编码器自注意力权重，解码器自注意力权重，解-编码器注意力权重
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns