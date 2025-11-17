"""
Position-wise Feed-Forward Network for Transformer model
位于: modules/feedforward.py
"""
import torch.nn as nn


class PoswiseFeedForwardNet(nn.Module):
    """逐位置前馈神经网络"""
    
    def __init__(self, d_embedding=512, d_ff=2048):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_embedding = d_embedding
        self.d_ff = d_ff
        
        # 定义一维卷积层 1，用于将输入映射到更高维度
        self.conv1 = nn.Conv1d(
            in_channels=d_embedding, 
            out_channels=d_ff, 
            kernel_size=1
        )
        # 定义一维卷积层 2，用于将输入映射回原始维度
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, 
            out_channels=d_embedding, 
            kernel_size=1
        )
        # 定义层归一化
        self.layer_norm = nn.LayerNorm(d_embedding)
        
    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, len_q, embedding_dim]
        Returns:
            output: [batch_size, len_q, embedding_dim]
        """
        residual = inputs  # 保留残差连接
        
        # 在卷积层 1 后使用 ReLU 激活函数
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        
        # 使用卷积层 2 进行降维
        output = self.conv2(output).transpose(1, 2)
        
        # 与输入进行残差链接，并进行层归一化
        output = self.layer_norm(output + residual)
        
        return output