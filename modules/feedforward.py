import torch.nn as nn


class PoswiseFeedForwardNet(nn.Module):
    """逐位置前馈神经网络（Linear版）"""

    def __init__(self, d_embedding=512, d_ff=2048):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_embedding = d_embedding
        self.d_ff = d_ff

        # 用 Linear 代替 Conv1d(kernel=1)
        self.linear1 = nn.Linear(d_embedding, d_ff)
        self.linear2 = nn.Linear(d_ff, d_embedding)
        self.relu = nn.ReLU()
        # LayerNorm
        self.layer_norm = nn.LayerNorm(d_embedding)

    def forward(self, inputs):
        """
        Args:
            inputs: [batch, seq_len, embedding_dim]
        Returns:
            output: [batch, seq_len, embedding_dim]
        """
        residual = inputs

        # 标准 Transformer FFN：Linear → ReLU → Linear
        output = self.linear1(inputs)
        output = self.relu(output)
        output = self.linear2(output)

        # 残差 + LayerNorm
        output = self.layer_norm(output + residual)

        return output
