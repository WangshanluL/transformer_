"""
Configuration file for Transformer model
位于: config/config.py
"""


class TransformerConfig:
    """Transformer 模型配置"""
    
    # 模型架构参数
    d_embedding = 512      # 词嵌入维度
    n_layers = 6           # 编码器/解码器层数
    n_heads = 8            # 多头注意力头数
    d_k = 64               # Q, K 的维度
    d_v = 64               # V 的维度
    d_ff = 2048            # 前馈网络隐藏层维度
    
    # 训练参数
    batch_size = 3         # 批次大小
    epochs = 5             # 训练轮数
    learning_rate = 0.0001 # 学习率
    
    # 数据参数
    # 这些参数会在创建语料库后动态设置
    src_vocab_size = None  # 源语言词汇表大小
    tgt_vocab_size = None  # 目标语言词汇表大小
    src_len = None         # 源序列最大长度
    tgt_len = None         # 目标序列最大长度
    
    @classmethod
    def update_from_corpus(cls, corpus):
        """从语料库更新配置"""
        cls.src_vocab_size = len(corpus.src_vocab)
        cls.tgt_vocab_size = len(corpus.tgt_vocab)
        cls.src_len = corpus.src_len
        cls.tgt_len = corpus.tgt_len
        
    @classmethod
    def get_model_params(cls):
        """获取模型初始化参数"""
        return {
            'src_vocab_size': cls.src_vocab_size,
            'tgt_vocab_size': cls.tgt_vocab_size,
            'src_len': cls.src_len,
            'tgt_len': cls.tgt_len,
            'd_embedding': cls.d_embedding,
            'n_layers': cls.n_layers,
            'n_heads': cls.n_heads,
            'd_k': cls.d_k,
            'd_v': cls.d_v,
            'd_ff': cls.d_ff
        }