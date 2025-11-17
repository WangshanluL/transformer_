# Transformer 实现

这是一个使用 PyTorch 实现的 Transformer 模型，用于机器翻译任务。代码结构清晰，模块化设计，易于理解和扩展。

## 项目结构

```
transformer_project/
├── models/                  # 模型架构
│   ├── __init__.py
│   ├── transformer.py      # Transformer主模型
│   ├── encoder.py          # 编码器
│   └── decoder.py          # 解码器
│
├── modules/                 # 基础组件
│   ├── __init__.py
│   ├── attention.py        # 注意力机制
│   ├── feedforward.py      # 前馈网络
│   ├── embedding.py        # 位置编码
│   └── mask.py             # 掩码生成
│
├── utils/                   # 工具函数
│   ├── __init__.py
│   └── data_utils.py       # 数据处理工具
│
├── config/                  # 配置文件
│   ├── __init__.py
│   └── config.py           # 模型配置
│
├── scripts/                 # 脚本文件
│   ├── train.py            # 训练脚本
│   └── test.py             # 测试脚本
│
├── requirements.txt         # 依赖列表
└── README.md               # 项目说明文档
```

## 模块说明

### 📦 models/ - 模型架构层
包含完整的模型定义
- `transformer.py`: 完整的 Transformer 模型
- `encoder.py`: 编码器及编码器层
- `decoder.py`: 解码器及解码器层

### 🔧 modules/ - 基础组件层
可复用的神经网络组件
- `attention.py`: 缩放点积注意力和多头注意力
- `feedforward.py`: 逐位置前馈神经网络
- `embedding.py`: 正弦位置编码
- `mask.py`: 各种掩码生成函数

### 🛠️ utils/ - 工具层
数据处理和辅助函数
- `data_utils.py`: 翻译语料库处理类

### ⚙️ config/ - 配置层
集中管理所有配置参数
- `config.py`: 模型超参数配置

### 🚀 scripts/ - 脚本层
可执行的训练和测试脚本
- `train.py`: 训练模型
- `test.py`: 测试模型

## 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练模型

```bash
python scripts/train.py
```

### 测试模型

```bash
python scripts/test.py
```

### 自定义训练

```python
from utils.data_utils import TranslationCorpus
from models.transformer import Transformer
from config.config import TransformerConfig

# 准备数据
sentences = [
    ['源语言句子1', '目标语言句子1'],
    ['源语言句子2', '目标语言句子2'],
    # ...
]

# 创建语料库
corpus = TranslationCorpus(sentences)

# 更新配置
config = TransformerConfig()
config.update_from_corpus(corpus)

# 创建模型
model = Transformer(
    src_vocab_size=config.src_vocab_size,
    tgt_vocab_size=config.tgt_vocab_size,
    src_len=config.src_len,
    tgt_len=config.tgt_len,
    d_embedding=config.d_embedding,
    n_layers=config.n_layers,
    n_heads=config.n_heads,
    d_k=config.d_k,
    d_v=config.d_v,
    d_ff=config.d_ff
)

# 训练模型
# ...
```

## 模型参数

默认配置：
- 嵌入维度 (d_embedding): 512
- 编码器/解码器层数 (n_layers): 6
- 多头注意力头数 (n_heads): 8
- Q, K 维度 (d_k): 64
- V 维度 (d_v): 64
- 前馈网络隐藏层维度 (d_ff): 2048
- 批次大小 (batch_size): 3
- 训练轮数 (epochs): 5
- 学习率 (learning_rate): 0.0001

## 项目特点

✅ **清晰的分层结构**：按功能分为 models、modules、utils、config、scripts
✅ **模块化设计**：每个模块职责单一，易于维护
✅ **易于扩展**：可以轻松添加新的组件或模型
✅ **配置集中管理**：所有超参数在 config 中统一设置
✅ **完善的文档**：每个模块都有详细注释

## 注意事项

1. 本实现是一个教学示例，数据集较小
2. 实际应用中需要更大的数据集和更长的训练时间
3. 可以根据需要调整模型参数和训练参数
4. 建议使用 GPU 进行训练以提高速度

## 参考文献

- Vaswani, A., et al. (2017). "Attention is All You Need."