"""
Testing script for Transformer model
位于: scripts/test.py
"""
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.data_utils import TranslationCorpus
from models.transformer import Transformer
from config.config import TransformerConfig


def test(model, corpus):
    """测试函数"""
    print("\n开始测试...")
    print("=" * 50)
    
    # 创建一个大小为 1 的批次，目标语言序列 dec_inputs 在测试阶段，仅包含句子开始符号 <sos>
    enc_inputs, dec_inputs, target_batch = corpus.make_batch(batch_size=1, test_batch=True)
    
    print("编码器输入:", enc_inputs)
    print("解码器输入:", dec_inputs)
    print("目标数据  :", target_batch)
    print("-" * 50)
    
    # 用模型进行翻译
    with torch.no_grad():
        predict, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
    
    # 将预测结果维度重塑
    predict = predict.view(-1, len(corpus.tgt_vocab))
    
    # 找到每个位置概率最大的词汇的索引
    predict = predict.data.max(1, keepdim=True)[1]
    
    # 解码预测的输出，将所预测的目标句子中的索引转换为单词
    translated_sentence = [corpus.tgt_idx2word[idx.item()] for idx in predict.squeeze()]
    
    # 将输入的源语言句子中的索引转换为单词
    input_sentence = ' '.join([corpus.src_idx2word[idx.item()] for idx in enc_inputs[0]])
    
    print(f"输入句子: {input_sentence}")
    print(f"翻译结果: {translated_sentence}")
    print("=" * 50)


def main():
    """主测试函数"""
    # 定义测试数据（与训练数据相同）
    sentences = [
        ['咖哥 喜欢 小冰', 'KaGe likes XiaoBing'],
        ['我 爱 学习 人工智能', 'I love studying AI'],
        ['深度学习 改变 世界', ' DL changed the world'],
        ['自然语言处理 很 强大', 'NLP is powerful'],
        ['神经网络 非常 复杂', 'Neural-networks are complex']
    ]
    
    # 创建语料库
    print("创建语料库...")
    corpus = TranslationCorpus(sentences)
    
    # 更新配置
    TransformerConfig.update_from_corpus(corpus)
    
    # 创建模型实例
    print("初始化模型...")
    model = Transformer(**TransformerConfig.get_model_params())
    
    # 加载训练好的模型（如果有保存）
    # model.load_state_dict(torch.load('transformer_model.pth'))
    
    # 如果没有训练好的模型，需要先运行 train.py
    print("\n注意: 请确保模型已经训练过，否则预测结果可能不准确。")
    print("运行 'python scripts/train.py' 来训练模型。\n")
    
    # 测试模型
    test(model, corpus)


if __name__ == "__main__":
    main()