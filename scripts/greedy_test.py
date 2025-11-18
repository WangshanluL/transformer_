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

# 定义贪婪解码器函数
def greedy_decoder(model, enc_input, start_symbol):
    # 对输入数据进行编码，并获得编码器输出以及自注意力权重
    enc_outputs, enc_self_attns = model.encoder(enc_input)    
    # 初始化解码器输入为全零张量，大小为 (1, 5)，数据类型与 enc_input 一致
    dec_input = torch.zeros(1, 7).type_as(enc_input.data)    
    # 设置下一个要解码的符号为开始符号
    next_symbol = start_symbol    
    # 循环 5 次，为解码器输入中的每一个位置填充一个符号
    for i in range(0, 7):
        # 将下一个符号放入解码器输入的当前位置
        dec_input[0][i] = next_symbol        
        # 运行解码器，获得解码器输出、解码器自注意力权重和编码器 - 解码器注意力权重
        dec_output, _, _ = model.decoder(dec_input, enc_input, enc_outputs)        
        # 将解码器输出投影到目标词汇空间
        projected = model.projection(dec_output)        
        # 找到具有最高概率的下一个单词
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]        
        # 将找到的下一个单词作为新的符号
        next_symbol = next_word.item()        
    # 返回解码器输入，它包含了生成的符号序列
    dec_outputs = dec_input
    return dec_outputs

def test(model, corpus):
    """测试函数"""
    print("\n开始测试...")
    print("=" * 50)
    
    # 创建一个大小为 1 的批次，目标语言序列 dec_inputs 在测试阶段，仅包含句子开始符号 <sos>
    enc_inputs, dec_inputs, target_batch = corpus.make_batch(batch_size=1, test_batch=True)
    
    # 使用贪婪解码器生成解码器输入
    greedy_dec_input = greedy_decoder(model, enc_inputs, start_symbol=corpus.tgt_vocab['<sos>'])
    # 将解码器输入转换为单词序列
    greedy_dec_output_words = [corpus.tgt_idx2word[n.item()] for n in greedy_dec_input.squeeze()]
    # 打印编码器输入和贪婪解码器生成的文本
    enc_inputs_words = [corpus.src_idx2word[code.item()] for code in enc_inputs[0]]
    print(enc_inputs_words, '->', greedy_dec_output_words)


def main():
    """主测试函数"""
    # 定义测试数据（与训练数据相同）
    sentences = [
        # 基础操作
        ['我 在 挖 方块', 'I am mining blocks'],
        ['苦力怕 爆炸 了', 'Creeper exploded'],
        ['末影人 传送 走 了', 'Enderman teleported away'],
        
        # 资源和工具
        ['我 需要 钻石 镐', 'I need a diamond pickaxe'],
        ['附魔台 需要 黑曜石', 'Enchanting table needs obsidian'],
        ['我 爱 你', 'I hate you'],
        
    ]
    
    # 创建语料库
    print("创建语料库...")
    corpus = TranslationCorpus(sentences)
    
    # 更新配置
    TransformerConfig.update_from_corpus(corpus)
    
    # 创建模型实例
    # print("初始化模型...")
    model = Transformer(**TransformerConfig.get_model_params())
    
    # 加载训练好的模型（如果有保存）
    model.load_state_dict(torch.load('transformer_model.pth'))
    
    # 如果没有训练好的模型，需要先运行 train.py
    print("\n注意: 请确保模型已经训练过，否则预测结果可能不准确。")
    print("运行 'python scripts/train.py' 来训练模型。\n")
    
    # 测试模型
    test(model, corpus)


if __name__ == "__main__":
    main()