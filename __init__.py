"""
Transformer Project
一个使用 PyTorch 实现的 Transformer 模型，用于机器翻译任务
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

from models import Transformer
from utils import TranslationCorpus
from config import TransformerConfig

__all__ = [
    'Transformer',
    'TranslationCorpus',
    'TransformerConfig',
]