"""BiLSTM baseline for GuardChat Task 1 (multi-label unsafe text recognition).

Architecture is a PyTorch port of the stacked Bidirectional-LSTM in
``vendors/BiLSTM/sentiment_analysis.py``, adapted for the six-category
multi-label setting. Output schemas, metrics, and the
``RecognitionPipeline`` API mirror :mod:`src.SafeGuider` so that
benchmarks can swap baselines without changing the surrounding code.
"""

from .model import BiLSTMClassifier, BiLSTMConfig
from .recognition import (
    RecognitionPipeline,
    RecognitionPrediction,
    RecognitionTrainer,
    TrainConfig,
)
from .tokenizer import (
    PAD_INDEX,
    PAD_TOKEN,
    UNK_INDEX,
    UNK_TOKEN,
    Vocab,
    basic_tokenize,
    preprocess_text,
)

__all__ = [
    "BiLSTMClassifier",
    "BiLSTMConfig",
    "RecognitionPipeline",
    "RecognitionPrediction",
    "RecognitionTrainer",
    "TrainConfig",
    "Vocab",
    "PAD_TOKEN",
    "UNK_TOKEN",
    "PAD_INDEX",
    "UNK_INDEX",
    "basic_tokenize",
    "preprocess_text",
]
