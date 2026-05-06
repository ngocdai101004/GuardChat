"""BERT baseline for GuardChat Task 1 (multi-label unsafe text recognition).

PyTorch + HuggingFace fine-tuning of BERT - retains the encoder
backbone but swaps the head from softmax single-label to sigmoid
multi-label over the six GuardChat NSFW categories. Output schemas,
metrics, and the :class:`RecognitionPipeline` API mirror
:mod:`src.SafeGuider` and :mod:`src.BiLSTM` exactly so all baselines
plug into a single benchmark aggregator.
"""

from .model import BERTClassifier, BERTConfig, DEFAULT_MODEL_NAME
from .recognition import (
    RecognitionPipeline,
    RecognitionPrediction,
    RecognitionTrainer,
    TrainConfig,
)

__all__ = [
    "BERTClassifier",
    "BERTConfig",
    "DEFAULT_MODEL_NAME",
    "RecognitionPipeline",
    "RecognitionPrediction",
    "RecognitionTrainer",
    "TrainConfig",
]
