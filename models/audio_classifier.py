"""
Complete Audio Classification Model
"""

import torch.nn as nn

from configs.config import FREEZE_ENCODER
from models.wav2vec2_encoder import Wav2Vec2Encoder
from models.classifier import AudioClassifierHead


class AudioClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = Wav2Vec2Encoder(
            freeze_encoder=FREEZE_ENCODER
        )

        self.classifier = AudioClassifierHead()

    def forward(
        self,
        input_values,
        attention_mask=None,
    ):

        embeddings = self.encoder(
            input_values,
            attention_mask,
        )

        logits = self.classifier(
            embeddings
        )

        return logits