"""
Wav2Vec2 Encoder
Extracts contextual audio embeddings.
"""

import torch.nn as nn
from transformers import AutoModel

from configs.config import MODEL_NAME


class Wav2Vec2Encoder(nn.Module):

    def __init__(self, freeze_encoder=False):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(MODEL_NAME)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, input_values, attention_mask=None):

        outputs = self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state

        # Mean Pooling
        embeddings = hidden_states.mean(dim=1)

        return embeddings