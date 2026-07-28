"""
Attention Pooling Layer
Learns to focus on the most informative audio frames.
"""

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):

    def __init__(self, hidden_size=768):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden_states):
        """
        hidden_states:
            (batch_size, sequence_length, hidden_size)
        """

        scores = self.attention(hidden_states)

        weights = torch.softmax(scores, dim=1)

        pooled = torch.sum(weights * hidden_states, dim=1)

        return pooled