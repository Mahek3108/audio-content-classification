"""
Classification head for ESC-50.
"""

import torch.nn as nn


class AudioClassifierHead(nn.Module):

    def __init__(
        self,
        input_dim=768,
        hidden_dim=512,
        num_classes=50,
        dropout=0.3,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)