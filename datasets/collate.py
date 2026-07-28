"""
Collate Function for Wav2Vec2
"""

import torch
from transformers import AutoProcessor

from configs.config import MODEL_NAME, SAMPLE_RATE


class Wav2Vec2Collator:

    def __init__(self):

        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME
        )

    def __call__(self, batch):

        waveforms = [item[0].numpy() for item in batch]

        labels = torch.tensor(
            [item[1].item() for item in batch],
            dtype=torch.long
        )

        inputs = self.processor(
            waveforms,
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "input_values": inputs["input_values"],
            "attention_mask": inputs.get(
                "attention_mask",
                torch.ones_like(inputs["input_values"], dtype=torch.long)
            ),
            "labels": labels
        }