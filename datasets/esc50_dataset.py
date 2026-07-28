"""
ESC-50 Dataset
Returns raw waveform and label.
"""

from pathlib import Path

import librosa
import pandas as pd
import torch
from torch.utils.data import Dataset

from configs.config import (
    AUDIO_DIR,
    META_FILE,
    SAMPLE_RATE,
    MAX_AUDIO_LENGTH,
)


class ESC50Dataset(Dataset):

    def __init__(self):
        self.metadata = pd.read_csv(META_FILE)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        row = self.metadata.iloc[index]

        audio_path = AUDIO_DIR / row["filename"]

        waveform, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        if len(waveform) > MAX_AUDIO_LENGTH:
            waveform = waveform[:MAX_AUDIO_LENGTH]

        else:
            waveform = librosa.util.fix_length(
                waveform,
                size=MAX_AUDIO_LENGTH
            )

        waveform = torch.tensor(
            waveform,
            dtype=torch.float32
        )

        label = torch.tensor(
            row["target"],
            dtype=torch.long
        )

        return waveform, label