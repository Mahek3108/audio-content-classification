from torch.utils.data import DataLoader

from datasets.esc50_dataset import ESC50Dataset
from datasets.collate import Wav2Vec2Collator

dataset = ESC50Dataset()

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=Wav2Vec2Collator()
)

batch = next(iter(loader))

print(batch.keys())

print()

print("Input Shape :", batch["input_values"].shape)

print("Mask Shape  :", batch["attention_mask"].shape)

print("Labels      :", batch["labels"])