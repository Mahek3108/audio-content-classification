from torch.utils.data import DataLoader

from datasets.esc50_dataset import ESC50Dataset
from datasets.collate import Wav2Vec2Collator
from models.audio_classifier import AudioClassifier

dataset = ESC50Dataset()

loader = DataLoader(
    dataset,
    batch_size=4,
    collate_fn=Wav2Vec2Collator(),
)

batch = next(iter(loader))

model = AudioClassifier()

logits = model(
    batch["input_values"],
    batch["attention_mask"]
)

print("Logits shape:", logits.shape)