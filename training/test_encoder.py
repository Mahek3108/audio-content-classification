from torch.utils.data import DataLoader

from datasets.esc50_dataset import ESC50Dataset
from datasets.collate import Wav2Vec2Collator
from models.wav2vec2_encoder import Wav2Vec2Encoder

dataset = ESC50Dataset()

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=Wav2Vec2Collator()
)

batch = next(iter(loader))

model = Wav2Vec2Encoder()

embeddings = model(
    batch["input_values"],
    batch["attention_mask"]
)

print("Embedding shape:", embeddings.shape)