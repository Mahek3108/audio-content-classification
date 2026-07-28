from datasets.esc50_dataset import ESC50Dataset

dataset = ESC50Dataset()

print("Dataset size:", len(dataset))

audio, label = dataset[0]

print("Audio shape:", audio.shape)
print("Label:", label)