from configs.config import *

print("Project Root:", ROOT_DIR)
print("Audio Directory:", AUDIO_DIR)
print("Metadata File:", META_FILE)
print("Device:", DEVICE)
print("Model:", MODEL_NAME)

import torch

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))