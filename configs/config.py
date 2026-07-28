"""
Configuration file for Audio Content Classification
using Wav2Vec2 Self-Supervised Embeddings.
"""

from pathlib import Path
import torch


# Project Paths


# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Dataset paths
DATA_DIR = ROOT_DIR / "data" / "ESC-50"
AUDIO_DIR = DATA_DIR / "audio"
META_FILE = DATA_DIR / "meta" / "esc50.csv"

# Output paths
OUTPUT_DIR = ROOT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Model Configuration

MODEL_NAME = "facebook/wav2vec2-base"

# Audio Configuration

SAMPLE_RATE = 16000
MAX_DURATION = 5  # seconds
MAX_AUDIO_LENGTH = SAMPLE_RATE * MAX_DURATION

# Training Configuration

BATCH_SIZE = 8
EPOCHS = 20

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

TRAIN_SPLIT = 0.8

NUM_CLASSES = 50

NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

FREEZE_ENCODER = False
# Classifier Configuration

HIDDEN_DIM = 512
DROPOUT = 0.3

# Device Configuration

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Reproducibility

SEED = 42