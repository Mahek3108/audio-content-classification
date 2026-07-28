import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from configs.config import *

from datasets.esc50_dataset import ESC50Dataset
from datasets.collate import Wav2Vec2Collator

from models.audio_classifier import AudioClassifier

from training.engine import train_one_epoch
from training.engine import validate


class Trainer:

    def __init__(self):

        dataset = ESC50Dataset()

        train_size = int(TRAIN_SPLIT * len(dataset))
        val_size = len(dataset) - train_size

        generator = torch.Generator().manual_seed(42)

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=generator,
        )

        collator = Wav2Vec2Collator()

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collator,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collator,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

        self.model = AudioClassifier().to(DEVICE)

        self.optimizer = optim.AdamW(
        [
            {
                "params": self.model.encoder.encoder.parameters(),
                "lr": 1e-5,
            },
            {
                "params": self.model.classifier.parameters(),
                "lr": 1e-4,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )

        self.criterion = nn.CrossEntropyLoss()

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=DEVICE.type == "cuda"
        )

        self.best_acc = 0

    def train(self):
        train_losses = []
        val_losses = []

        train_accs = []
        val_accs = []

        for epoch in range(EPOCHS):

            train_loss, train_acc = train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.criterion,
                self.scaler,
                DEVICE,
            )

            val_loss, val_acc = validate(
                self.model,
                self.val_loader,
                self.criterion,
                DEVICE,
            )

            self.scheduler.step(val_acc)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            train_accs.append(train_acc)
            val_accs.append(val_acc)

            print()

            print("=" * 60)

            print(f"Epoch {epoch+1}/{EPOCHS}")

            print(f"Train Loss : {train_loss:.4f}")

            print(f"Train Acc  : {train_acc*100:.2f}%")

            print(f"Val Loss   : {val_loss:.4f}")

            print(f"Val Acc    : {val_acc*100:.2f}%")

            if val_acc > self.best_acc:

                self.best_acc = val_acc

                torch.save(
                    self.model.state_dict(),
                    CHECKPOINT_DIR / "best_model.pth",
                )

                print("✅ Best model saved.")
        torch.save(
            {
                "train_loss": train_losses,
                "val_loss": val_losses,
                "train_acc": train_accs,
                "val_acc": val_accs,
            },
            OUTPUT_DIR / "training_history.pt",
        )

        print("✅ Training history saved.")