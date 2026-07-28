"""
Evaluate Trained Audio Classifier
"""

import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from configs.config import *

from datasets.esc50_dataset import ESC50Dataset
from datasets.collate import Wav2Vec2Collator

from models.audio_classifier import AudioClassifier

from training.calibration import ExpectedCalibrationError

from training.reliability import ReliabilityDiagram
def main():

    dataset = ESC50Dataset()

    train_size = int(TRAIN_SPLIT * len(dataset))
    test_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(SEED)

    _, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=generator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=Wav2Vec2Collator(),
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    model = AudioClassifier().to(DEVICE)

    model.load_state_dict(
        torch.load(
            CHECKPOINT_DIR / "best_model.pth",
            map_location=DEVICE,
        )
    )

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    ece_metric = ExpectedCalibrationError(n_bins=15)

    with torch.no_grad():

        for batch in test_loader:

            input_values = batch["input_values"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(
                input_values,
                attention_mask,
            )

            probs = torch.softmax(logits, dim=1)

            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            all_probs.append(probs.cpu())

    
    # Metrics

    accuracy = accuracy_score(
        all_labels,
        all_preds,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0,
    )

    all_probs = torch.cat(all_probs, dim=0)
    labels_tensor = torch.tensor(all_labels)

    ece = ece_metric.compute(
        all_probs,
        labels_tensor,
    )
    diagram = ReliabilityDiagram()
    diagram.plot(
    all_probs,
    labels_tensor,
    )
    print("Reliability diagram saved.")
    print("=" * 60)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ECE      : {ece:.4f}")
    print("=" * 60)

    with open("outputs/metrics.txt", "w") as f:
        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1 Score : {f1:.4f}\n")
        f.write(f"ECE      : {ece:.4f}\n")

    
    
    # Classification Report

    report = classification_report(
        all_labels,
        all_preds,
        zero_division=0,
    )

    print("\nClassification Report\n")
    print(report)

    with open("outputs/classification_report.txt", "w") as f:
        f.write(report)

    # Confusion Matrix

    class_names = [
        "dog", "rooster", "pig", "cow", "frog",
        "cat", "hen", "insects", "sheep", "crow",
        "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds",
        "water_drops", "wind", "pouring_water", "toilet_flush", "thunderstorm",
        "crying_baby", "sneezing", "clapping", "breathing", "coughing",
        "footsteps", "laughing", "brushing_teeth", "snoring", "drinking_sipping",
        "door_wood_knock", "mouse_click", "keyboard_typing", "door_wood_creaks", "can_opening",
        "washing_machine", "vacuum_cleaner", "clock_alarm", "clock_tick", "glass_breaking",
        "helicopter", "chainsaw", "siren", "car_horn", "engine",
        "train", "church_bells", "airplane", "fireworks", "hand_saw"
    ]

    cm = confusion_matrix(
        all_labels,
        all_preds,
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(figsize=(18, 18))

    disp.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=90,
        colorbar=False,
    )

    plt.title("ESC-50 Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        "outputs/confusion_matrix.png",
        dpi=300,
    )
    plt.close()

    print("\n Metrics saved to outputs/metrics.txt")
    print(" Classification report saved to outputs/classification_report.txt")
    print(" Confusion matrix saved to outputs/confusion_matrix.png")


if __name__ == "__main__":
    main()