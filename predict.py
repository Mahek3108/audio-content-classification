"""
Prediction with Monte Carlo Dropout Uncertainty
"""

import argparse

import librosa
import pandas as pd
import torch
from transformers import AutoProcessor

from configs.config import (
    MODEL_NAME,
    SAMPLE_RATE,
    MAX_AUDIO_LENGTH,
    META_FILE,
    CHECKPOINT_DIR,
    DEVICE,
)

from models.audio_classifier import AudioClassifier

from utils.uncertainty import (
    mc_dropout_predict,
    predictive_entropy,
    predictive_variance,
    confidence_score,
    top_k_predictions,
)


def load_audio(audio_path):

    waveform, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    if len(waveform) > MAX_AUDIO_LENGTH:
        waveform = waveform[:MAX_AUDIO_LENGTH]
    else:
        waveform = librosa.util.fix_length(
            waveform,
            size=MAX_AUDIO_LENGTH,
        )

    return waveform


def load_label_map():

    df = pd.read_csv(META_FILE)

    mapping = (
        df[["target", "category"]]
        .drop_duplicates()
        .sort_values("target")
    )

    return dict(
        zip(mapping.target, mapping.category)
    )


def load_model():

    model = AudioClassifier().to(DEVICE)

    model.load_state_dict(
        torch.load(
            CHECKPOINT_DIR / "best_model.pth",
            map_location=DEVICE,
        )
    )

    model.eval()

    return model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio",
        required=True,
        help="Path to audio file",
    )

    parser.add_argument(
        "--samples",
        default=30,
        type=int,
        help="Number of MC Dropout samples",
    )

    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    waveform = load_audio(args.audio)

    inputs = processor(
        waveform,
        sampling_rate=SAMPLE_RATE,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_values = inputs["input_values"].to(DEVICE)

    attention_mask = inputs.get(
        "attention_mask",
        torch.ones_like(
            input_values,
            dtype=torch.long,
        ),
    ).to(DEVICE)

    model = load_model()

    mean_probs, all_probs = mc_dropout_predict(
        model=model,
        input_values=input_values,
        attention_mask=attention_mask,
        num_samples=args.samples,
    )

    entropy = predictive_entropy(mean_probs)

    variance = predictive_variance(all_probs)

    confidence, prediction = confidence_score(
        mean_probs
    )

    top_probs, top_indices = top_k_predictions(
        mean_probs,
        k=5,
    )

    label_map = load_label_map()

    print()

    print("=" * 60)

    print("Audio Classification with Uncertainty")

    print("=" * 60)

    print()

    print(f"Audio File : {args.audio}")

    print()

    print(
        f"Prediction : {label_map[prediction.item()]}"
    )

    print(
        f"Confidence : {confidence.item()*100:.2f}%"
    )

    print(
        f"Entropy    : {entropy.item():.4f}"
    )

    print(
        f"Variance   : {variance.item():.6f}"
    )

    print()

    print("Top-5 Predictions")

    print("-" * 60)

    for prob, idx in zip(
        top_probs[0],
        top_indices[0],
    ):

        print(
            f"{label_map[idx.item()]:25s}"
            f"{prob.item()*100:.2f}%"
        )


if __name__ == "__main__":

    main()