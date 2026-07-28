"""
Monte Carlo Dropout utilities for uncertainty estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def enable_mc_dropout(model):
    """
    Enable dropout layers during inference.
    Keeps the rest of the model in evaluation mode.
    """

    model.eval()

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_predict(
    model,
    input_values,
    attention_mask=None,
    num_samples=30,
):
    """
    Perform Monte Carlo Dropout inference.

    Returns:
        mean_probs
        all_probs
    """

    enable_mc_dropout(model)

    predictions = []

    for _ in range(num_samples):

        logits = model(
            input_values=input_values,
            attention_mask=attention_mask,
        )

        probs = F.softmax(logits, dim=1)

        predictions.append(probs)

    all_probs = torch.stack(predictions)

    mean_probs = all_probs.mean(dim=0)

    return mean_probs, all_probs


def predictive_entropy(mean_probs):
    """
    Predictive entropy.

    Higher entropy = more uncertainty.
    """

    entropy = -torch.sum(
        mean_probs * torch.log(mean_probs + 1e-12),
        dim=1,
    )

    return entropy


def predictive_variance(all_probs):
    """
    Variance across MC samples.
    """

    variance = all_probs.var(dim=0)

    return variance.mean(dim=1)


def confidence_score(mean_probs):
    """
    Maximum predicted probability.
    """

    confidence, predicted_class = torch.max(
        mean_probs,
        dim=1,
    )

    return confidence, predicted_class

def top_k_predictions(mean_probs, k=5):
    """
    Returns top-k class indices and probabilities.
    """

    probs, indices = torch.topk(
        mean_probs,
        k=k,
        dim=1,
    )

    return probs, indices


import pandas as pd

df = pd.read_csv("data/ESC-50/meta/esc50.csv")
print(df.columns)
print(df.head())