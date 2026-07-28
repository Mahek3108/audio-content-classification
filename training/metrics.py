import torch


def calculate_accuracy(logits, labels):
    """
    Returns:
        correct predictions,
        total samples
    """
    preds = torch.argmax(logits, dim=1)

    correct = (preds == labels).sum().item()

    total = labels.size(0)

    return correct, total