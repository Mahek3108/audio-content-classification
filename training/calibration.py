"""
Calibration Metrics
"""

import torch


class ExpectedCalibrationError:
    """
    Computes Expected Calibration Error (ECE).
    """

    def __init__(self, n_bins=15):
        self.n_bins = n_bins

    def compute(self, probabilities, labels):

        confidences, predictions = torch.max(
            probabilities,
            dim=1,
        )

        accuracies = predictions.eq(labels)

        ece = torch.zeros(
            1,
            device=probabilities.device,
        )

        bin_boundaries = torch.linspace(
            0,
            1,
            self.n_bins + 1,
            device=probabilities.device,
        )

        for i in range(self.n_bins):

            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]

            in_bin = (
                confidences > lower
            ) & (
                confidences <= upper
            )

            prop = in_bin.float().mean()

            if prop.item() > 0:

                accuracy = accuracies[in_bin].float().mean()

                confidence = confidences[in_bin].mean()

                ece += torch.abs(
                    confidence - accuracy
                ) * prop

        return ece.item()