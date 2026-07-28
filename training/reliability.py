"""
Reliability Diagram
"""

import torch
import matplotlib.pyplot as plt


class ReliabilityDiagram:

    def __init__(self, n_bins=15):
        self.n_bins = n_bins

    def plot(
        self,
        probabilities,
        labels,
        save_path="outputs/reliability_diagram.png",
    ):

        confidences, predictions = torch.max(
            probabilities,
            dim=1,
        )

        accuracies = predictions.eq(labels)

        bin_boundaries = torch.linspace(
            0,
            1,
            self.n_bins + 1,
        )

        bin_centers = (
            bin_boundaries[:-1]
            + bin_boundaries[1:]
        ) / 2

        bin_acc = []
        bin_conf = []

        for i in range(self.n_bins):

            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]

            in_bin = (
                confidences > lower
            ) & (
                confidences <= upper
            )

            if in_bin.sum() > 0:

                accuracy = accuracies[in_bin].float().mean().item()

                confidence = confidences[in_bin].mean().item()

            else:

                accuracy = 0
                confidence = 0

            bin_acc.append(accuracy)
            bin_conf.append(confidence)

        plt.figure(figsize=(8, 8))

        # Perfect calibration
        plt.plot(
            [0, 1],
            [0, 1],
            "--",
            linewidth=2,
            label="Perfect Calibration",
        )

        # Accuracy bars
        plt.bar(
            bin_centers,
            bin_acc,
            width=1 / self.n_bins,
            alpha=0.6,
            edgecolor="black",
            label="Accuracy",
        )

        # Confidence line
        plt.plot(
            bin_centers,
            bin_conf,
            "o-",
            linewidth=2,
            markersize=5,
            label="Average Confidence",
        )

        plt.xlim(0, 1)
        plt.ylim(0, 1)

        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")

        plt.title("Reliability Diagram")

        plt.grid(alpha=0.3)

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            save_path,
            dpi=300,
        )

        plt.close()