import torch

from training.calibration import ExpectedCalibrationError

ece = ExpectedCalibrationError()

probs = torch.tensor([
    [0.9, 0.1],
    [0.8, 0.2],
    [0.6, 0.4],
    [0.55, 0.45],
])

labels = torch.tensor([0, 0, 1, 1])

score = ece.compute(probs, labels)

print(score)