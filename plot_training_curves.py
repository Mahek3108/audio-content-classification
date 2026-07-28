import matplotlib.pyplot as plt

# Training history
epochs = list(range(1, 21))

train_loss = [
    3.8321, 3.3838, 2.8518, 2.3761, 1.9931,
    1.6432, 1.3183, 1.0812, 0.8744, 0.6901,
    0.5416, 0.4750, 0.3787, 0.2985, 0.2573,
    0.1947, 0.1865, 0.1632, 0.1581, 0.0969
]

val_loss = [
    3.6333, 3.0371, 2.4812, 2.1451, 1.8918,
    1.6137, 1.4849, 1.4727, 1.2100, 1.2629,
    1.3360, 1.1920, 1.1886, 1.2225, 1.2321,
    1.1548, 1.2298, 1.2205, 1.6500, 1.2332
]

train_acc = [
    4.38, 11.38, 17.56, 29.06, 40.00,
    49.81, 59.00, 66.38, 73.88, 78.88,
    84.31, 86.44, 89.62, 91.50, 92.75,
    94.88, 94.56, 96.00, 95.81, 97.25
]

val_acc = [
    10.50, 20.00, 27.00, 35.00, 48.50,
    54.00, 56.75, 56.50, 65.75, 65.50,
    63.50, 71.00, 70.25, 71.50, 72.75,
    74.75, 72.00, 74.00, 69.25, 75.25
]

# Accuracy Curve
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_acc, marker="o", label="Train Accuracy")
plt.plot(epochs, val_acc, marker="s", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training vs Validation Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("outputs/accuracy_curve.png", dpi=300)
plt.close()

# Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, marker="o", label="Train Loss")
plt.plot(epochs, val_loss, marker="s", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("outputs/loss_curve.png", dpi=300)
plt.close()

print("✅ Training curves saved to outputs/")