import torch
from tqdm import tqdm

from training.metrics import calculate_accuracy


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
):

    model.train()

    running_loss = 0

    correct = 0
    total = 0

    for batch in tqdm(loader):

        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):

            logits = model(
                input_values,
                attention_mask,
            )

            loss = criterion(
                logits,
                labels,
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        c, t = calculate_accuracy(
            logits,
            labels,
        )

        correct += c
        total += t

    epoch_loss = running_loss / len(loader)

    epoch_acc = correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0

    correct = 0
    total = 0

    for batch in loader:

        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(
            input_values,
            attention_mask,
        )

        loss = criterion(
            logits,
            labels,
        )

        running_loss += loss.item()

        c, t = calculate_accuracy(
            logits,
            labels,
        )

        correct += c
        total += t

    epoch_loss = running_loss / len(loader)

    epoch_acc = correct / total

    return epoch_loss, epoch_acc