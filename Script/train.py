import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results" / "output_only"

SEED = 42
BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def set_seed():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TextDataset(Dataset):
    def __init__(self, records):
        self.records = records
        self.inputs = torch.tensor(
            [row["input_ids"] for row in records],
            dtype=torch.long,
        )
        self.labels = torch.tensor(
            [row["label"] for row in records],
            dtype=torch.long,
        )
        self.lengths = torch.tensor(
            [row["length"] for row in records],
            dtype=torch.long,
        )
        if (self.lengths <= 0).any():
            raise ValueError("Found an empty token sequence.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return (
            self.inputs[index],
            self.lengths[index],
            self.labels[index],
        )


class CNN(nn.Module):
    def __init__(self, vocab_size, classes):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, 128, padding_idx=0
        )
        self.convolutions = nn.ModuleList([
            nn.Conv1d(128, 64, kernel_size=k, padding=k // 2)
            for k in (3, 5, 7)
        ])
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(64 * 3, classes)

    def forward(self, inputs, lengths):
        embedded = self.embedding(inputs).transpose(1, 2)
        positions = torch.arange(
            inputs.size(1), device=inputs.device
        )[None, :]

        pooled = []
        for convolution in self.convolutions:
            features = torch.relu(convolution(embedded))
            # Only pool outputs centered on real tokens.
            valid = positions < lengths.to(inputs.device)[:, None]
            features = features.masked_fill(
                ~valid[:, None, :], float("-inf")
            )
            pooled.append(features.amax(dim=2))

        combined = torch.cat(pooled, dim=1)
        return self.classifier(self.dropout(combined))


class LSTM(nn.Module):
    def __init__(self, vocab_size, classes):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, 128, padding_idx=0
        )
        self.lstm = nn.LSTM(
            128, 64, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(128, classes)

    def forward(self, inputs, lengths):
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        combined = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(self.dropout(combined))


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    actual, predicted = [], []

    with torch.set_grad_enabled(training):
        for inputs, lengths, labels in loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            if training:
                optimizer.zero_grad()

            logits = model(inputs, lengths)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            actual.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(
            actual, predicted,
            labels=list(range(3)),
            average="macro",
            zero_division=0,
        ),
        "actual": actual,
        "predicted": predicted,
    }


def save_curves(name, history):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [row["epoch"] for row in history]

    for axis, metric in zip(axes, ("loss", "accuracy")):
        axis.plot(
            epochs, [row[f"train_{metric}"] for row in history],
            label="Training",
        )
        axis.plot(
            epochs, [row[f"val_{metric}"] for row in history],
            label="Validation",
        )
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric.title())
        axis.legend()

    fig.suptitle(f"{name}: Output-only")
    fig.tight_layout()
    fig.savefig(
        RESULT_DIR / f"{name}_training_curves.png", dpi=200
    )
    plt.close(fig)


def train_model(name, config, datasets, class_names):
    set_seed()

    train_loader = DataLoader(
        datasets["train"], batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        datasets["validation"], batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0,
    )

    model_class = CNN if name == "CNN" else LSTM
    model = model_class(
        config["vocab_size"], len(class_names)
    ).to(DEVICE)

    # CrossEntropyLoss expects raw logits and integer labels.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE,
        weight_decay=0.0001,
    )

    checkpoint = MODEL_DIR / f"{name.lower()}_output_only.pt"
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train = run_epoch(
            model, train_loader, criterion, optimizer
        )
        val = run_epoch(model, val_loader, criterion)

        history.append({
            "epoch": epoch,
            "train_loss": train["loss"],
            "val_loss": val["loss"],
            "train_accuracy": train["accuracy"],
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
        })

        print(
            f"{name} Epoch {epoch:02d} | "
            f"train loss={train['loss']:.4f} "
            f"acc={train['accuracy']:.3f} | "
            f"val loss={val['loss']:.4f} "
            f"acc={val['accuracy']:.3f} "
            f"F1={val['macro_f1']:.3f}"
        )

        if val["loss"] < best_loss:
            best_loss = val["loss"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale_epochs += 1

        if stale_epochs >= PATIENCE:
            print(f"{name}: early stopping.")
            break

    with (RESULT_DIR / f"{name}_history.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(history[0])
        )
        writer.writeheader()
        writer.writerows(history)

    save_curves(name, history)

    model.load_state_dict(torch.load(
        checkpoint, map_location=DEVICE, weights_only=True
    ))

    # Evaluate the selected checkpoint on the test set.
    test_loader = DataLoader(
        datasets["test"], batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0,
    )
    test = run_epoch(model, test_loader, criterion)
    label_ids = list(range(len(class_names)))

    report = classification_report(
        test["actual"], test["predicted"],
        labels=label_ids, target_names=class_names,
        zero_division=0,
    )
    print(f"\n{name} test results:\n{report}")

    (RESULT_DIR / f"{name}_classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    matrix = confusion_matrix(
        test["actual"], test["predicted"], labels=label_ids
    )
    fig, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        matrix, display_labels=class_names
    ).plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title(f"{name}: Output-only Test")
    fig.tight_layout()
    fig.savefig(
        RESULT_DIR / f"{name}_confusion_matrix.png", dpi=200
    )
    plt.close(fig)

    with (RESULT_DIR / f"{name}_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "response_id", "prompt_id", "actual", "predicted"
        ])
        for row, actual, predicted in zip(
            datasets["test"].records,
            test["actual"], test["predicted"],
        ):
            writer.writerow([
                row["response_id"], row["prompt_id"],
                class_names[actual], class_names[predicted],
            ])

    return {
        "model": name,
        "input_mode": "output_only",
        "best_epoch": best_epoch,
        "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
        "test_loss": test["loss"],
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    config = read_json(DATA_DIR / "config.json")
    class_names = [
        name for name, label in sorted(
            config["label_mapping"].items(),
            key=lambda item: item[1],
        )
    ]
    datasets = {
        split: TextDataset(read_json(
            DATA_DIR / f"{split}_output.json"
        ))
        for split in ("train", "validation", "test")
    }

    print(f"Device: {DEVICE}")
    print(f"Classes: {class_names}")

    summaries = [
        train_model(name, config, datasets, class_names)
        for name in ("CNN", "LSTM")
    ]

    with (RESULT_DIR / "summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(summaries[0])
        )
        writer.writeheader()
        writer.writerows(summaries)

    settings = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "maximum_epochs": EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.0001,
        "embedding_dimension": 128,
        "dropout": 0.5,
        "checkpoint_selection": "minimum validation loss",
        "preprocessing": config,
        "torch_version": torch.__version__,
    }
    (RESULT_DIR / "experiment_config.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )

    print("\nCompleted:")
    for row in summaries:
        print(
            f"{row['model']}: "
            f"accuracy={row['test_accuracy']:.3f}, "
            f"macro-F1={row['test_macro_f1']:.3f}"
        )
    print(f"Results: {RESULT_DIR}")


if __name__ == "__main__":
    main()