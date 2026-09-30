import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import train as training

import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "splits"
PROCESSED = ROOT / "data" / "processed" / "rq3"
RESULTS = ROOT / "results" / "rq3"

SOURCE_CATEGORIES = {"Coding", "Math", "Factual"}
TARGET_CATEGORY = "Writing"
LABELS = {"Claude": 0, "DeepSeek": 1, "GPT": 2}
CLASS_NAMES = ["Claude", "DeepSeek", "GPT"]

MAX_VOCAB = 15000
PATTERN = re.compile(r"\n|[A-Za-z0-9_]+|[^\s]")


def tokenize(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return PATTERN.findall(text)


def read_split(split):
    with (SOURCE / f"{split}.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        return list(csv.DictReader(file))


def write_json(path, content):
    with path.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_data():
    original = {
        split: read_split(split)
        for split in ("train", "validation", "test")
    }

    available = {
        row["category"]
        for rows in original.values()
        for row in rows
    }
    required = SOURCE_CATEGORIES | {TARGET_CATEGORY}
    if not required.issubset(available):
        raise ValueError(
            f"Missing categories: {required - available}. "
            f"Available categories: {sorted(available)}"
        )

    raw = {
        split: [
            row for row in original[split]
            if row["category"] in SOURCE_CATEGORIES
        ]
        for split in ("train", "validation", "test")
    }
    raw["cross_domain"] = [
        row
        for rows in original.values()
        for row in rows
        if row["category"] == TARGET_CATEGORY
    ]

    # Validate complete groups and disjoint prompt IDs.
    prompt_sets = {}
    for split, rows in raw.items():
        if not rows:
            raise ValueError(f"{split}: empty dataset")

        groups = {}
        for row in rows:
            if not row["LLM_output"].strip():
                raise ValueError(f"Empty answer: {row['response_id']}")
            groups.setdefault(row["prompt_id"], []).append(row)

        for prompt_id, group in groups.items():
            counts = Counter(row["LLM_name"] for row in group)
            if counts != Counter({name: 1 for name in LABELS}):
                raise ValueError(f"Incomplete model group: {prompt_id}")
            if len({row["LLM_Input"] for row in group}) != 1:
                raise ValueError(f"Inconsistent input: {prompt_id}")

        prompt_sets[split] = set(groups)

    names = list(prompt_sets)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            if prompt_sets[first] & prompt_sets[second]:
                raise ValueError(f"Prompt overlap: {first}, {second}")

    # Fit vocabulary and sequence length on source training data only.
    counts = Counter()
    lengths = []

    for row in raw["train"]:
        tokens = tokenize(row["LLM_output"])
        counts.update(tokens)
        lengths.append(len(tokens))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    ranked = sorted(counts, key=lambda token: (-counts[token], token))
    for token in ranked[:MAX_VOCAB - 2]:
        vocab[token] = len(vocab)

    ordered = sorted(lengths)
    index = math.ceil(0.95 * len(ordered)) - 1
    max_length = min(512, max(16, ordered[index]))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    datasets = {}
    manifest = []
    statistics = []

    for split, rows in raw.items():
        rows.sort(key=lambda row: (
            row["prompt_id"], row["LLM_name"]
        ))
        encoded = []
        unknown = 0
        retained_total = 0
        truncated = 0

        for row in rows:
            tokens = tokenize(row["LLM_output"])
            truncated += int(len(tokens) > max_length)
            ids = [vocab.get(token, 1) for token in tokens[:max_length]]
            length = len(ids)

            if length == 0:
                raise ValueError(f"Empty sequence: {row['response_id']}")

            unknown += ids.count(1)
            retained_total += length
            ids += [0] * (max_length - length)

            encoded.append({
                "response_id": row["response_id"],
                "prompt_id": row["prompt_id"],
                "label": LABELS[row["LLM_name"]],
                "length": length,
                "input_ids": ids,
            })

        for prompt_id in sorted(prompt_sets[split]):
            category = next(
                row["category"] for row in rows
                if row["prompt_id"] == prompt_id
            )
            manifest.append({
                "prompt_id": prompt_id,
                "category": category,
                "rq3_split": split,
            })

        statistics.append({
            "split": split,
            "prompts": len(prompt_sets[split]),
            "responses": len(rows),
            "truncated_responses": truncated,
            "unknown_token_rate": unknown / max(1, retained_total),
        })
        write_json(PROCESSED / f"{split}_output.json", encoded)
        datasets[split] = training.TextDataset(encoded)

        print(
            f"{split}: {len(prompt_sets[split])} prompts, "
            f"{len(rows)} responses"
        )

    config = {
        "input_mode": "output_only",
        "vocab_size": len(vocab),
        "max_length": max_length,
        "label_mapping": LABELS,
        "source_categories": sorted(SOURCE_CATEGORIES),
        "target_category": TARGET_CATEGORY,
        "padding_id": 0,
        "unknown_id": 1,
        "token_pattern": PATTERN.pattern,
        "length_selection": "source training 95th percentile, cap 512",
        "normalization": "actual CRLF/CR to LF; literal escapes retained",
    }

    write_json(PROCESSED / "vocab.json", vocab)
    write_json(PROCESSED / "config.json", config)
    write_csv(RESULTS / "split_manifest.csv", manifest)
    write_csv(RESULTS / "preprocessing_statistics.csv", statistics)

    return config, datasets


def save_cross_domain_results(model_name, model, dataset):
    loader = training.DataLoader(
        dataset,
        batch_size=training.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    scores = training.run_epoch(
        model, loader, training.nn.CrossEntropyLoss()
    )

    report = classification_report(
        scores["actual"], scores["predicted"],
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    (training.RESULT_DIR / "cross_domain_report.txt").write_text(
        report, encoding="utf-8"
    )

    report_dict = classification_report(
        scores["actual"], scores["predicted"],
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )
    write_json(
        training.RESULT_DIR / "cross_domain_metrics.json",
        report_dict,
    )

    matrix = confusion_matrix(
        scores["actual"], scores["predicted"], labels=[0, 1, 2]
    )
    fig, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        matrix, display_labels=CLASS_NAMES
    ).plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title(f"{model_name}: Unseen Writing")
    fig.tight_layout()
    fig.savefig(
        training.RESULT_DIR / "cross_domain_confusion_matrix.png",
        dpi=200,
    )
    plt.close(fig)

    predictions = [
        {
            "response_id": row["response_id"],
            "prompt_id": row["prompt_id"],
            "actual": CLASS_NAMES[actual],
            "predicted": CLASS_NAMES[predicted],
        }
        for row, actual, predicted in zip(
            dataset.records, scores["actual"], scores["predicted"]
        )
    ]
    write_csv(
        training.RESULT_DIR / "cross_domain_predictions.csv",
        predictions,
    )

    print(f"\n{model_name}: unseen Writing\n{report}")
    return scores


def main():
    config, datasets = prepare_data()
    summaries = []

    for model_name in ("CNN", "LSTM"):
        training.MODEL_DIR = (
            ROOT / "models" / "rq3" / model_name.lower()
        )
        training.RESULT_DIR = RESULTS / model_name.lower()
        training.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        training.RESULT_DIR.mkdir(parents=True, exist_ok=True)

        # The reused function selects a checkpoint using source validation
        # loss and evaluates it on the source-domain test set.
        source_datasets = {
            split: datasets[split]
            for split in ("train", "validation", "test")
        }
        in_domain = training.train_model(
            model_name, config, source_datasets, CLASS_NAMES
        )

        checkpoint = (
            training.MODEL_DIR / f"{model_name.lower()}_output_only.pt"
        )
        model_class = (
            training.CNN if model_name == "CNN" else training.LSTM
        )
        model = model_class(
            config["vocab_size"], len(CLASS_NAMES)
        ).to(training.DEVICE)

        model.load_state_dict(training.torch.load(
            checkpoint,
            map_location=training.DEVICE,
            weights_only=True,
        ))

        # Same checkpoint: no adaptation to Writing.
        cross_domain = save_cross_domain_results(
            model_name, model, datasets["cross_domain"]
        )

        summaries.append({
            "model": model_name,
            "best_epoch": in_domain["best_epoch"],
            "in_domain_accuracy": in_domain["test_accuracy"],
            "in_domain_macro_f1": in_domain["test_macro_f1"],
            "cross_domain_accuracy": cross_domain["accuracy"],
            "cross_domain_macro_f1": cross_domain["macro_f1"],
            "accuracy_change_percentage_points": 100 * (
                cross_domain["accuracy"] - in_domain["test_accuracy"]
            ),
            "macro_f1_change": (
                cross_domain["macro_f1"] - in_domain["test_macro_f1"]
            ),
        })

    write_csv(RESULTS / "rq3_comparison.csv", summaries)
    write_json(RESULTS / "experiment_config.json", {
        "preprocessing": config,
        "seed": training.SEED,
        "batch_size": training.BATCH_SIZE,
        "maximum_epochs": training.EPOCHS,
        "patience": training.PATIENCE,
        "learning_rate": training.LEARNING_RATE,
        "checkpoint_selection": "minimum source validation loss",
        "torch_version": training.torch.__version__,
    })

    print("\nRQ3 completed:")
    for row in summaries:
        print(
            f"{row['model']}: "
            f"in-domain={row['in_domain_accuracy']:.2%}, "
            f"Writing={row['cross_domain_accuracy']:.2%}, "
            f"change={row['accuracy_change_percentage_points']:+.2f} pp"
        )
    print(f"\nResults saved to: {RESULTS}")


if __name__ == "__main__":
    main()