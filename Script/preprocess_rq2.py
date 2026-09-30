import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / "data" / "splits"
PROCESSED_DIR = ROOT / "data" / "processed"

LABELS = {"Claude": 0, "DeepSeek": 1, "GPT": 2}
MAX_VOCAB = 15000
PATTERN = re.compile(r"\n|[A-Za-z0-9_]+|[^\s]")


def tokenize(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return PATTERN.findall(text)


def save_json(path, content):
    with path.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)


def read_split(name):
    with (SPLIT_DIR / f"{name}.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        return list(csv.DictReader(file))


def choose_length(lengths, cap):
    ordered = sorted(lengths)
    index = math.ceil(0.95 * len(ordered)) - 1
    return min(cap, max(16, ordered[index]))


def validate(datasets):
    prompt_sets = {}

    for split, rows in datasets.items():
        if not rows:
            raise ValueError(f"{split}: empty dataset")

        groups = {}
        for row in rows:
            if row["LLM_name"] not in LABELS:
                raise ValueError("Unknown LLM label")
            if not row["LLM_Input"].strip():
                raise ValueError(f"{split}: empty input")
            if not row["LLM_output"].strip():
                raise ValueError(f"{split}: empty output")
            groups.setdefault(row["prompt_id"], []).append(row)

        for prompt_id, group in groups.items():
            counts = Counter(row["LLM_name"] for row in group)
            if counts != Counter({name: 1 for name in LABELS}):
                raise ValueError(
                    f"{split}, {prompt_id}: incomplete model group"
                )
            if len({row["LLM_Input"] for row in group}) != 1:
                raise ValueError(
                    f"{split}, {prompt_id}: inconsistent inputs"
                )

        prompt_sets[split] = set(groups)

    for first, second in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if prompt_sets[first] & prompt_sets[second]:
            raise ValueError(f"Prompt overlap: {first}, {second}")


def prepare(mode, datasets):
    destination = PROCESSED_DIR / mode
    destination.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    input_lengths = []
    output_lengths = []

    for row in datasets["train"]:
        inputs = tokenize(row["LLM_Input"])
        outputs = tokenize(row["LLM_output"])

        input_lengths.append(len(inputs))
        counts.update(inputs)

        if mode == "input_output":
            output_lengths.append(len(outputs))
            counts.update(outputs)

    input_limit = choose_length(input_lengths, cap=256)
    output_limit = (
        choose_length(output_lengths, cap=512)
        if mode == "input_output" else 0
    )

    # Fixed special tokens; no model names are added to the text.
    vocab = {"<PAD>": 0, "<UNK>": 1, "<SEP>": 2}
    ranked = sorted(counts, key=lambda token: (-counts[token], token))

    for token in ranked:
        if token not in vocab:
            vocab[token] = len(vocab)
        if len(vocab) >= MAX_VOCAB:
            break

    max_length = input_limit
    if mode == "input_output":
        max_length += 1 + output_limit

    statistics = {}

    for split, rows in datasets.items():
        records = []
        input_truncated = 0
        output_truncated = 0
        unknown_count = 0
        token_count = 0

        for row in rows:
            inputs = tokenize(row["LLM_Input"])
            outputs = tokenize(row["LLM_output"])

            input_truncated += int(len(inputs) > input_limit)
            ids = [
                vocab.get(token, 1)
                for token in inputs[:input_limit]
            ]

            if mode == "input_output":
                output_truncated += int(len(outputs) > output_limit)
                ids += [vocab["<SEP>"]]
                ids += [
                    vocab.get(token, 1)
                    for token in outputs[:output_limit]
                ]

            # Padding only goes at the end, so LSTM lengths remain valid.
            length = len(ids)
            unknown_count += ids.count(1)
            token_count += length
            ids += [0] * (max_length - length)

            records.append({
                "response_id": row["response_id"],
                "prompt_id": row["prompt_id"],
                "label": LABELS[row["LLM_name"]],
                "length": length,
                "input_ids": ids,
            })

        # Names match the file-reading convention in train.py.
        save_json(destination / f"{split}_output.json", records)

        statistics[split] = {
            "samples": len(records),
            "input_truncated": input_truncated,
            "output_truncated": output_truncated,
            "unknown_token_rate": unknown_count / max(1, token_count),
        }
        print(mode, split, statistics[split])

    config = {
        "input_mode": mode,
        "vocab_size": len(vocab),
        "max_length": max_length,
        "input_limit": input_limit,
        "output_limit": output_limit,
        "label_mapping": LABELS,
        "padding_id": 0,
        "unknown_id": 1,
        "separator_id": 2,
        "token_pattern": PATTERN.pattern,
        "length_selection": "training 95th percentile with caps",
    }

    save_json(destination / "vocab.json", vocab)
    save_json(destination / "config.json", config)
    save_json(destination / "preprocessing_statistics.json", statistics)

    print(f"Saved: {destination}\n")


def main():
    datasets = {
        split: read_split(split)
        for split in ("train", "validation", "test")
    }
    validate(datasets)

    for mode in ("input_only", "input_output"):
        prepare(mode, datasets)


if __name__ == "__main__":
    main()