import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

LABELS = {"Claude": 0, "DeepSeek": 1, "GPT": 2}
MAX_VOCAB = 15000

# Preserve case, punctuation, and line breaks as potential style features.
TOKEN_PATTERN = re.compile(r"\n|[A-Za-z0-9_]+|[^\s]")


def tokenize(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return TOKEN_PATTERN.findall(text)


def read_split(name):
    path = INPUT_DIR / f"{name}.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        if row["LLM_name"] not in LABELS:
            raise ValueError(f"Unknown label: {row['LLM_name']}")
        if not row["LLM_output"].strip():
            raise ValueError(f"Empty answer: {row['response_id']}")

    return rows


def main():
    datasets = {
        name: read_split(name)
        for name in ("train", "validation", "test")
    }

    # Build vocabulary using training responses only.
    counts = Counter()
    training_lengths = []

    for row in datasets["train"]:
        tokens = tokenize(row["LLM_output"])
        counts.update(tokens)
        training_lengths.append(len(tokens))

    # 0 = padding; 1 = unknown token.
    vocab = {"<PAD>": 0, "<UNK>": 1}
    ranked_tokens = sorted(
        counts, key=lambda token: (-counts[token], token)
    )
    for token in ranked_tokens[:MAX_VOCAB - 2]:
        vocab[token] = len(vocab)

    # Choose length using training data only.
    # Use the 95th percentile, capped at 512 tokens.
    lengths = sorted(training_lengths)
    percentile_index = math.ceil(0.95 * len(lengths)) - 1
    max_length = min(512, max(16, lengths[percentile_index]))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, rows in datasets.items():
        records = []
        truncated = 0
        unknown_count = 0
        retained_count = 0

        for row in rows:
            tokens = tokenize(row["LLM_output"])
            if len(tokens) > max_length:
                truncated += 1

            retained = tokens[:max_length]
            token_ids = [vocab.get(token, 1) for token in retained]
            length = len(token_ids)

            unknown_count += token_ids.count(1)
            retained_count += length
            token_ids += [0] * (max_length - length)

            records.append({
                "response_id": row["response_id"],
                "prompt_id": row["prompt_id"],
                "label": LABELS[row["LLM_name"]],
                "length": length,
                "input_ids": token_ids,
            })

        path = OUTPUT_DIR / f"{name}_output.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(records, file, ensure_ascii=False)

        unknown_rate = unknown_count / max(1, retained_count)
        print(
            f"{name}: {len(records)} samples, "
            f"truncated={truncated}, "
            f"unknown tokens={unknown_rate:.2%}"
        )

    config = {
        "input_mode": "output_only",
        "vocab_size": len(vocab),
        "max_length": max_length,
        "label_mapping": LABELS,
        "padding_id": 0,
        "unknown_id": 1,
        "token_pattern": TOKEN_PATTERN.pattern,
        "length_selection": "training 95th percentile, capped at 512",
    }

    for filename, content in (
        ("vocab.json", vocab),
        ("config.json", config),
    ):
        with (OUTPUT_DIR / filename).open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(content, file, ensure_ascii=False, indent=2)

    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Sequence length: {max_length}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()