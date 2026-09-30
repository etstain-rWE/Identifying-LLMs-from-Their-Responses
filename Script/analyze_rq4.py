import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "splits"
DEST = ROOT / "results" / "rq4"
MODELS = ["Claude", "DeepSeek", "GPT"]

WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
FENCE = re.compile(
    r"(?ms)^\s*```[^\n]*\n.*?^\s*```[^\n]*(?:\n|$)"
)
LIST = re.compile(r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+)")
HEADING = re.compile(r"(?m)^\s{0,3}#{1,6}\s+\S")

METRICS = [
    "characters", "words", "mattr50",
    "has_list", "has_heading", "has_code_fence",
]


def normalize_for_analysis(text):
    # Heuristic restoration for this CSV collection.
    # Keep raw files unchanged; literal escapes may be intentional in code.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\\+n", "\n", text)
    return re.sub(r"\\+t", "\t", text)


def features(text):
    had_escape = bool(re.search(r"\\+[nt]", text))
    text = normalize_for_analysis(text)
    prose = FENCE.sub("", text)

    words = WORD.findall(text.lower())
    prose_words = WORD.findall(prose.lower())

    diversity = [
        len(set(prose_words[i:i + 50])) / 50
        for i in range(len(prose_words) - 49)
    ]

    return {
        "characters": len(text),
        "words": len(words),
        "mattr50": mean(diversity) if diversity else None,
        "has_list": int(bool(LIST.search(prose))),
        "has_heading": int(bool(HEADING.search(prose))),
        "has_code_fence": int(bool(
            re.search(r"(?m)^\s*```", text)
        )),
        "had_literal_escape": int(had_escape),
    }


def write_csv(filename, rows):
    if not rows:
        return
    with (DEST / filename).open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(records, group_columns):
    groups = defaultdict(list)
    for row in records:
        key = tuple(row[column] for column in group_columns)
        groups[key].append(row)

    summaries = []
    for key, group in sorted(groups.items()):
        result = dict(zip(group_columns, key))
        result["responses"] = len(group)

        for metric in METRICS:
            values = [
                row[metric] for row in group
                if row[metric] is not None
            ]
            result[f"mean_{metric}"] = mean(values) if values else None

        result["mattr50_eligible"] = sum(
            row["mattr50"] is not None for row in group
        )
        result["literal_escape_responses"] = sum(
            row["had_literal_escape"] for row in group
        )
        summaries.append(result)

    return summaries


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    records = []
    seen = set()
    prompt_splits = {}

    for split in ("train", "validation", "test"):
        with (SOURCE / f"{split}.csv").open(
            encoding="utf-8-sig", newline=""
        ) as file:
            for row in csv.DictReader(file):
                key = (row["prompt_id"], row["LLM_name"])
                if key in seen:
                    raise ValueError(f"Duplicate response: {key}")
                seen.add(key)

                previous = prompt_splits.setdefault(
                    row["prompt_id"], split
                )
                if previous != split:
                    raise ValueError("Prompt overlap across splits")

                records.append({
                    "response_id": row["response_id"],
                    "prompt_id": row["prompt_id"],
                    "LLM_name": row["LLM_name"],
                    "category": row["category"],
                    "split": split,
                    **features(row["LLM_output"]),
                })

    # Ensure every prompt has one response from each model.
    by_prompt = defaultdict(dict)
    for row in records:
        by_prompt[row["prompt_id"]][row["LLM_name"]] = row

    for prompt_id, group in by_prompt.items():
        if set(group) != set(MODELS):
            raise ValueError(f"Incomplete group: {prompt_id}")

    overall = summarize(records, ["LLM_name"])
    write_csv("response_features.csv", records)
    write_csv("model_summary.csv", overall)
    write_csv(
        "category_summary.csv",
        summarize(records, ["category", "LLM_name"]),
    )
    write_csv(
        "split_summary.csv",
        summarize(records, ["split", "LLM_name"]),
    )

    # Matched comparisons: each difference uses the same prompt.
    # Diversity only uses prompts where both answers have >=50 prose words.
    paired = []
    for first, second in (
        ("GPT", "Claude"),
        ("GPT", "DeepSeek"),
        ("Claude", "DeepSeek"),
    ):
        for metric in METRICS:
            differences = [
                group[first][metric] - group[second][metric]
                for group in by_prompt.values()
                if group[first][metric] is not None
                and group[second][metric] is not None
            ]
            paired.append({
                "first_model": first,
                "second_model": second,
                "metric": metric,
                "paired_prompts": len(differences),
                "mean_difference_first_minus_second":
                    mean(differences) if differences else None,
            })
    write_csv("paired_differences.csv", paired)

    lookup = {row["LLM_name"]: row for row in overall}
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    titles = [
        "Mean characters", "Mean English words", "MATTR-50",
        "Responses with lists (%)",
        "Responses with Markdown headings (%)",
        "Responses with Markdown code fences (%)",
    ]

    for axis, metric, title in zip(
        axes.flat, METRICS, titles
    ):
        values = [
            lookup[model][f"mean_{metric}"] for model in MODELS
        ]
        multiplier = 100 if metric.startswith("has_") else 1
        axis.bar(
            MODELS,
            [value * multiplier if value is not None else 0
             for value in values],
            color=["#4C78A8", "#F58518", "#54A24B"],
        )
        axis.set_title(title)
        if metric == "mattr50":
            axis.set_ylim(0, 1)

    fig.suptitle("RQ4: Descriptive features of collected responses")
    fig.tight_layout()
    fig.savefig(DEST / "rq4_features.png", dpi=200)
    plt.close(fig)

    for row in overall:
        print(row)
    print(f"\nSaved to: {DEST}")


if __name__ == "__main__":
    main()