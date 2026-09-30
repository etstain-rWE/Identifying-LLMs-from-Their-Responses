import csv
import random
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "data"
SEED = 42
MODELS = {"GPT", "Claude", "DeepSeek"}


def main():
    source = BASE_DIR / "raw_responses.csv"

    with source.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, strict=True)
        columns = reader.fieldnames

        required = {"prompt_id", "LLM_name", "LLM_Input",
                    "LLM_output", "category"}
        if not columns or not required.issubset(columns):
            raise ValueError("Missing required columns.")

        rows = list(reader)

    if any(
        None in row or any(value is None for value in row.values())
        for row in rows
    ):
        raise ValueError("Invalid CSV structure.")

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["prompt_id"]].append(row)

    # Validate each prompt's three responses.
    categories = defaultdict(list)

    for prompt_id, responses in grouped.items():
        if not prompt_id.strip():
            raise ValueError("Empty prompt_id.")

        counts = Counter(row["LLM_name"] for row in responses)
        if counts != Counter({model: 1 for model in MODELS}):
            raise ValueError(
                f"{prompt_id}: expected one response per model; "
                f"found {dict(counts)}"
            )

        if len({row["LLM_Input"] for row in responses}) != 1:
            raise ValueError(f"{prompt_id}: inconsistent inputs.")

        category_set = {row["category"] for row in responses}
        if len(category_set) != 1:
            raise ValueError(f"{prompt_id}: inconsistent categories.")

        if any(not row["LLM_output"].strip() for row in responses):
            raise ValueError(f"{prompt_id}: empty response.")

        category = next(iter(category_set))
        if not category.strip():
            raise ValueError(f"{prompt_id}: empty category.")

        categories[category].append(prompt_id)

    # This allocation implements the planned 300-prompt dataset.
    if len(categories) != 6 or any(
        len(ids) != 50 for ids in categories.values()
    ):
        raise ValueError(
            "Expected six categories with 50 unique prompts each. "
            f"Found: {dict((k, len(v)) for k, v in categories.items())}"
        )

    rng = random.Random(SEED)
    assignments = {}

    for index, category in enumerate(sorted(categories)):
        ids = sorted(categories[category])
        rng.shuffle(ids)

        # Each category: 35 training prompts.
        # Alternate 7/8 validation prompts to reach 45 overall.
        n_train = 35
        n_validation = 7 if index % 2 == 0 else 8

        for position, prompt_id in enumerate(ids):
            if position < n_train:
                split = "train"
            elif position < n_train + n_validation:
                split = "validation"
            else:
                split = "test"

            assignments[prompt_id] = split

    split_ids = {
        split: {
            pid for pid, assigned in assignments.items()
            if assigned == split
        }
        for split in ("train", "validation", "test")
    }

    # Explicitly verify that groups do not overlap.
    assert split_ids["train"].isdisjoint(split_ids["validation"])
    assert split_ids["train"].isdisjoint(split_ids["test"])
    assert split_ids["validation"].isdisjoint(split_ids["test"])
    assert set.union(*split_ids.values()) == set(grouped)

    output_dir = BASE_DIR / "splits"
    output_dir.mkdir(exist_ok=True)

    for split in ("train", "validation", "test"):
        selected = [
            row for row in rows
            if assignments[row["prompt_id"]] == split
        ]
        selected.sort(
            key=lambda row: (row["prompt_id"], row["LLM_name"])
        )

        output_path = output_dir / f"{split}.csv"
        with output_path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=columns, quoting=csv.QUOTE_ALL
            )
            writer.writeheader()
            writer.writerows(selected)

        print(
            f"{split}: {len(split_ids[split])} prompts, "
            f"{len(selected)} responses"
        )
        print("  Models:", dict(Counter(
            row["LLM_name"] for row in selected
        )))

    # Save the split assignments for reproducibility.
    with (output_dir / "split_manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=["prompt_id", "category", "split"]
        )
        writer.writeheader()

        for prompt_id in sorted(assignments):
            writer.writerow({
                "prompt_id": prompt_id,
                "category": grouped[prompt_id][0]["category"],
                "split": assignments[prompt_id],
            })

    print(f"\nSplit completed. Random seed: {SEED}")
    print(f"Files saved in: {output_dir}")


if __name__ == "__main__":
    main()