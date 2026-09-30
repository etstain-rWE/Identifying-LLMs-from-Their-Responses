import csv
from pathlib import Path
from collections import Counter

# Locate CSV files relative to this script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "data"

MODEL_FILES = {
    "GPT": "gpt_responses.csv",
    "Claude": "claude_responses.csv",
    "DeepSeek": "deepseek_responses.csv",
}

# Optional: fill these in using the model names shown in the interfaces.
# Leave blank if the exact versions were not recorded.
MODEL_VERSIONS = {
    "GPT": "",
    "Claude": "",
    "DeepSeek": "",
}


def read_csv(filename, required_columns):
    path = BASE_DIR / filename

    if not path.exists():
        raise ValueError(f"File not found: {path}")

    # utf-8-sig also handles CSV files with a UTF-8 BOM.
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, strict=True)

        if not reader.fieldnames:
            raise ValueError(f"{filename}: missing header")

        headers = reader.fieldnames
        if len(headers) != len(set(headers)):
            raise ValueError(f"{filename}: duplicate column names")

        missing = set(required_columns) - set(headers)
        if missing:
            raise ValueError(
                f"{filename}: missing columns {sorted(missing)}"
            )

        rows = []
        for row_number, row in enumerate(reader, start=2):
            # Catch extra or missing CSV fields.
            if None in row or any(value is None for value in row.values()):
                raise ValueError(
                    f"{filename}, record {row_number}: "
                    "incorrect number of fields; check CSV quoting"
                )
            rows.append(row)

    if not rows:
        raise ValueError(f"{filename}: no data rows")

    return rows


def index_by_prompt(rows, filename):
    result = {}

    for row in rows:
        prompt_id = row["prompt_id"].strip()

        if not prompt_id:
            raise ValueError(f"{filename}: empty prompt_id")

        if prompt_id in result:
            raise ValueError(
                f"{filename}: duplicate prompt_id {prompt_id}"
            )

        result[prompt_id] = row

    return result


def main():
    prompts = read_csv(
        "prompts.csv",
        [
            "prompt_id",
            "category",
            "task_type",
            "difficulty",
            "prompt_text",
        ],
    )
    prompt_lookup = index_by_prompt(prompts, "prompts.csv")
    expected_ids = set(prompt_lookup)

    for prompt_id, row in prompt_lookup.items():
        if not row["prompt_text"].strip():
            raise ValueError(
                f"prompts.csv: empty prompt_text for {prompt_id}"
            )

    combined = []

    for model_name, filename in MODEL_FILES.items():
        rows = read_csv(
            filename,
            ["prompt_id", "LLM_Input", "LLM_output"],
        )
        response_lookup = index_by_prompt(rows, filename)

        missing_ids = expected_ids - set(response_lookup)
        extra_ids = set(response_lookup) - expected_ids

        if missing_ids or extra_ids:
            raise ValueError(
                f"{filename}:\n"
                f"  Missing prompt IDs: {sorted(missing_ids)}\n"
                f"  Unexpected prompt IDs: {sorted(extra_ids)}"
            )

        for prompt_id, prompt in prompt_lookup.items():
            response = response_lookup[prompt_id]

            # Compare exact text; do not silently replace changed inputs.
            if response["LLM_Input"] != prompt["prompt_text"]:
                raise ValueError(
                    f"{filename}: LLM_Input does not exactly match "
                    f"prompt_text for {prompt_id}"
                )

            if not response["LLM_output"].strip():
                raise ValueError(
                    f"{filename}: empty answer for {prompt_id}"
                )

            # Metadata columns are optional in response files.
            # If present, their values must match prompts.csv.
            for column in ("category", "task_type", "difficulty"):
                if (
                    column in response
                    and response[column] != prompt[column]
                ):
                    raise ValueError(
                        f"{filename}: mismatched {column} "
                        f"for {prompt_id}"
                    )

            combined.append({
                "prompt_id": prompt_id,
                "LLM_name": model_name,
                "model_version": MODEL_VERSIONS[model_name],
                "LLM_Input": response["LLM_Input"],
                "LLM_output": response["LLM_output"],
                "category": prompt["category"],
                "task_type": prompt["task_type"],
                "difficulty": prompt["difficulty"],
            })

    # Arrange each prompt's three answers together.
    model_order = {
        name: index for index, name in enumerate(MODEL_FILES)
    }
    combined.sort(
        key=lambda row: (
            row["prompt_id"],
            model_order[row["LLM_name"]],
        )
    )

    output_rows = [
        {"response_id": f"R{index:04d}", **row}
        for index, row in enumerate(combined, start=1)
    ]

    columns = [
        "response_id",
        "prompt_id",
        "LLM_name",
        "model_version",
        "LLM_Input",
        "LLM_output",
        "category",
        "task_type",
        "difficulty",
    ]

    output_path = BASE_DIR / "raw_responses.csv"
    with output_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print("Validation passed. Merge completed.")
    print(f"Unique prompts: {len(prompt_lookup)}")
    print(f"Total responses: {len(output_rows)}")

    for model_name, count in Counter(
        row["LLM_name"] for row in output_rows
    ).items():
        print(f"{model_name}: {count}")

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, csv.Error, UnicodeError) as error:
        print(f"\nMerge stopped:\n{error}")
        raise SystemExit(1)