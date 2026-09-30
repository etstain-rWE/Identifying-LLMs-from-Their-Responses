import csv
import json
from pathlib import Path

import train as training

ROOT = Path(__file__).resolve().parent.parent
MODES = ("input_only", "input_output")


def write_csv(path, rows):
    with path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    rq2_summaries = []

    for mode in MODES:
        data_dir = ROOT / "data" / "processed" / mode
        config = training.read_json(data_dir / "config.json")

        class_names = [
            name for name, label in sorted(
                config["label_mapping"].items(),
                key=lambda item: item[1],
            )
        ]

        datasets = {
            split: training.TextDataset(training.read_json(
                data_dir / f"{split}_output.json"
            ))
            for split in ("train", "validation", "test")
        }

        # Separate directories preserve the original Output-only results.
        training.MODEL_DIR = ROOT / "models" / mode
        training.RESULT_DIR = ROOT / "results" / mode
        training.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        training.RESULT_DIR.mkdir(parents=True, exist_ok=True)

        print(f"\n========== {mode} ==========")
        print(f"Device: {training.DEVICE}")

        summaries = []

        for model_name in ("CNN", "LSTM"):
            result = training.train_model(
                model_name, config, datasets, class_names
            )
            result["input_mode"] = mode
            summaries.append(result)
            rq2_summaries.append(result)

            # The reused function uses Output-only in its filenames.
            # Rename checkpoints to reflect the current experiment.
            old_checkpoint = (
                training.MODEL_DIR
                / f"{model_name.lower()}_output_only.pt"
            )
            new_checkpoint = (
                training.MODEL_DIR
                / f"{model_name.lower()}_{mode}.pt"
            )
            old_checkpoint.replace(new_checkpoint)

            # Correct titles in figures produced by the reused function.
            import matplotlib.pyplot as plt
            from sklearn.metrics import (
                confusion_matrix, ConfusionMatrixDisplay
            )

            history_path = (
                training.RESULT_DIR / f"{model_name}_history.csv"
            )
            with history_path.open(
                encoding="utf-8-sig", newline=""
            ) as file:
                history = list(csv.DictReader(file))

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            epochs = [int(row["epoch"]) for row in history]

            for axis, metric in zip(axes, ("loss", "accuracy")):
                for prefix, label in (
                    ("train", "Training"), ("val", "Validation")
                ):
                    axis.plot(
                        epochs,
                        [float(row[f"{prefix}_{metric}"])
                         for row in history],
                        label=label,
                    )
                axis.set_xlabel("Epoch")
                axis.set_ylabel(metric.title())
                axis.legend()

            fig.suptitle(f"{model_name}: {mode}")
            fig.tight_layout()
            fig.savefig(
                training.RESULT_DIR
                / f"{model_name}_training_curves.png",
                dpi=200,
            )
            plt.close(fig)

            predictions_path = (
                training.RESULT_DIR / f"{model_name}_predictions.csv"
            )
            with predictions_path.open(
                encoding="utf-8-sig", newline=""
            ) as file:
                predictions = list(csv.DictReader(file))

            matrix = confusion_matrix(
                [row["actual"] for row in predictions],
                [row["predicted"] for row in predictions],
                labels=class_names,
            )
            fig, axis = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(
                matrix, display_labels=class_names
            ).plot(ax=axis, cmap="Blues", colorbar=False)
            axis.set_title(f"{model_name}: {mode} Test")
            fig.tight_layout()
            fig.savefig(
                training.RESULT_DIR
                / f"{model_name}_confusion_matrix.png",
                dpi=200,
            )
            plt.close(fig)

        write_csv(training.RESULT_DIR / "summary.csv", summaries)

        settings = {
            "input_mode": mode,
            "seed": training.SEED,
            "batch_size": training.BATCH_SIZE,
            "maximum_epochs": training.EPOCHS,
            "patience": training.PATIENCE,
            "learning_rate": training.LEARNING_RATE,
            "checkpoint_selection": "minimum validation loss",
            "preprocessing": config,
            "torch_version": training.torch.__version__,
        }
        (training.RESULT_DIR / "experiment_config.json").write_text(
            json.dumps(settings, indent=2), encoding="utf-8"
        )

    # Include the two completed Output-only experiments.
    output_summary = ROOT / "results" / "output_only" / "summary.csv"
    if output_summary.exists():
        with output_summary.open(
            encoding="utf-8-sig", newline=""
        ) as file:
            rq2_summaries.extend(csv.DictReader(file))

    comparison_path = ROOT / "results" / "rq2_comparison.csv"
    write_csv(comparison_path, rq2_summaries)

    print("\nRQ2 results:")
    for row in rq2_summaries:
        print(
            f"{row['model']:4s} | {row['input_mode']:12s} | "
            f"Accuracy={float(row['test_accuracy']):.4f} | "
            f"Macro-F1={float(row['test_macro_f1']):.4f}"
        )

    print(f"\nSaved comparison: {comparison_path}")


if __name__ == "__main__":
    main()