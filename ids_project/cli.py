"""CLI: python -m ids_project.cli {inspect,train,evaluate,predict}."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .core import DatasetLoader, ModelTrainer, PredictionService


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline UNSW-NB15 intrusion detection research prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Validate and describe a labeled CSV")
    inspect.add_argument("--file", type=Path, required=True)
    train = sub.add_parser("train", help="Train and validate on official TRAINING CSV only")
    train.add_argument("--train", type=Path, required=True)
    train.add_argument("--out", type=Path, default=Path("runs"))
    train.add_argument("--seed", type=int, default=42)
    for mode in ("evaluate", "predict"):
        p = sub.add_parser(mode, help=f"{mode.title()} using a saved trusted model")
        p.add_argument("--model", type=Path, required=True, help="Directory holding model.joblib and run_config.json")
        p.add_argument("--input", type=Path, required=True)
        p.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            ds = DatasetLoader().load(args.file, labels_required=True)
            print(json.dumps({"rows": len(ds.table), "columns": len(ds.table.columns), "sha256": ds.checksum, "class_distribution": ds.table["label"].value_counts().to_dict()}, ensure_ascii=False, indent=2))
        elif args.command == "train":
            print("TRAIN RUN DIRECTORY:", ModelTrainer().run(args.train, args.out, seed=args.seed))
        else:
            print("RESULT DIRECTORY:", PredictionService.run(args.model, args.input, args.out, require_labels=(args.command == "evaluate")))
    except (ValueError, FileNotFoundError) as e:
        parser.exit(status=2, message=f"ERROR: {e}\n")


if __name__ == "__main__":
    main()
