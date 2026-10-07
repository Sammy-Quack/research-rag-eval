#!/usr/bin/env python3
"""List evaluation records whose metric scores are exactly zero."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


RECORD_LIST_KEYS = ("results", "evaluations", "data", "items", "per_eval")


def load_records(path: Path) -> list[Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        for key in RECORD_LIST_KEYS:
            if isinstance(data.get(key), list):
                data = data[key]
                break

    return data if isinstance(data, list) else []


def find_zero_metrics(directory: Path) -> list[dict[str, str]]:
    matches = []

    for path in sorted(directory.glob("*.json")):
        try:
            records = load_records(path)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read {path.name}: {error}") from error

        for record in records:
            if not isinstance(record, dict):
                continue

            scores = record.get("scores")
            if not isinstance(scores, dict):
                continue

            eval_id = record.get("eval_id", record.get("id", ""))
            for metric, value in scores.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0:
                    matches.append({
                        "parent_json_file": path.name,
                        "eval_id": str(eval_id),
                        "metric": str(metric),
                    })

    return matches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find evaluation metrics with a score of exactly 0.0."
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing JSON files (default: this script's directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path (default: <directory>/zero_metric_results.csv)",
    )
    args = parser.parse_args()

    directory = args.directory.resolve()
    if not directory.is_dir():
        parser.error(f"Not a directory: {directory}")

    output = args.output or directory / "zero_metric_results.csv"
    matches = find_zero_metrics(directory)

    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("parent_json_file", "eval_id", "metric"),
        )
        writer.writeheader()
        writer.writerows(matches)

    print(f"Found {len(matches)} zero-valued metric entries; report: {output}")


if __name__ == "__main__":
    main()
