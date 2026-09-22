"""Replace one disjoint dataset contribution in additive WWLLN bin summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


WEIGHTED_FIELDS = (
    "mae_km",
    "bias_km",
    "exact_bin_accuracy",
    "within_1_bin",
    "within_2_bins",
)
COUNT_FIELDS = (
    "year_2016_count",
    "year_2017_count",
    "year_2021_count",
    "NCG_count",
    "NNBE_count",
    "PCG_count",
    "PNBE_count",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--remove", type=Path, required=True)
    parser.add_argument("--add", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    combined = json.loads(args.combined.read_text(encoding="utf-8"))
    removed = json.loads(args.remove.read_text(encoding="utf-8"))
    added = json.loads(args.add.read_text(encoding="utf-8"))
    combined["datasets"] = [
        item for item in combined["datasets"] if item["label"] != args.label
    ] + added["datasets"]
    rows = []
    for current, old, new in zip(
        combined["bins"], removed["bins"], added["bins"], strict=True
    ):
        if len({current["true_bin"], old["true_bin"], new["true_bin"]}) != 1:
            raise RuntimeError("distance bins do not align")
        counts = [
            int(current["matched_count"]),
            int(old["matched_count"]),
            int(new["matched_count"]),
        ]
        total = counts[0] - counts[1] + counts[2]
        row = dict(current)
        row["matched_count"] = total
        for field in WEIGHTED_FIELDS:
            numerator = sum(
                sign * count * float(source[field])
                for sign, count, source in (
                    (1, counts[0], current),
                    (-1, counts[1], old),
                    (1, counts[2], new),
                )
                if count
            )
            row[field] = numerator / total if total else None
        squared = sum(
            sign * count * float(source["rmse_km"]) ** 2
            for sign, count, source in (
                (1, counts[0], current),
                (-1, counts[1], old),
                (1, counts[2], new),
            )
            if count
        )
        row["rmse_km"] = math.sqrt(max(0.0, squared / total)) if total else None
        row["median_abs_error_km"] = None
        for field in COUNT_FIELDS:
            row[field] = (
                int(current[field]) - int(old[field]) + int(new[field])
            )
        rows.append(row)
    # Only aggregate bin totals; inherited per-type summaries would be stale.
    combined.pop("type_bins", None)
    combined["bins"] = rows
    combined["matched_total"] = sum(row["matched_count"] for row in rows)
    combined["aggregation_note"] = (
        "All additive metrics are exact; combined per-bin median is omitted."
    )
    if combined["matched_total"] != sum(
        item["matched"] for item in combined["datasets"]
    ):
        raise RuntimeError("replaced sample count does not reconcile")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with args.output.with_suffix(".csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
