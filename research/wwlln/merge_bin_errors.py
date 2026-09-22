"""Merge exact additive WWLLN bin summaries from disjoint periods."""

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
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extra", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    extra = json.loads(args.extra.read_text(encoding="utf-8"))
    base["datasets"] = [
        item for item in base["datasets"] if item["matched"] > 0
    ] + extra["datasets"]
    rows = []
    for left, right in zip(base["bins"], extra["bins"], strict=True):
        if left["true_bin"] != right["true_bin"]:
            raise RuntimeError("distance bins do not align")
        row = dict(left)
        left_count = int(left["matched_count"])
        right_count = int(right["matched_count"])
        total = left_count + right_count
        row["matched_count"] = total
        for field in WEIGHTED_FIELDS:
            numerator = 0.0
            if left_count:
                numerator += float(left[field]) * left_count
            if right_count:
                numerator += float(right[field]) * right_count
            row[field] = numerator / total if total else None
        squared = 0.0
        if left_count:
            squared += float(left["rmse_km"]) ** 2 * left_count
        if right_count:
            squared += float(right["rmse_km"]) ** 2 * right_count
        row["rmse_km"] = math.sqrt(squared / total) if total else None
        row["median_abs_error_km"] = None
        for field in COUNT_FIELDS:
            row[field] = int(left[field]) + int(right[field])
        rows.append(row)
    # Only aggregate bin totals; inherited per-type summaries would be stale.
    base.pop("type_bins", None)
    base["bins"] = rows
    base["matched_total"] = sum(row["matched_count"] for row in rows)
    base["aggregation_note"] = (
        "All additive metrics are exact; combined per-bin median is omitted."
    )
    if base["matched_total"] != sum(item["matched"] for item in base["datasets"]):
        raise RuntimeError("merged sample count does not reconcile")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with args.output.with_suffix(".csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
