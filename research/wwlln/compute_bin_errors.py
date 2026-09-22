"""Aggregate matched WWLLN distance errors by true 100 km bin."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.lig import LigFileIndex
from evaluate_wwlln import (
    C_KM_S,
    MAX_DISTANCE_KM,
    MAX_TIME_DIFF_S,
    STATION_LAT,
    STATION_LON,
    load_wwlln,
    match_wwlln,
    peak_offset,
    spherical_distance,
)


TYPES = ("NCG", "NNBE", "PCG", "PNBE")
MAX_PEAK_OFFSET_S = 15999 * 0.0000002


def could_match(timestamp, events: list, times: list, distances: dict[int, float]) -> bool:
    """Return whether any event can satisfy time and propagation constraints."""
    lower = bisect.bisect_left(
        times, timestamp - timedelta(seconds=MAX_TIME_DIFF_S)
    )
    upper = bisect.bisect_left(
        times, timestamp + timedelta(seconds=MAX_PEAK_OFFSET_S)
    )
    shifted_max = timestamp + timedelta(seconds=MAX_PEAK_OFFSET_S)
    rounding = timedelta(microseconds=1)
    for index in range(lower, upper):
        actual = distances.get(index)
        if actual is None:
            event = events[index]
            actual = spherical_distance(
                STATION_LAT, STATION_LON, event[1], event[2]
            )
            distances[index] = actual
        if actual > MAX_DISTANCE_KM:
            continue
        event_time = events[index][0]
        allowed_low = event_time + timedelta(seconds=0.9 * actual / C_KM_S)
        allowed_high = event_time + timedelta(
            seconds=min(MAX_TIME_DIFF_S, 1.1 * actual / C_KM_S)
        )
        if allowed_high + rounding >= timestamp and allowed_low - rounding <= shifted_max:
            return True
    return False


@dataclass
class BinStats:
    errors: list[float] = field(default_factory=list)
    bin_errors: list[int] = field(default_factory=list)
    years: Counter[str] = field(default_factory=Counter)
    types: Counter[str] = field(default_factory=Counter)

    def add(
        self,
        error_km: float,
        error_bin: int,
        year: str,
        type_name: str,
    ) -> None:
        self.errors.append(error_km)
        self.bin_errors.append(error_bin)
        self.years[year] += 1
        self.types[type_name] += 1


def load_aliases(manifest_path: Path | None) -> dict[str, dict[str, str]]:
    if manifest_path is None:
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    types = payload.get("types", payload)
    aliases: dict[str, dict[str, str]] = {}
    for type_name, details in types.items():
        inverse = {
            alias: distance
            for distance, alias in details.get("bin_aliases", {}).items()
        }
        aliases[type_name] = inverse
    return aliases


def resolve_source(
    lig_base: Path,
    source_path: str,
    aliases: dict[str, dict[str, str]],
) -> Path:
    parts = Path(source_path).parts
    if len(parts) >= 3 and parts[0] in aliases:
        distance = aliases[parts[0]].get(parts[1])
        if distance is not None:
            return lig_base / parts[0] / distance / Path(*parts[2:])
    if len(parts) >= 3:
        alias_match = re.fullmatch(r"bin_([a-z]+)", parts[1])
        if alias_match:
            number = 0
            for character in alias_match.group(1):
                number = number * 26 + ord(character) - ord("a") + 1
            index = number - 1
            if 0 <= index < 30:
                distance = f"{index * 100:04d}-{(index + 1) * 100:04d}km"
                candidate = lig_base / parts[0] / distance / Path(*parts[2:])
                if candidate.is_file():
                    return candidate
    direct = lig_base / source_path
    if direct.is_file():
        return direct
    date_match = re.search(r"(20\d{6})", source_path)
    if date_match:
        dated = lig_base / f"GZ_{date_match.group(1)}" / source_path
        if dated.is_file():
            return dated
    return direct


def grouped_rows(csv_path: Path):
    seen: set[str] = set()
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        current_path: str | None = None
        current: list[dict[str, str]] = []
        for row in reader:
            source_path = row["source_path"]
            if current_path is None:
                current_path = source_path
            if source_path != current_path:
                if current_path in seen:
                    raise RuntimeError(f"non-contiguous source rows: {csv_path}")
                seen.add(current_path)
                yield current_path, current
                current_path, current = source_path, []
            current.append(row)
        if current_path is not None:
            if current_path in seen:
                raise RuntimeError(f"non-contiguous source rows: {csv_path}")
            yield current_path, current


def grouped_output_rows(csv_path: Path, database_path: Path):
    """Rewrite regrouped output paths to byte positions with bounded memory."""
    if database_path.exists():
        database_path.unlink()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(
            "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; "
            "CREATE TABLE predictions ("
            "output_file TEXT NOT NULL, piece_index INTEGER NOT NULL, "
            "distance_bin INTEGER NOT NULL, expected_distance_km REAL NOT NULL);"
        )
        counters: Counter[str] = Counter()
        pending = []
        with csv_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                output_file = row["output_file"]
                pending.append((
                    output_file,
                    counters[output_file],
                    int(row["distance_bin"]),
                    float(row["expected_distance_km"]),
                ))
                counters[output_file] += 1
                if len(pending) == 10000:
                    connection.executemany(
                        "INSERT INTO predictions VALUES (?, ?, ?, ?)", pending
                    )
                    pending.clear()
        if pending:
            connection.executemany(
                "INSERT INTO predictions VALUES (?, ?, ?, ?)", pending
            )
        connection.execute(
            "CREATE INDEX output_order ON predictions(output_file, piece_index)"
        )
        connection.commit()
        cursor = connection.execute(
            "SELECT output_file, piece_index, distance_bin, expected_distance_km "
            "FROM predictions ORDER BY output_file, piece_index"
        )
        current_path = None
        current = []
        for output_file, piece_index, distance_bin, expected_km in cursor:
            if current_path is None:
                current_path = output_file
            if output_file != current_path:
                yield current_path, current
                current_path, current = output_file, []
            current.append({
                "piece_index": str(piece_index),
                "distance_bin": str(distance_bin),
                "expected_distance_km": str(expected_km),
            })
        if current_path is not None:
            yield current_path, current
    finally:
        connection.close()


def grouped_source_rows_sqlite(csv_path: Path, database_path: Path):
    """Order source-piece predictions without retaining the full CSV in RAM."""
    if database_path.exists():
        database_path.unlink()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(
            "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; "
            "CREATE TABLE predictions ("
            "source_path TEXT NOT NULL, piece_index INTEGER NOT NULL, "
            "distance_bin INTEGER NOT NULL, expected_distance_km REAL NOT NULL);"
        )
        pending = []
        with csv_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                pending.append((
                    row["source_path"],
                    int(row["piece_index"]),
                    int(row["distance_bin"]),
                    float(row["expected_distance_km"]),
                ))
                if len(pending) == 10000:
                    connection.executemany(
                        "INSERT INTO predictions VALUES (?, ?, ?, ?)", pending
                    )
                    pending.clear()
        if pending:
            connection.executemany(
                "INSERT INTO predictions VALUES (?, ?, ?, ?)", pending
            )
        connection.execute(
            "CREATE INDEX source_order ON predictions(source_path, piece_index)"
        )
        connection.commit()
        cursor = connection.execute(
            "SELECT source_path, piece_index, distance_bin, expected_distance_km "
            "FROM predictions ORDER BY source_path, piece_index"
        )
        current_path = None
        current = []
        for source_path, piece_index, distance_bin, expected_km in cursor:
            if current_path is None:
                current_path = source_path
            if source_path != current_path:
                yield current_path, current
                current_path, current = source_path, []
            current.append({
                "piece_index": str(piece_index),
                "distance_bin": str(distance_bin),
                "expected_distance_km": str(expected_km),
            })
        if current_path is not None:
            yield current_path, current
    finally:
        connection.close()


def evaluate_dataset(
    label: str,
    prediction_root: Path,
    lig_base: Path,
    wwlln_dir: Path,
    aliases: dict[str, dict[str, str]],
    bins: list[BinStats],
    type_bins: dict[str, list[BinStats]],
    row_source: str = "source_path",
    database_dir: Path | None = None,
) -> dict[str, object]:
    cache: dict[str, tuple[list, list, dict[int, float]] | None] = {}
    predicted = matched = missing_sources = eligible = 0
    type_matches: Counter[str] = Counter()
    errors: list[float] = []
    for type_name in TYPES:
        csv_path = prediction_root / type_name / f"{type_name}_distance_predictions.csv"
        if not csv_path.is_file():
            continue
        print(f"{label} {type_name}: {csv_path}", flush=True)
        if row_source == "output_file":
            if database_dir is None:
                raise ValueError("database_dir is required for output_file rows")
            groups = grouped_output_rows(
                csv_path, database_dir / f"{label}_{type_name}.sqlite"
            )
        elif row_source == "source_path_sqlite":
            if database_dir is None:
                raise ValueError("database_dir is required for sqlite rows")
            groups = grouped_source_rows_sqlite(
                csv_path, database_dir / f"{label}_{type_name}.sqlite"
            )
        else:
            groups = grouped_rows(csv_path)
        for source_path, rows in groups:
            predicted += len(rows)
            source = resolve_source(lig_base, source_path, aliases)
            if not source.is_file():
                missing_sources += len(rows)
                continue
            date_match = re.search(r"(20\d{6})", source_path)
            if not date_match:
                continue
            date = date_match.group(1)
            if date not in cache:
                loc = wwlln_dir / f"AE{date}.loc"
                if loc.is_file():
                    events = load_wwlln(loc)
                    cache[date] = (events, [event[0] for event in events], {})
                else:
                    cache[date] = None
            cached = cache[date]
            if cached is None:
                continue
            events, times, distance_cache = cached
            row_by_piece = {int(row["piece_index"]): row for row in rows}
            with LigFileIndex([source], validate=True) as index:
                for start in range(0, len(index), 256):
                    stop = min(start + 256, len(index))
                    positions = list(range(start, stop))
                    timestamps = index.read_timestamps_batch(
                        positions, allow_invalid=True
                    )
                    selected = [
                        (position, row_by_piece[position], timestamp)
                        for position, timestamp in zip(positions, timestamps)
                        if position in row_by_piece
                    ]
                    eligible += len(selected)
                    candidates = [
                        item for item in selected
                        if item[2] is not None
                        and could_match(item[2], events, times, distance_cache)
                    ]
                    if not candidates:
                        continue
                    waveforms = index.read_pieces_batch(
                        item[0] for item in candidates
                    )
                    for (_, row, timestamp), waveform in zip(candidates, waveforms):
                        try:
                            shifted = timestamp + timedelta(
                                seconds=peak_offset(waveform)
                            )
                        except Exception:
                            shifted = timestamp
                        match = match_wwlln(shifted, events, times)
                        if match is None:
                            continue
                        true_km = float(match[2])
                        pred_km = float(row["expected_distance_km"])
                        pred_bin = int(row["distance_bin"])
                        true_bin = min(int(true_km // 100), 29)
                        error_km = pred_km - true_km
                        bins[true_bin].add(
                            error_km,
                            pred_bin - true_bin,
                            label[:4],
                            type_name,
                        )
                        type_bins[type_name][true_bin].add(
                            error_km,
                            pred_bin - true_bin,
                            label[:4],
                            type_name,
                        )
                        errors.append(error_km)
                        matched += 1
                        type_matches[type_name] += 1
        print(f"  matched so far: {matched}", flush=True)
    values = np.asarray(errors, dtype=np.float64)
    return {
        "label": label,
        "predicted_rows": predicted,
        "eligible_rows": eligible,
        "missing_source_rows": missing_sources,
        "matched": matched,
        "mae_km": float(np.abs(values).mean()) if matched else None,
        "rmse_km": float(np.sqrt(np.square(values).mean())) if matched else None,
        "bias_km": float(values.mean()) if matched else None,
        "type_matches": dict(type_matches),
    }


def bin_rows(bins: list[BinStats]) -> list[dict[str, object]]:
    output = []
    for index, stats in enumerate(bins):
        values = np.asarray(stats.errors, dtype=np.float64)
        bin_values = np.asarray(stats.bin_errors, dtype=np.int64)
        count = int(values.size)
        output.append({
            "true_bin": index,
            "distance_bin": f"{index * 100}-{(index + 1) * 100}",
            "distance_low_km": index * 100,
            "distance_high_km": (index + 1) * 100,
            "matched_count": count,
            "mae_km": float(np.abs(values).mean()) if count else None,
            "rmse_km": float(np.sqrt(np.square(values).mean())) if count else None,
            "bias_km": float(values.mean()) if count else None,
            "median_abs_error_km": float(np.median(np.abs(values))) if count else None,
            "exact_bin_accuracy": float(np.mean(bin_values == 0)) if count else None,
            "within_1_bin": float(np.mean(np.abs(bin_values) <= 1)) if count else None,
            "within_2_bins": float(np.mean(np.abs(bin_values) <= 2)) if count else None,
            "year_2016_count": stats.years["2016"],
            "year_2017_count": stats.years["2017"],
            "year_2021_count": stats.years["2021"],
            "NCG_count": stats.types["NCG"],
            "NNBE_count": stats.types["NNBE"],
            "PCG_count": stats.types["PCG"],
            "PNBE_count": stats.types["PNBE"],
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    wwlln_dir = Path(config["wwlln_dir"])
    bins = [BinStats() for _ in range(30)]
    type_bins = {
        type_name: [BinStats() for _ in range(30)]
        for type_name in TYPES
    }
    database_dir = args.output.parent / "bin_error_databases"
    dataset_summaries = []
    for dataset in config["datasets"]:
        manifest = dataset.get("manifest")
        dataset_summaries.append(evaluate_dataset(
            dataset["label"],
            Path(dataset["prediction_root"]),
            Path(dataset["lig_base"]),
            wwlln_dir,
            load_aliases(Path(manifest) if manifest else None),
            bins,
            type_bins,
            dataset.get("row_source", "source_path"),
            database_dir,
        ))
    rows = bin_rows(bins)
    total = sum(row["matched_count"] for row in rows)
    if total != sum(item["matched"] for item in dataset_summaries):
        raise RuntimeError("matched sample count does not reconcile")
    type_rows = {
        type_name: bin_rows(type_stats)
        for type_name, type_stats in type_bins.items()
    }
    for type_name, rows_for_type in type_rows.items():
        expected = sum(
            item["type_matches"].get(type_name, 0)
            for item in dataset_summaries
        )
        if sum(row["matched_count"] for row in rows_for_type) != expected:
            raise RuntimeError(f"{type_name} sample count does not reconcile")
    payload = {
        "schema": "wwlln_true_distance_bin_error_v1",
        "metric": "mean(abs(expected_distance_km - wwlln_distance_km))",
        "grouping": "WWLLN true distance, 100 km bins",
        "matching": {
            "nearest_previous_event_only": True,
            "max_time_difference_ms": 11,
            "propagation_consistency_tolerance": 0.10,
            "one_to_one": False,
        },
        "datasets": dataset_summaries,
        "matched_total": total,
        "bins": rows,
        "type_bins": type_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dataset_summaries, ensure_ascii=False, indent=2), flush=True)
    print(f"matched total: {total}", flush=True)
    print(f"wrote: {args.output}", flush=True)


if __name__ == "__main__":
    main()
